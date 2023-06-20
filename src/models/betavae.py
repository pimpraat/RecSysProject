import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.core import IRecommender
from src.dataset import NBRDatasetBase
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import optuna

Tensor = TypeVar('torch.tensor')

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class PredictorVAE(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        hidden_dims: list = [256, 512, 1024],
        output_size: int = 13674,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.output_size = output_size
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(0.25))
            )
            input_size = h_dim

        self.predictor = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.output_size),
            nn.Tanh())


    def forward(self, input_t: Tensor, **kwargs) -> Tensor:
        result = self.predictor(input_t)
        result = self.final_layer(result)
        return result


class BetaVAERecommender(BaseVAE, IRecommender):

    def __init__(
        self,
        num_nearest_neighbors: int = 100,
        within_decay_rate: float = 0.9,
        group_decay_rate: float = 0.7,
        alpha: float = 0.7,
        group_count: int = 7,
        similarity_measure: str = 'euclidean',
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.group_count = group_count

        self._user_vectors = None
        self._nbrs = None

        self.similarity_measure = similarity_measure

        self.latent_dim = 128
        self.beta = 4
        self.hidden_dims = None

        self.encoder = None
        self.fc_mu = None
        self.fc_var = None
        self.decoder_input = None
        self.decoder = None
        self.final_layer = None

        self.lr = 0.005
        self.weight_decay = 0.0
        self.kld_weight = 0.015

        self.predictor = None
        self.predictor_loss_fn = None
        self.all_user_vecs = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x I]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]


    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def forward(self, input_t: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input_t)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input_t, mu, log_var]


    def loss_function(self, args, kld_weight) -> dict:
        recons = args[0]
        input_t = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input_t)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.beta * kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
    
    
    def prepare_user_vecs(self, dataset, user_basket_df):
        df = user_basket_df.explode("basket", ignore_index=True).rename(columns={"basket": "item_id"})
        df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("weight", "sum"))
        _user_vectors = sps.csr_matrix(
            (df.value, (df.user_id, df.item_id)),
            shape=(dataset.num_users, dataset.num_items),
        )

        total_items = _user_vectors.shape[1]
        return _user_vectors, total_items


    def validate_vae(self, val_user_vectors):
        total_users = val_user_vectors.shape[0]
        batch_size = 32
        loss_total = 0.0
        recon_loss_total = 0.0
        kld_loss_total = 0.0
        iters = 0

        self.eval()
        with torch.no_grad():
            for i in range(0, total_users, batch_size):
                batch = torch.tensor(val_user_vectors[i:i+batch_size].todense(), dtype=torch.float)
                results = self.forward(batch.to(self.device))
                val_loss = self.loss_function(results, self.kld_weight)
                loss_total += val_loss['loss']
                recon_loss_total += val_loss['Reconstruction_Loss']
                kld_loss_total += val_loss['KLD']
                iters += 1
        
        agg_loss = {'loss': loss_total, 'Reconstruction_Loss': recon_loss_total, 'KLD': kld_loss_total}
        return agg_loss


    def validate_predictor(self, user_vectors, next_basket):
        self.eval()
        self.predictor.eval()
        user_latent_reps = []
        batch_size = 32
        total_users = user_vectors.shape[0]
        loss_total = 0
        iters = 0

        with torch.no_grad():
            for i in range(0, total_users, batch_size):
                batch = torch.tensor(user_vectors[i:i+batch_size].todense(), dtype=torch.float).to(self.device)
                mu, log_var = self.encode(batch)
                z = self.reparameterize(mu, log_var)
                user_latent_reps.append(z)

            user_latent_reps = torch.cat(user_latent_reps)

            for i in range(0, total_users, batch_size):
                predict = self.predictor.forward(user_latent_reps[i:i+batch_size].to(self.device))
                # predict = softmaxer(predict)

                target_basket = torch.tensor(next_basket[i:i+batch_size].todense(), dtype=torch.float).to(self.device)
                loss = self.predictor_loss_fn(predict, target_basket)

                loss_total += loss.item()
                iters += 1
        return loss_total
    

    def fit(self, dataset: NBRDatasetBase):
        # Prepare train dataset
        user_basket_df = dataset.train_df.groupby("user_id", as_index=False).apply(self._calculate_basket_weight)
        user_basket_df.reset_index(drop=True, inplace=True)
        self._user_vectors, train_total_items = self.prepare_user_vecs(dataset, user_basket_df)
        input_size = train_total_items

        # Split into train-val dataset
        train_user_vectors, val_user_vectors = train_test_split(self._user_vectors, test_size=0.16, random_state=42)

        print("Train total items: ", train_total_items)
        print("Train total users: ", train_user_vectors.shape[0])
        print("Val total users: ", val_user_vectors.shape[0])
        
        # Build Encoder
        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512, 256]

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            input_size = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1])
        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], train_total_items),
            nn.Tanh())
        
        # Training VAE
        print("Training VAE")

        n_epochs = 25
        batch_size = 32

        optimizer = optim.Adam(self.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

        self.to(self.device)

        total_users = train_user_vectors.shape[0]

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total parameters: ", total_params)

        for epoch in range(n_epochs):
            self.train()
            print(f"Training epoch {epoch}")

            for i in tqdm(range(0, total_users, batch_size)):
                optimizer.zero_grad()
                batch = torch.tensor(train_user_vectors[i:i+batch_size].todense(), dtype=torch.float)
                results = self.forward(batch.to(self.device))
                train_loss = self.loss_function(results, self.kld_weight)
                train_loss['loss'].backward()
                optimizer.step()

            val_loss = self.validate_vae(val_user_vectors)
            scheduler.step(val_loss['loss'].item())
            print("Epoch: {}, Loss: {}, ReconLoss: {}, KLD: {}".format(epoch, val_loss['loss'].item(), 
                val_loss['Reconstruction_Loss'].item(), val_loss['KLD'].item()))


        # Train the predictor

        print("Training Predictor")
        self.predictor = PredictorVAE(input_size=self.latent_dim, hidden_dims=self.hidden_dims, output_size=train_total_items)
        self.predictor.to(self.device)

        # val_user_ids = list(dataset.val_df.user_id)
        # val_user_basket_df = dataset.val_df.groupby("user_id", as_index=False).apply(self._calculate_basket_weight)
        # val_user_basket_df.reset_index(drop=True, inplace=True)
        # val_user_vectors_true, _ = self.prepare_user_vecs(dataset, val_user_basket_df)
        # val_user_vectors_true = val_user_vectors_true[val_user_ids]

        all_user_ids = list(dataset.test_df.user_id)
        train_user_ids, test_user_ids = train_test_split(all_user_ids, test_size=0.3, random_state=42)

        next_user_basket_df = dataset.test_df.groupby("user_id", as_index=False).apply(self._calculate_basket_weight)
        next_user_basket_df.reset_index(drop=True, inplace=True)
        next_user_vectors_true, _ = self.prepare_user_vecs(dataset, next_user_basket_df)
        
        
        test_user_vectors_true = next_user_vectors_true[test_user_ids]
        train_user_vectors_true = next_user_vectors_true[train_user_ids]

        user_vectors_train = self._user_vectors[train_user_ids]
        user_vectors_test = self._user_vectors[test_user_ids]

        total_users = user_vectors_train.shape[0]
        batch_size = 32
        n_epochs = 50
        lr_predictor = 0.1
        
        predictor_optimizer = optim.Adam(self.predictor.parameters(),
                               lr=lr_predictor,
                               weight_decay=self.weight_decay)
        predictor_scheduler = ReduceLROnPlateau(predictor_optimizer, factor=0.5, patience=1)
        # softmaxer = nn.LogSoftmax(dim=1)
        self.predictor_loss_fn = nn.MSELoss()
        # self.predictor_loss_fn = nn.CrossEntropyLoss()
        
        self.eval()

        for epoch in range(n_epochs):
            self.predictor.train()
            user_latent_reps = []
            train_loss_total = 0
            iters = 0

            print(f"Training epoch {epoch}")

            with torch.no_grad():
                for i in range(0, total_users, batch_size):
                    batch = torch.tensor(user_vectors_train[i:i+batch_size].todense(), dtype=torch.float)
                    mu, log_var = self.encode(batch.to(self.device))
                    z = self.reparameterize(mu, log_var)
                    user_latent_reps.append(z)

            user_latent_reps = torch.cat(user_latent_reps).cpu()

            self._nbrs = NearestNeighbors(
                n_neighbors=self.num_nearest_neighbors + 1, # TODO: Why +1?
                algorithm="brute",
                metric=self.similarity_measure
            ).fit(user_latent_reps)

            user_nn_indices = self._nbrs.kneighbors(user_latent_reps, return_distance=False)
            user_nn_vectors = []
            for nn_indices in user_nn_indices:
                nn_vector = user_latent_reps[nn_indices[1:], :].mean(dim=0)
                user_nn_vectors.append(nn_vector)
            user_nn_vectors = torch.stack(user_nn_vectors)

            user_combined_vec = self.alpha * user_latent_reps + (1 - self.alpha) * user_nn_vectors
            
            for i in tqdm(range(0, total_users, batch_size)):
                predictor_optimizer.zero_grad()
                predict = self.predictor.forward(user_combined_vec[i:i+batch_size].to(self.device))
                # predict = softmaxer(predict)

                target_basket = torch.tensor(train_user_vectors_true[i:i+batch_size].todense(), dtype=torch.float).to(self.device)
                train_loss = self.predictor_loss_fn(predict, target_basket)
                train_loss.backward()
                predictor_optimizer.step()

                train_loss_total += train_loss.item()
                iters += 1

            val_loss = self.validate_predictor(user_vectors_test, test_user_vectors_true)
            predictor_scheduler.step(val_loss)
            print("Epoch: {}, Val Loss: {}".format(epoch, val_loss))

        print("PREDICTOR TRAINED!")

        self.all_user_vecs = user_latent_reps

        return self
    
    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_vectors.shape[1]  ## Not being used, copied from tifuknn

        user_vectors = self._user_vectors[user_ids, :]
        # softmaxer = nn.LogSoftmax(dim=1)
        self.eval()
        self.predictor.eval()

        with torch.no_grad():
            batch = torch.tensor(user_vectors.todense(), dtype=torch.float)
            mu, log_var = self.encode(batch.to(self.device))
            user_latent_reps = self.reparameterize(mu, log_var).cpu()

            user_nn_indices = self._nbrs.kneighbors(user_latent_reps, return_distance=False)
            user_nn_vectors = []
            for nn_indices in user_nn_indices:
                nn_vector = self.all_user_vecs[nn_indices[1:], :].mean(dim=0)
                user_nn_vectors.append(nn_vector)
            user_nn_vectors = torch.stack(user_nn_vectors)

            user_combined_vec = self.alpha * user_latent_reps + (1 - self.alpha) * user_nn_vectors

            predict = self.predictor.forward(user_combined_vec.to(self.device))
            # predict = softmaxer(predict)
        
        return predict.cpu().numpy()


    def _calculate_basket_weight(self, df: pd.DataFrame):
            # Faster implementation using numpy instead of pandas
            df = df.sort_values(by="timestamp", ascending=False, ignore_index=True)

            group_size = np.ceil(len(df) / self.group_count)
            real_group_count = np.ceil(len(df) / group_size)

            group_num = np.arange(len(df)) // group_size
            basket_count = np.full(len(df), group_size)

            last_group_size = len(df) - (real_group_count - 1) * group_size
            basket_count[group_num == real_group_count - 1] = last_group_size

            basket_num = np.arange(len(df)) % group_size

            group_decay = self.group_decay_rate ** group_num / real_group_count
            within_decay = self.within_decay_rate ** basket_num / basket_count
            weight = group_decay * within_decay

            df["weight"] = weight

            return df
    
    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        num_nearest_neighbors = trial.suggest_categorical(
            "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
        )
        within_decay_rate = trial.suggest_categorical(
            "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        group_decay_rate = trial.suggest_categorical(
            "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        group_count = trial.suggest_int("group_count", 2, 23)
        return {
            "num_nearest_neighbors": num_nearest_neighbors,
            "within_decay_rate": within_decay_rate,
            "group_decay_rate": group_decay_rate,
            "alpha": alpha,
            "group_count": group_count,
        }