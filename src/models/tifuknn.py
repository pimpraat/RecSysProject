import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.neighbors import NearestNeighbors

from src.models.core import IRecommender
from src.dataset import NBRDatasetBase


class TIFUKNNRecommender(IRecommender):
    def __init__(
        self,
        num_nearest_neighbors: int = 300,
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

    def fit(self, dataset: NBRDatasetBase):
        user_basket_df = dataset.train_df.groupby("user_id", as_index=False).apply(self._calculate_basket_weight)
        user_basket_df.reset_index(drop=True, inplace=True)

        df = user_basket_df.explode("basket", ignore_index=True).rename(columns={"basket": "item_id"})
        df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("weight", "sum"))
        self._user_vectors = sps.csr_matrix(
            (df.value, (df.user_id, df.item_id)),
            shape=(dataset.num_users, dataset.num_items),
        )

        self._nbrs = NearestNeighbors(
            n_neighbors=self.num_nearest_neighbors + 1, # TODO: Why +1?
            algorithm="brute",
            metric=self.similarity_measure
        ).fit(self._user_vectors)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_vectors.shape[1]

        user_vectors = self._user_vectors[user_ids, :]

        user_nn_indices = self._nbrs.kneighbors(user_vectors, return_distance=False)

        user_nn_vectors = []
        for nn_indices in user_nn_indices:
            nn_vectors = self._user_vectors[nn_indices[1:], :].mean(axis=0)
            user_nn_vectors.append(sps.csr_matrix(nn_vectors))
        user_nn_vectors = sps.vstack(user_nn_vectors)

        pred_matrix = self.alpha * user_vectors + (1 - self.alpha) * user_nn_vectors
        return pred_matrix

    # def _calculate_basket_weight(self, df: pd.DataFrame):
    #     df = df.sort_values(by="timestamp", ascending=False, ignore_index=True)
    #
    #     group_size = math.ceil(len(df) / self.group_count)
    #     df["group_num"] = df.index // group_size
    #     real_group_count = df["group_num"].max() + 1
    #
    #     df["basket_count"] = group_size
    #     df.loc[df["group_num"] == len(df) // group_size, "basket_count"] = len(df) % group_size
    #     df["basket_num"] = df.groupby("group_num").cumcount()
    #
    #     df["weight"] = (self.group_decay_rate ** df["group_num"] / real_group_count) * (
    #         self.within_decay_rate ** df["basket_num"] / df["basket_count"]
    #     )
    #
    #     df.drop(columns=["group_num", "basket_count", "basket_num"], inplace=True)
    #     return df

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

    # @classmethod
    # def sample_params(cls, trial: optuna.Trial) -> dict:
    #     num_nearest_neighbors = trial.suggest_categorical(
    #         "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
    #     )
    #     within_decay_rate = trial.suggest_categorical(
    #         "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #     )
    #     group_decay_rate = trial.suggest_categorical(
    #         "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #     )
    #     alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #     group_count = trial.suggest_int("group_count", 2, 23)
    #     return {
    #         "num_nearest_neighbors": num_nearest_neighbors,
    #         "within_decay_rate": within_decay_rate,
    #         "group_decay_rate": group_decay_rate,
    #         "alpha": alpha,
    #         "group_count": group_count,
    #     }
