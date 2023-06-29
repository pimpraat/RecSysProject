# This file is used to generate all the final results as reported in our paper. We take care of hyperparameter setting etc to avoid copying/setting mistakes.
# For hyperparameter finding, saving user metrics, and running with custom hyperparameters we refer tot the run_experiment.py file.

from src.dataset import DATASETS
from src.models import MODELS
from src.settings import DATA_DIR
from src.evaluation import Evaluator
from src.run import run_model
import json
import argparse
import os


HPARAMS = {
    'instacart': {
        'num_nearest_neighbors': 900, 
        'within_decay_rate': 0.9, 
        'group_decay_rate': 0.7, 
        'alpha': 0.9, 
        'group_count': 3
        },
    'tafeng': {
        'num_nearest_neighbors': 300, 
        'within_decay_rate': 0.9, 
        'group_decay_rate': 0.7, 
        'alpha': 0.7, 
        'group_count': 7
        },
    'dunnhumby': {
        'num_nearest_neighbors': 900, 
        'within_decay_rate': 0.9, 
        'group_decay_rate': 0.6, 
        'alpha': 0.2, 
        'group_count': 3
        },
    'valuedshopper': {
        'num_nearest_neighbors': 300, 
        'within_decay_rate': 1, 
        'group_decay_rate': 0.6, 
        'alpha': 0.7, 
        'group_count': 7
        },
    'tmall': {
        'num_nearest_neighbors': 100, 
        'within_decay_rate': 0.6, 
        'group_decay_rate': 0.8, 
        'alpha': 0.7, 
        'group_count': 18
        },
    'taobao': {
        'num_nearest_neighbors': 300, 
        'within_decay_rate': 0.6, 
        'group_decay_rate': 0.8, 
        'alpha': 0.1, 
        'group_count': 10
        }
}

BETA_VAE_K_PARAM = {
    'instacart': 50,
    'tafeng': 10,
    'dunnhumby': 20,
    'valuedshopper': 10,
    'tmall': 250,
    'taobao': 150
}


def run_eval_model(dataset:str,model:str, batch_size:int, cutoff_list:list, verbose=True):
    """
    Run a single experiment with the given parameters.

    Args:
        dataset (str): Name of the dataset.
        model (str): Name of the model.
        cutoff_list (list): List of cutoffs.
        batch_size (int): Batch size.
    """

    # Load the selected model, dataset and hyperparameters
    dataset_cls, model_cls = DATASETS[dataset], MODELS[model]
    selected_params = HPARAMS[dataset]
    data = dataset_cls(dataset, verbose=verbose)
    
    # # Min_freq is in the Naumov paper set between 1 and 20, and the preprocssesing to None, 'binary', or 'log'
    if model == 'top_personal':
        selected_params = {
            'min_freq': 1,
            'preprocessing_popular': None,
            'preprocessing_personal': None
        }
    elif model == 'tifuknn':
        selected_params = HPARAMS[dataset]
    elif model == 'betavae':
        beta_vae_nn = BETA_VAE_K_PARAM[dataset]
        selected_params['num_nearest_neighbors'] = beta_vae_nn

    # Create the split if not already present in the data
    dataset_split_path = os.path.join(DATA_DIR, dataset, "split")
    if not os.path.exists(os.path.join(dataset_split_path, "train.csv")):
        data.make_leave_one_basket_split()
    data.load_split()
    

    data_object = None
    with open('results/data.json', 'r') as openfile: data_object = json.load(openfile)

    evaluator_test = Evaluator(dataset_df=data.test_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)

    performance_dct = run_model(
            dataset=data,
            model_cls=model_cls,
            evaluator=evaluator_test,
            vparams=selected_params,
        )
    # best_vmodel = model_cls(**selected_params)
    # best_vmodel.fit(dataset=data)
    # performance_dct = evaluator_test.evaluate_recommender(best_vmodel)

    s_metric = 'euclidean'
    data_object[str([model, dataset, s_metric])]= performance_dct
    for k, v in performance_dct.items(): print(f"{k}: {v}")

    with open("results/data.json", "w+") as fp: json.dump(data_object, fp, indent=4)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tafeng")
    parser.add_argument("--model", type=str, default="tifuknn")
    parser.add_argument("--cutoff_list", type=int, default=[10, 20])
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args, _ = parser.parse_known_args()
    run_eval_model(dataset=args.dataset, model=args.model, batch_size=args.batch_size, cutoff_list=args.cutoff_list)