import argparse
import os
import json

from src.dataset import DATASETS
from src.models import MODELS
from src.settings import DATA_DIR, RESULTS_DIR
from src.evaluation import Evaluator
from src.hypertuning import run_model


def run_experiment(
    dataset: str,
    model: str,
    cutoff_list: list,
    batch_size: int,
    dataset_dir_name: str,
    verbose=True,
    num_nearest_neighbors=300,
    within_decay_rate=0.9,
    group_decay_rate=0.7,
    group_count=7,
    alpha=0.7,
):
    """
    Run a single experiment with the given parameters.

    Args:
        dataset (str): Name of the dataset.
        model (str): Name of the model.
        cutoff_list (list): List of cutoffs.
        batch_size (int): Batch size.
        dataset_dir_name (str): Name of the dataset directory.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        num_nearest_neighbors (int, optional): Number of nearest neighbors to use. Defaults to 300.
        within_decay_rate (float, optional): Within decay rate to use. Defaults to 0.9.
        group_decay_rate (float, optional): Group decay rate to use. Defaults to 0.7.
        group_count (int, optional): Group count to use. Defaults to 7.
        alpha (float, optional): Alpha to use. Defaults to 0.7.

    Returns:
        dict: Dictionary with the results of the experiment.
    """

    vparams = {
        "num_nearest_neighbors": num_nearest_neighbors,
        "within_decay_rate": within_decay_rate,
        "group_decay_rate": group_decay_rate,
        "group_count": group_count,
        "alpha": alpha,
    }

    if dataset in DATASETS.keys():
        dataset_cls = DATASETS[dataset]
    else:
        raise ValueError(f"Dataset {dataset} is unknown")

    if model in MODELS.keys():
        model_cls = MODELS[model]
    else:
        raise ValueError(f"Model {model} is unknown")

    if batch_size < 1:
        raise ValueError(f"Invalid batch_size: {batch_size}")

    if dataset_dir_name is None:
        dataset_dir_name = dataset
    else:
        dataset_full_path = os.path.join(DATA_DIR, dataset_dir_name)
        if not os.path.exists(dataset_full_path):
            raise ValueError(f"Dataset path doesn't exist: {dataset_full_path}")

    data = dataset_cls(dataset_dir_name, verbose=verbose)
    dataset_split_path = os.path.join(DATA_DIR, dataset_dir_name, "split")
    if not os.path.exists(dataset_split_path):
        data.make_leave_one_basket_split()
    data.load_split()

    evaluator_test = Evaluator(dataset_df=data.test_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)
    print("Running model...")
    results = run_model(
        dataset=data,
        model_cls=model_cls,
        evaluator=evaluator_test,
        vparams=vparams,
    )
    print("Done!")
    print(results)

    with open(os.path.join(RESULTS_DIR, f"{dataset}_{num_nearest_neighbors}_{within_decay_rate}_{group_decay_rate}_{group_count}_{alpha}.txt"), "w") as fp:
        json.dump(results, fp)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="instacart")
    parser.add_argument("--model", type=str, default="tifuknn")
    parser.add_argument("--cutoff_list", type=int, default=[10, 20])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dataset-dir-name", type=str, default=None)
    parser.add_argument("--num_nearest_neighbors", type=int, default=300)
    parser.add_argument("--within_decay_rate", type=float, default=0.9)
    parser.add_argument("--group_decay_rate", type=float, default=0.7)
    parser.add_argument("--group_count", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.7)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args, _ = parser.parse_known_args()
    run_experiment(
        dataset=args.dataset,
        model=args.model,
        cutoff_list=args.cutoff_list,
        batch_size=args.batch_size,
        dataset_dir_name=args.dataset_dir_name,
        num_nearest_neighbors=args.num_nearest_neighbors,
        within_decay_rate=args.within_decay_rate,
        group_decay_rate=args.group_decay_rate,
        group_count=args.group_count,
        alpha=args.alpha,
    )