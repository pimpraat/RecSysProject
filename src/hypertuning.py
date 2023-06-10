from typing import Dict, Type

from src.evaluation import Evaluator
from src.utils import set_global_seed
from src.models import IRecommender
from src.dataset import NBRDatasetBase


def run_model(
    dataset: NBRDatasetBase,
    model_cls: Type[IRecommender],
    evaluator: Evaluator,
    vparams: Dict,
):
    set_global_seed(42)
    model = model_cls(**vparams)

    set_global_seed(42)
    model.fit(dataset=dataset)

    set_global_seed(42)
    performance_dict = evaluator.evaluate_recommender(model)

    return performance_dict
