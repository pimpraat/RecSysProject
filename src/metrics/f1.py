import numpy as np

from .base import IMetric

from .precision import precision_at_k
from .recall import recall_at_k


def f1_at_k(true_basket: np.ndarray, model_scores: np.ndarray, topk: int):

    precision = precision_at_k(true_basket=true_basket, model_scores=model_scores, topk=topk)
    recall = recall_at_k(true_basket=true_basket, model_scores=model_scores, topk=topk)
    
    #Handle the 0 case for the f1 score:
    if (precision + recall) == 0: return 0
    
    return 2 * (precision * recall) / (precision + recall)

class F1(IMetric):

    metric_name: str = "f1"

    def __init__(self, topk=None):
        super().__init__(topk=topk)
        self.cumulative_value = 0.0
        self.n_users = 0

    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        if self.topk is None:
            self.topk = len(model_scores)
        self.cumulative_value += f1_at_k(true_basket, model_scores, self.topk)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_value / self.n_users

    def merge_with_other(self, other_metric_object):
        self.cumulative_value += other_metric_object.cumulative_value
        self.n_users += other_metric_object.n_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0
