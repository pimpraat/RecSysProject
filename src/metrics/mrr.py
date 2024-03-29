import numpy as np

from .base import IMetric


def mrr_at_k(true_basket: np.ndarray, model_scores: np.ndarray, topk: int):
   
    scores = model_scores.copy()

    predicted_items = model_scores.argsort()[::-1]
    predicted_items = predicted_items#[0:topk]

    # TODO: Check/verify!
    for idx, real_item in enumerate(predicted_items):
       if real_item in true_basket:
           return 1/(idx+1)

class MRR(IMetric):

    metric_name: str = "mrr"

    def __init__(self, topk=None):
        super().__init__(topk=topk)
        self.cumulative_value = 0.0
        self.n_users = 0

    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        if self.topk is None:
            self.topk = len(model_scores)
        self.cumulative_value += mrr_at_k(true_basket, model_scores, self.topk)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_value / self.n_users

    def merge_with_other(self, other_metric_object):
        self.cumulative_value += other_metric_object.cumulative_value
        self.n_users += other_metric_object.n_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0
