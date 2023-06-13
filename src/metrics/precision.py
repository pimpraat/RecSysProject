import numpy as np

from .base import IMetric

# Note this function calculates the metric for just one basket
def precision_at_k(true_basket: np.ndarray, model_scores: np.ndarray, topk: int):
    #TODO: Verify
    scores = model_scores.copy()
    scores[scores.argsort()[:-topk]] = 0
    tp = np.count_nonzero(scores[true_basket])
    fp = np.count_nonzero(scores[true_basket][:len(true_basket)]==0)
    # print(f"tp: {tp}, fp:{fp}")
    
    precision = tp / (tp + fp)
    assert (tp + fp) == len(true_basket)
    return precision

class Precision(IMetric):

    metric_name: str = "precision"

    def __init__(self, topk=None):
        super().__init__(topk=topk)
        self.cumulative_value = 0.0
        self.n_users = 0

    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        if self.topk is None:
            self.topk = len(model_scores)
        self.cumulative_value += precision_at_k(true_basket, model_scores, self.topk)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_value / self.n_users

    def merge_with_other(self, other_metric_object):
        self.cumulative_value += other_metric_object.cumulative_value
        self.n_users += other_metric_object.n_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0
