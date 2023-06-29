import numpy as np

from .base import IMetric


def accuracy_at_k(true_basket: np.ndarray, model_scores: np.ndarray, topk: int):
    # print(f"topk={topk}")
    # print(f"true_basket: {true_basket}")
    # print(f"model_scores: {len(model_scores)}, {model_scores}")
    # print(f"model selected: ")
    scores = model_scores.copy()

    predicted_items = model_scores.argsort()[::-1][:topk]
    #TODO: Check this is a mess!
    # return np.count_nonzero(np.isin(predicted_items, true_basket))/topk

    return np.count_nonzero(np.isin(predicted_items, true_basket))/len(true_basket)

    # print(np.isin(predicted_items, true_basket))
    # print(f"predicted items for the basket: {predicted_items}")



    # scores[scores.argsort()[:-topk]] = 0
    # tp = np.count_nonzero(scores[true_basket])
    # correctly_selected = np.array(scores[true_basket] > 0, dtype=int)
    # print(f"scores[true_basket], which count_nonzero gives np: {scores[true_basket]}")
    



    # assert False

    # print(f"predicted items: {scores.argsort()[::-1]}")

    # predicted_items = model_scores.argsort()[::-1][:topk]
    # print(f"{np.isin(scores.argsort()[::-1], true_basket)}")

    # print(scores[true_basket])


    # assert False

    # print(f"Input for function {true_basket}, {predicted_items}")
    # assert len(true_basket) == len(predicted_items)

    print(true_basket,predicted_items)

    # return sklearn.metrics.top_k_accuracy_score(y_true=true_basket, y_score=predicted_items, k=topk)


    scores = model_scores.copy()
    scores[scores.argsort()[:-topk]] = 0
    tp = np.count_nonzero(scores[true_basket])

    recall_score = tp / min(topk, len(true_basket))
    assert 0 <= recall_score <= 1, recall_score
    return recall_score


class Accuracy(IMetric):

    metric_name: str = "accuracy"

    def __init__(self, topk=None):
        super().__init__(topk=topk)
        self.cumulative_value = 0.0
        self.n_users = 0

    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        if self.topk is None:
            self.topk = len(model_scores)
        self.cumulative_value += accuracy_at_k(true_basket, model_scores, self.topk)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_value / self.n_users

    def merge_with_other(self, other_metric_object):
        self.cumulative_value += other_metric_object.cumulative_value
        self.n_users += other_metric_object.n_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0
