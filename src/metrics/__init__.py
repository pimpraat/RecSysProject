from .base import IMetric
from .recall import Recall
from .ndcg import NDCG
from .accuracy import Accuracy
from .precision import Precision
from .f1 import F1
from .mrr import MRR
from .phr import PHR


METRICS = {
    Recall.metric_name: Recall,
    NDCG.metric_name: NDCG,
    Accuracy.metric_name: Accuracy,
    Precision.metric_name: Precision,
    F1.metric_name: F1,
    MRR.metric_name: MRR,
    PHR.metric_name: PHR
}
