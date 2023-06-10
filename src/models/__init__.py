from .tifuknn import (
    TIFUKNNRecommender,
)
from .core import IRecommender, IRecommenderNextTs


MODELS = {
    "tifuknn": TIFUKNNRecommender
}
