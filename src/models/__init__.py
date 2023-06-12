from .tifuknn import (
    TIFUKNNRecommender,
)
from .betavae import (
    BetaVAERecommender,
)
from .core import IRecommender, IRecommenderNextTs


MODELS = {
    "tifuknn": TIFUKNNRecommender,
    "betavae": BetaVAERecommender
}
