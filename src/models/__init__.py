from .tifuknn import TIFUKNNRecommender
from .betavae import BetaVAERecommender
from .statistical import TopPersonalRecommender

from .core import IRecommender, IRecommenderNextTs


MODELS = {
    "tifuknn": TIFUKNNRecommender,
    "betavae": BetaVAERecommender,
    "top_personal": TopPersonalRecommender
}
