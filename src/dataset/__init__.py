from .base import NBRDatasetBase
from .dunnhumby import DunnhumbyDataset
from .instacart import InstacartDataset
from .tafeng import TafengDataset
from .taobao import TaobaoDataset
from .tmall import TmallDataset
from .valuedshopper import VSDataset
from .tianchi import TianchiDataset


DATASETS = {
    "dunnhumby": DunnhumbyDataset,
    "instacart": InstacartDataset,
    "valuedshopper": VSDataset,
    "tafeng": TafengDataset,
    "taobao": TaobaoDataset,
    "tmall": TmallDataset,
    "tianchi": TianchiDataset,
}
