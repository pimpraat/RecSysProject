from .base import NBRDatasetBase
from .dunnhumby import DunnhumbyDataset
from .instacart import InstacartDataset
from .tafeng import TafengDataset
from .taobao import TaobaoDataset
from .tmall import TmallDataset


DATASETS = {
    "dunnhumby": DunnhumbyDataset,
    "instacart": InstacartDataset,
    "tafeng": TafengDataset,
    "taobao": TaobaoDataset,
    "tmall": TmallDataset,
}
