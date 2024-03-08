
from collections import OrderedDict
from .glue import GLUE_DATASETS
from .tony_zhao import TONYZHAO_DATASETS

SUPPORTED_DATASETS = OrderedDict([
    *GLUE_DATASETS,
    *TONYZHAO_DATASETS,
])