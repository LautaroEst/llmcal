
from collections import OrderedDict
from .glue import GLUE_DATASETS
from .tony_zhao import TONYZHAO_DATASETS
from .refind import load_refind

SUPPORTED_DATASETS = OrderedDict([
    ("refind", load_refind),
    *GLUE_DATASETS,
    *TONYZHAO_DATASETS,
])