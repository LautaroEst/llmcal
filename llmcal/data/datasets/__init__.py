
from collections import OrderedDict
from .glue import GLUE_DATASETS
from .tony_zhao import TONYZHAO_DATASETS
from .refind import load_refind
from .newsgroup import load_newsgroup
from .banking77 import load_banking77
from .medical_abstracts import load_medical_abstracts

SUPPORTED_DATASETS = OrderedDict([
    ("refind", load_refind),
    *GLUE_DATASETS,
    *TONYZHAO_DATASETS,
    ("20newsgroup", load_newsgroup),
    ("banking77", load_banking77),
    ("medical_abstracts", load_medical_abstracts),
])