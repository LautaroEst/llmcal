from typing import Literal
from .sst2 import load_sst2

SUPPORTED_DATASETS = Literal["sst2", "20newsgroup", "medical_abstracts", "dbpedia", "banking77"]

