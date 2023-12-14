
from .datasets import dataset2class

def load_dataset(dataset_name, split):
    cls = dataset2class[dataset_name]
    return cls(split=split)