
import os
from typing import Optional, Dict
from datasets import load_from_disk, Dataset
import numpy as np
from .datasets import SUPPORTED_DATASETS

def load_dataset_from_hub(dataset_name: str) -> Dict[str,Dataset]:
    try:
        dataset_attrs = SUPPORTED_DATASETS[dataset_name]
    except KeyError:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are {list(SUPPORTED_DATASETS.keys())}")
    full_dataset = {}
    for split_name in ["train", "validation", "test"]:
        full_dataset[split_name] = dataset_attrs['loading_function'](split_name)
    return full_dataset

def load_dataset_from_disk(path: str) -> Dict[Dataset]:
    full_dataset = {}
    for split_name in ["train", "validation", "test"]:
        full_dataset[split_name] = load_from_disk(os.path.join(path,split_name))
    return full_dataset


def load_dataset(
    dataset: str, 
    load_from_hub: bool = False,
    random_state: int = 0,
    train_samples: Optional[int] = None,
    validation_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
)-> Dict[str,Dataset]:
    
    if load_from_hub:
        splits = load_dataset_from_hub(dataset)
    else:
        splits = load_dataset_from_disk(dataset)

    for n, split in zip([train_samples, validation_samples, test_samples],splits):
        data_split = splits[split]
        if n is None:
            n = len(data_split)
        rs = np.random.RandomState(random_state)
        idx = rs.choice(len(data_split), n, replace=False)
        data_split = data_split.select(idx)
        splits[split] = data_split

    return splits