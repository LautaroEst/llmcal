
import numpy as np
from .sst2 import load_sst2
# from .banking77 import load_banking
from .dbpedia import load_dbpedia
# from .medical_abstracts import load_medical_abstracts
# from .newsgroup import load_newsgroup
from typing import Literal
from datasets import Dataset


SUPPORTED_DATASETS = Literal["sst2", "20newsgroup", "medical_abstracts", "dbpedia", "banking77"]
dataset2load_fn = {
    "sst2": load_sst2,
    # "20newsgroup": load_newsgroup,
    # "medical_abstracts": load_medical_abstracts,
    "dbpedia": load_dbpedia,
    # "banking77": load_banking
}

def sample_and_shuffle(dataset, num_samples, random_state):
    rs = np.random.RandomState(random_state)
    
    if isinstance(dataset, Dataset):
        if num_samples is None:
            num_samples = len(dataset)
        idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
        dataset = dataset.select(idx)
    elif isinstance(dataset, dict):
        if num_samples is None:
            num_samples = len(next(iter(dataset)))
        idx = rs.choice(next(iter(dataset.values())).shape[0], num_samples, replace=False).tolist()
        dataset = {k: v[idx] for k, v in dataset.items()}
    else:
        raise ValueError(f"Invalid type of dataset: {type(dataset)}")
    return dataset


def load_dataset(dataset_name, num_train_samples, num_val_samples, num_shots, random_state = 0):
    load_fn = dataset2load_fn[dataset_name]
    train_datadict = load_fn()
    train_datadict["train"] = sample_and_shuffle(train_datadict["train"], num_train_samples, random_state)
    train_datadict["validation"] = sample_and_shuffle(train_datadict["validation"], num_val_samples, random_state)
    if num_shots > 0:
        shots = sample_and_shuffle(train_datadict["train"], num_shots, random_state + 1)
    else:
        shots = []

    predict_datadict = load_fn() # original dataset
    return train_datadict, predict_datadict, shots