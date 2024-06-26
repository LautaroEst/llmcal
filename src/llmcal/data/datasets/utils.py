
import os
import numpy as np
from .sst2 import load_sst2
from .banking77 import load_banking
from .dbpedia import load_dbpedia
from .medical_abstracts import load_medical_abstracts
from .newsgroups import load_newsgroups
from .agnews import load_agnews
from typing import Literal
from datasets import Dataset, load_from_disk
from sklearn.model_selection import KFold


SUPPORTED_DATASETS = Literal["sst2", "dbpedia", "agnews", "20newsgroups", "medical-abstracts", "banking77"]
dataset2load_fn = {
    "sst2": load_sst2,
    "dbpedia": load_dbpedia,
    "agnews": load_agnews,
    "20newsgroups": load_newsgroups,
    "medical-abstracts": load_medical_abstracts,
    "banking77": load_banking
}

# def sample_and_shuffle(dataset, num_samples, random_state):
#     rs = np.random.RandomState(random_state)
    
#     if isinstance(dataset, Dataset):
#         if num_samples is None:
#             num_samples = len(dataset)
#         idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
#         dataset = dataset.select(idx)
#     elif isinstance(dataset, dict):
#         if num_samples is None:
#             num_samples = len(next(iter(dataset)))
#         idx = rs.choice(next(iter(dataset.values())).shape[0], num_samples, replace=False).tolist()
#         dataset = {k: v[idx] for k, v in dataset.items()}
#     else:
#         raise ValueError(f"Invalid type of dataset: {type(dataset)}")
#     return dataset


# def load_dataset(dataset_name, num_train_samples, num_val_samples, num_shots, random_state = 0):
#     if dataset_name in dataset2load_fn:
#         load_fn = dataset2load_fn[dataset_name]
#     else:
#         load_fn = lambda: {split: load_from_disk(os.path.join(dataset_name, split)) for split in ["train", "validation", "test"]}
#     train_datadict = load_fn()
#     train_datadict["train"] = sample_and_shuffle(train_datadict["train"], num_train_samples, random_state)
#     train_datadict["validation"] = sample_and_shuffle(train_datadict["validation"], num_val_samples, random_state)
#     if num_shots > 0:
#         shots = sample_and_shuffle(train_datadict["train"], num_shots, random_state + 1)
#         all_ids = np.asarray(train_datadict["train"]["idx"])
#         shot_ids = np.asarray(shots["idx"])
#         new_train_ids = np.setdiff1d(all_ids, shot_ids)
#         new_train_ids = [i for i in range(len(all_ids)) if train_datadict["train"][i]["idx"] in new_train_ids]
#         train_datadict["train"] = train_datadict["train"].select(new_train_ids)
#     else:
#         shots = []

#     predict_datadict = load_fn() # original dataset
#     return train_datadict, predict_datadict, shots


def load_dataset(dataset_name: SUPPORTED_DATASETS, total_train_samples: int, val_prop: float, num_shots: int, random_state: int = 0, timing: bool = False):
    if dataset_name in dataset2load_fn:
        datadict = dataset2load_fn[dataset_name]()

        train_samples = int(total_train_samples * (1 - val_prop))
        val_samples = total_train_samples - train_samples
        rs = np.random.RandomState(random_state)
        idx = rs.permutation(len(datadict["train"]))
        train_idx = idx[:train_samples]
        val_idx = idx[train_samples:train_samples + val_samples]
        datadict["validation"] = datadict["train"].select(val_idx)

        if num_shots > 0:
            shots = datadict["train"].select(train_idx[:num_shots])
            train_idx = train_idx[num_shots:]
        else:
            shots = []
        datadict["train"] = datadict["train"].select(train_idx)
    
    else:
        if not timing:
            datadict = {split: load_from_disk(os.path.join(dataset_name, split)) for split in ["train", "validation", "test"]}
        else:
            datadict = {split: load_from_disk(os.path.join(dataset_name, split)) for split in ["train", "validation"]}
        shots = []

    train_datadict = {split: datadict[split] for split in ["train", "validation"]}
    if not timing:
        predict_datadict = {split: datadict[split] for split in ["train", "validation", "test"]}
    else:
        predict_datadict = {split: datadict[split] for split in ["train", "validation"]}

    return train_datadict, predict_datadict, shots
        
def load_dataset_for_xval(dataset_name: SUPPORTED_DATASETS, total_train_samples: int, nfolds: int, random_state: int = 0):
    if dataset_name in dataset2load_fn:
        datadict = dataset2load_fn[dataset_name]()

        rs = np.random.RandomState(random_state)
        idx = rs.permutation(len(datadict["train"]))
        all_data = datadict["train"].select(idx[:total_train_samples])
        
        folds = []
        skf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
        all_idx = np.arange(total_train_samples)
        for train_idx, val_idx in skf.split(all_idx):
            train_data = all_data.select(train_idx)
            val_data = all_data.select(val_idx)
            folds.append({"train": train_data, "validation": val_data})

    return folds
        
