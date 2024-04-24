import os
import numpy as np
from datasets import load_dataset, load_from_disk


def _sample_and_shuffle(dataset, num_samples, random_state):
    rs = np.random.RandomState(random_state)
    if num_samples is None:
        num_samples = len(dataset)
    idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
    dataset = dataset.select(idx)
    return dataset


def load_sst2(data_dir, num_train_samples, num_val_samples, num_shots, random_state = 0):
    if not os.path.exists(data_dir):
        datadict = load_dataset("nyu-mll/glue", "sst2")
        datadict.save_to_disk(data_dir)
    else:
        datadict = load_from_disk(data_dir)
    datadict["train"] = _sample_and_shuffle(datadict["train"], num_train_samples, random_state)
    datadict["validation"] = _sample_and_shuffle(datadict["validation"], num_val_samples, random_state)
    if num_shots > 0:
        shots = _sample_and_shuffle(datadict["train"], num_shots, random_state + 1)
    else:
        shots = []
    return datadict, shots
