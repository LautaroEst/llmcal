import os
from datasets import load_dataset, load_from_disk
from .utils import sample_and_shuffle


def load_sst2(data_dir, num_train_samples, num_val_samples, num_shots, random_state = 0):
    if not os.path.exists(data_dir):
        datadict = load_dataset("nyu-mll/glue", "sst2")
        datadict.save_to_disk(data_dir)
    else:
        datadict = load_from_disk(data_dir)
    datadict["train"] = sample_and_shuffle(datadict["train"], num_train_samples, random_state)
    datadict["validation"] = sample_and_shuffle(datadict["validation"], num_val_samples, random_state)
    if num_shots > 0:
        shots = sample_and_shuffle(datadict["train"], num_shots, random_state + 1)
    else:
        shots = []
    return datadict, shots
