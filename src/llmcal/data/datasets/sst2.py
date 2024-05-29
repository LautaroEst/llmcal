import os
from datasets import load_dataset
import numpy as np


def load_sst2():
    datadict = load_dataset("nyu-mll/glue", "sst2")
    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(datadict["train"]))
    train_idx = idx[:10000]
    val_idx = idx[10000:11000]
    datadict["test"] = datadict["validation"]
    datadict["validation"] = datadict["train"].select(val_idx)
    datadict["train"] = datadict["train"].select(train_idx)
    return datadict