import os
from datasets import load_dataset
import numpy as np


def load_sst2():
    datadict = load_dataset("nyu-mll/glue", "sst2")
    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(datadict["train"]))[:10000]
    datadict["train"] = datadict["train"].select(idx)
    return datadict