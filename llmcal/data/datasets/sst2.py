import os
from datasets import load_dataset, load_from_disk
import numpy as np


def load_sst2(data_dir):
    if not os.path.exists(data_dir):
        datadict = load_dataset("nyu-mll/glue", "sst2")
        rs = np.random.RandomState(7348)
        idx = rs.permutation(len(datadict["train"]))[:10000]
        datadict["train"] = datadict["train"].select(idx)
        datadict.save_to_disk(data_dir)
    else:
        datadict = load_from_disk(data_dir)
    return datadict