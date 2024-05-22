from datasets import load_dataset
import numpy as np


def load_agnews():
    datadict = load_dataset("ag_news")
    datadict["train"] = datadict["train"].add_column("idx", np.arange(len(datadict["train"])))
    datadict["test"] = datadict["test"].add_column("idx", np.arange(len(datadict["test"])))
    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(datadict["train"]))
    train_idx = idx[:10000]
    val_idx = idx[10000:11000]
    datadict["validation"] = datadict["train"].select(val_idx)
    datadict["train"] = datadict["train"].select(train_idx)
    return datadict