import os
from datasets import load_dataset, load_from_disk, ClassLabel
import numpy as np


def load_newsgroups():
    datadict = load_dataset("SetFit/20_newsgroups")
    classes_names = datadict["train"].to_pandas().loc[:,["label","label_text"]].drop_duplicates().set_index("label").squeeze().sort_index().tolist()
    datadict["train"] = datadict["train"].add_column("idx", np.arange(len(datadict["train"])))
    datadict["test"] = datadict["test"].add_column("idx", np.arange(len(datadict["test"])))
    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(datadict["train"]))
    train_idx = idx[:10000]
    val_idx = idx[10000:]
    datadict["validation"] = datadict["train"].select(val_idx)
    datadict["train"] = datadict["train"].select(train_idx)
    for split in datadict:
        features = datadict[split].features
        features["label"] = ClassLabel(
            num_classes=20, 
            names=classes_names)
        datadict[split] = datadict[split].cast(features)
    return datadict