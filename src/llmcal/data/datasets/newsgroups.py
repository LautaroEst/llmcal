import os
from datasets import load_dataset, load_from_disk, ClassLabel
import numpy as np

VAL_HELD_OUT_SAMPLES = 12 * 20

def load_newsgroups():
    data = load_dataset("SetFit/20_newsgroups")
    classes_names = data["train"].to_pandas().loc[:,["label","label_text"]].drop_duplicates().set_index("label").squeeze().sort_index().tolist()
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))
    for split in data:
        features = data[split].features
        features["label"] = ClassLabel(
            num_classes=20, 
            names=classes_names)
        data[split] = data[split].cast(features)
    
    datadict = {
        "train": data["train"],
        "test": data["test"]
    }

    return datadict