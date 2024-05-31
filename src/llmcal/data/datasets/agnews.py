from datasets import load_dataset
import numpy as np

def load_agnews():
    data = load_dataset("ag_news")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))
    datadict = {
        "train": data["train"],
        "test": data["test"]
    }
    return datadict