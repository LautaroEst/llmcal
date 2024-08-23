import os
from datasets import load_dataset, load_from_disk
import numpy as np

def load_dbpedia():
    # data = load_dataset("fancyzhx/dbpedia_14")
    # data = load_dataset("dbpedia", data_dir="data")
    data = load_from_disk("data/dbpedia")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))

    rs = np.random.RandomState(27834)
    test_idx = rs.choice(len(data["test"]), 7000, replace=False)
    datadict = {
        "train": data["train"],
        "test": data["test"].select(test_idx),
    }
    return datadict