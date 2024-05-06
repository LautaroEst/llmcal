import os
from datasets import load_dataset, load_from_disk
import numpy as np


def load_dbpedia(data_dir):
    if not os.path.exists(data_dir):
        datadict = load_dataset("fancyzhx/dbpedia_14")
        datadict["train"] = datadict["train"].add_column("idx", np.arange(len(datadict["train"])))
        datadict["test"] = datadict["test"].add_column("idx", np.arange(len(datadict["test"])))
        rs = np.random.RandomState(7348)
        idx = rs.permutation(len(datadict["train"]))
        train_idx = idx[:10000]
        val_idx = idx[10000:11000]
        datadict["validation"] = datadict["train"].select(val_idx)
        datadict["train"] = datadict["train"].select(train_idx)

        rs = np.random.RandomState(7348)
        idx = rs.permutation(len(datadict["test"]))
        test_idx = idx[:7000]
        datadict["test"] = datadict["test"].select(test_idx)
        datadict.save_to_disk(data_dir)
    else:
        datadict = load_from_disk(data_dir)
    return datadict

if __name__ == "__main__":
    data_dir = "data/newsgroup"
    data = load_dbpedia(data_dir)
    print(data["train"])
    print(data["validation"])
    print(data["test"])
    print(data["validation"][:2])