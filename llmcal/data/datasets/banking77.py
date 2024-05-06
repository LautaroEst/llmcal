import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np


def load_banking(data_dir):
    if not os.path.exists(data_dir):
        datadict = load_dataset("PolyAI/banking77")
        datadict["train"] = datadict["train"].add_column("idx", np.arange(len(datadict["train"])))
        datadict["test"] = datadict["test"].add_column("idx", np.arange(len(datadict["train"])), len(datadict["train"])+len(datadict["test"]))
        all_data = concatenate_datasets([datadict["train"], datadict["test"]])
        rs = np.random.RandomState(7348)
        idx = rs.permutation(len(all_data))
        train_idx = idx[:10000]
        val_idx = idx[10000:11000]
        test_idx = idx[11000:]
        datadict["train"] = all_data.select(train_idx)
        datadict["validation"] = all_data.select(val_idx)
        datadict["test"] = all_data.select(test_idx)
        datadict.save_to_disk(data_dir)
    else:
        datadict = load_from_disk(data_dir)
    return datadict

if __name__ == "__main__":
    data_dir = "data/banking77"
    data = load_banking(data_dir)
    print(data["train"])
    print(data["validation"])
    print(data["test"])
    print(data["validation"].to_pandas()["label"].value_counts().sort_index().values)
    print(len(data["validation"].to_pandas()["label"].value_counts().sort_index()))
    # print(data["train"].to_pandas()["label"].value_counts().sort_index().values)