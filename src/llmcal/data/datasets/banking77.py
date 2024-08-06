from datasets import load_dataset, concatenate_datasets
import numpy as np


def load_banking():
    # data = load_dataset("PolyAI/banking77")
    data = load_dataset("banking77", data_dir="data")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["train"]), len(data["train"])+len(data["test"])))
    all_data = concatenate_datasets([data["train"], data["test"]])

    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(all_data))
    train_idx = idx[:len(data["train"])]
    test_idx = idx[len(data["train"]):]
    
    datadict = {
        "train": all_data.select(train_idx),
        "test": all_data.select(test_idx),
    }
    
    return datadict

