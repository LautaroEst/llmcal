import os
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
import pandas as pd

def load_medical_abstracts(data_dir):
    if not os.path.exists(data_dir):
        train_df = pd.read_csv("data/Medical-Abstracts-TC-Corpus/medical_tc_train.csv").rename(columns={"condition_label": "label", "medical_abstract": "abstract"})
        train_df["label"] = train_df["label"] - 1
        test_df = pd.read_csv("data/Medical-Abstracts-TC-Corpus/medical_tc_test.csv").rename(columns={"condition_label": "label", "medical_abstract": "abstract"})
        test_df["label"] = test_df["label"] - 1
        datadict = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df)
        })
        datadict["train"] = datadict["train"].add_column("idx", np.arange(len(datadict["train"])))
        datadict["test"] = datadict["test"].add_column("idx", np.arange(len(datadict["test"])))
        rs = np.random.RandomState(7348)
        idx = rs.permutation(len(datadict["train"]))
        train_idx = idx[:10000]
        val_idx = idx[10000:]
        datadict["validation"] = datadict["train"].select(val_idx)
        datadict["train"] = datadict["train"].select(train_idx)
        datadict.save_to_disk(data_dir)
    else:
        datadict = load_from_disk(data_dir)
    return datadict

if __name__ == "__main__":
    data_dir = "data/medical_abstracts"
    data = load_medical_abstracts(data_dir)
    print(data["train"])
    print(data["validation"])
    print(data["test"])
    print(data["validation"][:2])