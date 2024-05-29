from datasets import Dataset, DatasetDict, ClassLabel
import numpy as np
import pandas as pd

def load_medical_abstracts():
    train_df = pd.read_csv("/home/lestienne/Documents/llmcal/data/Medical-Abstracts-TC-Corpus/medical_tc_train.csv").rename(columns={"condition_label": "label", "medical_abstract": "abstract"})
    train_df["label"] = train_df["label"] - 1
    test_df = pd.read_csv("/home/lestienne/Documents/llmcal/data/Medical-Abstracts-TC-Corpus/medical_tc_test.csv").rename(columns={"condition_label": "label", "medical_abstract": "abstract"})
    test_df["label"] = test_df["label"] - 1
    datadict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
    for split in datadict:
        features = datadict[split].features
        features["label"] = ClassLabel(num_classes=5, names=["Neoplasms", "Digestive system diseases", "Nervous system diseases", "Cardiovascular diseases", "General pathological conditions"])
        datadict[split] = datadict[split].cast(features)

    datadict["train"] = datadict["train"].add_column("idx", np.arange(len(datadict["train"])))
    datadict["test"] = datadict["test"].add_column("idx", np.arange(len(datadict["test"])))
    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(datadict["train"]))
    train_idx = idx[:10000]
    val_idx = idx[10000:]
    datadict["validation"] = datadict["train"].select(val_idx)
    datadict["train"] = datadict["train"].select(train_idx)
    return datadict
