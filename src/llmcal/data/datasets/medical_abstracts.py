from datasets import Dataset, DatasetDict, ClassLabel
import numpy as np
import pandas as pd

VAL_HELD_OUT_SAMPLES = 12 * 5

def load_medical_abstracts():
    train_df = pd.read_csv("/home/lestienne/Documents/llmcal/data/Medical-Abstracts-TC-Corpus/medical_tc_train.csv").rename(columns={"condition_label": "label", "medical_abstract": "abstract"})
    train_df["label"] = train_df["label"] - 1
    test_df = pd.read_csv("/home/lestienne/Documents/llmcal/data/Medical-Abstracts-TC-Corpus/medical_tc_test.csv").rename(columns={"condition_label": "label", "medical_abstract": "abstract"})
    test_df["label"] = test_df["label"] - 1
    data = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
    for split in data:
        features = data[split].features
        features["label"] = ClassLabel(num_classes=5, names=["Neoplasms", "Digestive system diseases", "Nervous system diseases", "Cardiovascular diseases", "General pathological conditions"])
        data[split] = data[split].cast(features)
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))

    datadict = {
        "train": data["train"],
        "test": data["test"]
    }

    return datadict
