import pandas as pd
import numpy as np
from datasets import Dataset

def load_medical_abstracts(split):

    def load_split(split):
        df = pd.read_csv(
            f"data/raw_data/Medical-Abstracts-TC-Corpus/medical_tc_{split}.csv",
            sep=",",
            header=0,
        )
        df.columns = ["target", "abstract"]
        df["target"] = df["target"] - 1
        dataset = Dataset.from_pandas(df)
        dataset = dataset.add_column("idx", list(range(len(dataset))))
        dataset = dataset.map(lambda x: {"input": {"abstract": x["abstract"]}})
        dataset = dataset.remove_columns(["abstract"])
        return dataset

    if split == "test":
        dataset = load_split(split="test")
    elif split in ["train", "validation"]:
        dataset = load_split(split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:1000]
        elif split == "train":
            idx = idx[1000:]
        dataset = dataset.select(idx)
    return dataset

if __name__ == "__main__":
    ds = load_medical_abstracts("train").flatten()
    print(ds)
    ds = load_medical_abstracts("validation").flatten()
    print(ds)
    ds = load_medical_abstracts("test").flatten()
    print(ds)