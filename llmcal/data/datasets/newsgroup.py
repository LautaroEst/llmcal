
from datasets import load_dataset
import numpy as np

def load_newsgroup(split):
    if split == "test":
        dataset = load_dataset("SetFit/20_newsgroups", split="test")
        dataset = dataset.add_column("idx", list(range(len(dataset))))
    elif split in ["train", "validation"]:
        dataset = load_dataset("SetFit/20_newsgroups", split="train")
        dataset = dataset.add_column("idx", list(range(len(dataset))))
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:1000]
        elif split == "train":
            idx = idx[1000:11000]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    dataset = dataset.map(lambda x: {"input": {"text": x["text"]}})
    dataset = dataset.remove_columns(["text","label_text"])
    return dataset
