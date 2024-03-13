
from datasets import load_dataset
import numpy as np

def load_cola(split):
    if split == "test":
        dataset = load_dataset("glue", "cola", split="validation")
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "cola", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:400]
        elif split == "train":
            idx = idx[400:10400]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset

def load_mnli(split):
    if split == "test":
        dataset = load_dataset("glue", "mnli", split="validation_matched")
        rs = np.random.RandomState(0)
        idx = rs.permutation(len(dataset))[:2000]
        dataset = dataset.select(idx)
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "mnli", split="train_matched")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:300]
        elif split == "train":
            idx = idx[300:10300]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset

def load_mrpc(split):
    if split == "test":
        dataset = load_dataset("glue", "mrpc", split="validation")
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "mrpc", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:400]
        elif split == "train":
            idx = idx[400:]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset


def load_qnli(split):
    if split == "test":
        dataset = load_dataset("glue", "qnli", split="validation")
        rs = np.random.RandomState(0)
        idx = rs.permutation(len(dataset))[:2000]
        dataset = dataset.select(idx)
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "qnli", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:400]
        elif split == "train":
            idx = idx[400:10400]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset


def load_qqp(split):
    if split == "test":
        dataset = load_dataset("glue", "qqp", split="validation")
        rs = np.random.RandomState(0)
        idx = rs.permutation(len(dataset))[:2000]
        dataset = dataset.select(idx)
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "qqp", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:400]
        elif split == "train":
            idx = idx[400:10400]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset

def load_rte(split):
    if split == "test":
        dataset = load_dataset("glue", "rte", split="validation")
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "rte", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:400]
        elif split == "train":
            idx = idx[400:]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset


def load_sst2(split):
    if split == "test":
        dataset = load_dataset("glue", "sst2", split="validation")
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "sst2", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:400]
        elif split == "train":
            idx = idx[400:10400]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset


def load_wnli(split):
    if split == "test":
        dataset = load_dataset("glue", "wnli", split="validation")
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "wnli", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:100]
        elif split == "train":
            idx = idx[100:]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    return dataset



GLUE_DATASETS = [
    ("glue--cola", {
        "loading_function": load_cola,
        "features": ["sentence"],
    }),
    ("glue--mnli", {
        "loading_function": load_mnli,
        "features": ["premise", "hypothesis"],
    }),
    ("glue--mrpc", {
        "loading_function": load_mrpc,
        "features": ["sentence1", "sentence2"],
    }),
    ("glue--qnli", {
        "loading_function": load_qnli,
        "features": ["question", "sentence"],
    }),
    ("glue--qqp", {
        "loading_function": load_qqp,
        "features": ["question1", "question2"],
    }),
    ("glue--rte", {
        "loading_function": load_rte,
        "features": ["sentence1", "sentence2"],
    }),
    ("glue--sst2", {
        "loading_function": load_sst2,
        "features": ["sentence"],
    }),
    ("glue--wnli", {
        "loading_function": load_wnli,
        "features": ["sentence1", "sentence2"],
    }),
]