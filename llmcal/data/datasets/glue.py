
from datasets import load_dataset
import numpy as np

def load_cola(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "cola", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"sentence": x["sentence"]}})
    dataset = dataset.remove_columns(["sentence"])
    return dataset


def load_mnli(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
        rs = np.random.RandomState(0)
        idx = rs.permutation(len(dataset))[:2000]
        dataset = dataset.select(idx)
    elif split in ["train", "validation"]:
        dataset = load_dataset("glue", "mnli", split="train")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(dataset))
        if split == "validation":
            idx = idx[:300]
        elif split == "train":
            idx = idx[300:10300]
        dataset = dataset.select(idx)
    dataset = dataset.rename_column("label","target")
    dataset = dataset.map(lambda x: {"input": {"premise": x["premise"], "hypothesis": x["hypothesis"]}})
    dataset = dataset.remove_columns(["premise", "hypothesis"])
    return dataset


def load_mrpc(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "mrpc", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"sentence1": x["sentence1"], "sentence2": x["sentence2"]}})
    dataset = dataset.remove_columns(["sentence1", "sentence2"])
    return dataset


def load_qnli(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "qnli", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"question": x["question"], "sentence": x["sentence"]}})
    dataset = dataset.remove_columns(["question", "sentence"])
    return dataset


def load_qqp(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "qqp", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"question1": x["question1"], "question2": x["question2"]}})
    dataset = dataset.remove_columns(["question1", "question2"])
    return dataset


def load_rte(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "rte", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"sentence1": x["sentence1"], "sentence2": x["sentence2"]}})
    dataset = dataset.remove_columns(["sentence1", "sentence2"])
    return dataset


def load_sst2(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"sentence": x["sentence"]}})
    dataset = dataset.remove_columns(["sentence"])
    return dataset


def load_wnli(split):
    if split == "test":
        dataset = load_dataset("nyu-mll/glue", "wnli", split="validation")
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
    dataset = dataset.map(lambda x: {"input": {"sentence1": x["sentence1"], "sentence2": x["sentence2"]}})
    dataset = dataset.remove_columns(["sentence1", "sentence2"])
    return dataset



GLUE_DATASETS = [
    ("glue_cola", load_cola),
    ("glue_mnli", load_mnli),
    ("glue_mrpc", load_mrpc),
    ("glue_qnli", load_qnli),
    ("glue_qqp", load_qqp),
    ("glue_rte", load_rte),
    ("glue_sst2", load_sst2),
    ("glue_wnli", load_wnli),
]

if __name__ == "__main__":
    dataset = load_mnli("train")
    print(dataset)