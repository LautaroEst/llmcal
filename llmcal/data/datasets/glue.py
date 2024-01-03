
from .base import BaseDataset
from datasets import load_dataset


class GLUEcola(BaseDataset):

    idx2label = {0: "unacceptable", 1: "acceptable"}
    features = ["sentence"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "cola", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUEmnli(BaseDataset):

    idx2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    features = ["premise", "hypothesis"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        if split == "validation":
            split = "validation_matched"
        elif split == "test":
            split = "test_matched"
        dataset = load_dataset("glue", "mnli", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)
    

class GLUEmrpc(BaseDataset):

    idx2label = {0: "not_equivalent", 1: "equivalent"}
    features = ["sentence1", "sentence2"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "mrpc", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUEqnli(BaseDataset):

    idx2label = {0: "entailment", 1: "not_entailment"}
    features = ["question", "sentence"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "qnli", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUEqqp(BaseDataset):
    
    idx2label = {0: "not_duplicate", 1: "duplicate"}
    features = ["question1", "question2"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "qqp", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUErte(BaseDataset):

    idx2label = {0: "entailment", 1: "not_entailment"}
    features = ["sentence1", "sentence2"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "rte", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUEsst2(BaseDataset):

    idx2label = {0: "negative", 1: "positive"}
    features = ["sentence"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "sst2", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUEstsb(BaseDataset):

    idx2label = None
    features = ["sentence1", "sentence2"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "stsb", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class GLUEwnli(BaseDataset):

    idx2label = {0: "not_entailment", 1: "entailment"}
    features = ["sentence1", "sentence2"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        dataset = load_dataset("glue", "wnli", split=split)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


if __name__ == "__main__":
    dataset = GLUEsst2("train", subsample=100)
    print(dataset[0])