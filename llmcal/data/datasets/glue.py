
from torch.utils.data import Dataset
from datasets import load_dataset

class BaseGLUEDataset(Dataset):

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class GLUEcola(BaseGLUEDataset):
    idx2label = {0: "unacceptable", 1: "acceptable"}
    features = ["sentence"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "cola", split=split)

class GLUEmnli(BaseGLUEDataset):

    idx2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    features = ["premise", "hypothesis"]

    def __init__(self, split):
        if split == "validation":
            split = "validation_matched"
        elif split == "test":
            split = "test_matched"
        self.dataset = load_dataset("glue", "mnli", split=split)
    
class GLUEmrpc(BaseGLUEDataset):

    idx2label = {0: "not_equivalent", 1: "equivalent"}
    features = ["sentence1", "sentence2"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "mrpc", split=split)

class GLUEqnli(BaseGLUEDataset):

    idx2label = {0: "entailment", 1: "not_entailment"}
    features = ["question", "sentence"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "qnli", split=split)

class GLUEqqp(BaseGLUEDataset):
    
    idx2label = {0: "not_duplicate", 1: "duplicate"}
    features = ["question1", "question2"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "qqp", split=split)

class GLUErte(BaseGLUEDataset):

    idx2label = {0: "entailment", 1: "not_entailment"}
    features = ["sentence1", "sentence2"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "rte", split=split)

class GLUEsst2(BaseGLUEDataset):

    idx2label = {0: "negative", 1: "positive"}
    features = ["sentence"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "sst2", split=split)

class GLUEstsb(BaseGLUEDataset):

    idx2label = None
    features = ["sentence1", "sentence2"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "stsb", split=split)

class GLUEwnli(BaseGLUEDataset):

    idx2label = {0: "not_entailment", 1: "entailment"}
    features = ["sentence1", "sentence2"]

    def __init__(self, split):
        self.dataset = load_dataset("glue", "wnli", split=split)
