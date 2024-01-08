
from .datasets import dataset2class

def load_dataset(dataset_name, split="train", subsample=None, random_state=None, sort_by_length=False):
    cls = dataset2class[dataset_name]
    return cls(split=split, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class Template:

    def __init__(self, prompt: str, prompt_label_separator: str = " ", features: list = None, labels: dict = None):
        
        self.prompt = prompt
        self.prompt_label_separator = prompt_label_separator
        self.features = features
        self.labels = {int(i): l for i, l in labels.items()}
    
    def construct_prompt(self, **kwargs):
        return self.prompt.format(**kwargs)

    def construct_label(self, label: str):
        return f"{self.prompt_label_separator}{label}"