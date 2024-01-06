
import re
from .datasets import dataset2class

def load_dataset(dataset_name, split="train", subsample=None, random_state=None, sort_by_length=False):
    cls = dataset2class[dataset_name]
    return cls(split=split, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class Template:

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.features = self._get_features(prompt)
    
    def construct_prompt(self, **kwargs):
        prompt = self.prompt.format(**kwargs)
        return prompt
    
    def _get_features(self, prompt):
        features = re.findall(r"\{(\w+)\}", prompt)
        return features
