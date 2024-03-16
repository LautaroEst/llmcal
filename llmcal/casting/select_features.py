
from typing import Union, Dict, List
from datasets import Dataset

class FeaturesSelector:

    def __init__(self, features: Dict[str,str]):
        self.features = features

    def fit(self, data):
        pass

    def _transform(self, sample):
        sample = {new_feature: sample[old_feature] for old_feature, new_feature in self.features.items()}
        kv = list(sample.items())
        for key, value in kv:
            if "." in key:
                keys = key.split(".")
                if keys[0] not in sample:
                    sample[keys[0]] = {keys[1]: value}
                else:
                    sample[keys[0]][keys[1]] = value
                del sample[key]
        return sample

    def transform(self, data: Dataset) -> Dataset:
        data = data.flatten().map(self._transform, batched=False)
        return data
