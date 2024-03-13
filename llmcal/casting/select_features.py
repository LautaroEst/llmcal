
from typing import Union, Dict, List
from datasets import Dataset

class FeaturesSelector:

    def __init__(self, features: Union[Dict[str,str],List[str],str]):
        if isinstance(features,dict):
            self.features = features
            self._transform = self._transform_dict_of_features
        elif isinstance(features, list):
            self.features = {f: f for f in features}
            self._transform = self._transform_dict_of_features
        elif isinstance(features, str):
            self.feature = features
            self._transform = self._transform_single_feature
        else:
            raise ValueError("features should be a list, a dict or a string")

    def fit(self, data):
        pass

    def _transform_single_feature(self, sample):
        sample["input"] = sample[self.feature]
        return sample
    
    def _transform_dict_of_features(self, sample):
        sample["input"] = {new_feature: sample[old_feature] for old_feature, new_feature in self.features.items()}
        return sample

    def transform(self, data: Dataset) -> Dataset:
        data = data.flatten().map(self._transform, batched=False)
        return data
