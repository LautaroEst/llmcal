
import re
from typing import Any, Dict, List
from datasets import Dataset
import numpy as np


class FewShotClassificationPrompt:

    def __init__(self, preface, shots_template, question, label2answer, n_shots=2, random_state=0):
        self.preface = preface
        self.shots_template = shots_template
        self.question = question
        self.label2answer = {i: label for i, label in enumerate(label2answer)}
        self.n_shots = n_shots
        self.random_state = random_state

    def fit(self, shots: Dataset):

        shots = shots.flatten()
        self.question_features = re.findall(r'\{([\w\.]+)\}', self.question)

        if self.n_shots > 0:
            shots_template_features = re.findall(r'\{([\w\.]+)\}', self.shots_template)
            if not set(shots_template_features).issubset(set(shots.features.keys())):
                raise ValueError(f"Expected features {shots_template_features} in shots, got {list(shots.features.keys())}")
            index = np.random.RandomState(self.random_state).choice(len(shots), self.n_shots, replace=False).tolist()
        else:
            index = []
        
        shots_str = ""
        for idx in index:
            features = {}
            for feature in shots_template_features:
                if feature == "target":
                    features["target"] = self.label2answer[shots[idx]["target"]]
                else:
                    features[feature] = shots[idx][feature]
            shots_str += self.shots_template.format(**features)
        self.prompt_template = f"{self.preface}{shots_str}{self.question}"
        return self
    
    def _transform(self, samples: Dict[str,List[Any]]):
        batch_size = len(samples["idx"])
        inputs = []
        for i in range(batch_size):
            features = {}
            prompt_template = self.prompt_template
            for feature in self.question_features:
                features[feature.replace(f".", "-")] = samples[feature][i]
                prompt_template = prompt_template.replace(f"{{{feature}}}", f"{{{feature.replace('.', '-')}}}")
            inputs.append({
                "prompt": prompt_template.format(**features),
                "answers": [self.label2answer[label] for label in range(len(self.label2answer))],
            })
        return {"idx": samples["idx"], "input": inputs, "target": samples["target"]}
    
    def transform(self, data: Dataset) -> Dataset:
        data = data.flatten().map(lambda samples: self._transform(samples), batched=True, batch_size=1000).select_columns(["idx", "input", "target"])
        return data
