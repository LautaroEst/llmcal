


import os
import pickle
from typing import Literal
import numpy as np
from torch.utils.data import TensorDataset as TD, DataLoader


class TensorDataset:

    def __init__(self, tensors_dir, tensor_name: Literal["logits", "embeddings"] = "logits"):
        self.tensors_dir = tensors_dir
        self.tensor_name = tensor_name

    def create_dataloader(self, split, batch_size, num_samples=None, shuffle=True, random_state=None):
        with open(os.path.join(self.tensors_dir, f"{split}_outputs.pkl"), "rb") as f:
            outputs = pickle.load(f)
        tensors = outputs[self.tensor_name]
        labels = outputs["labels"]
        dataset = TD(tensors, labels)
        if shuffle:
            rs = np.random.RandomState(random_state)
            if num_samples is None:
                num_samples = len(dataset)
            idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
            dataset = dataset.select(idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader