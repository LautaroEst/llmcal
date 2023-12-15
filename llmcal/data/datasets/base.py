from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):

    idx2label = None
    features = None

    def __init__(self, dataset, subsample=None, random_state=0, sort_by_length=False):
        if subsample is not None:
            rs = np.random.RandomState(random_state)
            rndm_idx = rs.choice(len(dataset), subsample, replace=False)
            dataset = dataset.select(rndm_idx)
        
        if sort_by_length:
            dataset = dataset.map(lambda example: {"length": sum(len(example[feature]) for feature in self.features)})
            dataset = dataset.sort("length", reverse=True)
            dataset = dataset.remove_columns("length")
        
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]