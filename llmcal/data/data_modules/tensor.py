import os

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from ..datasets.utils import load_dataset

class TensorDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(next(iter(self._data.values())))
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}
    

class TensorDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        data_cache_dir: str,
        num_train_samples: int,
        num_val_samples: int,
        random_state: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_cache_dir = data_cache_dir
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.random_state = random_state

    def prepare_data(self) -> None:
        train_datadict, predict_datadict, _ = load_dataset(
            dataset_name="tensor",
            data_dir=self.data_dir,
            num_train_samples=self.num_train_samples, 
            num_val_samples=self.num_val_samples, 
            num_shots=0, 
            random_state=self.random_state
        )
        torch.save(train_datadict, os.path.join(self.data_cache_dir, "train_datadict.pt"))
        torch.save(predict_datadict, os.path.join(self.data_cache_dir, "predict_datadict.pt"))

    def setup(self, stage):
        if stage == "fit":
            train_datadict = torch.load(os.path.join(self.data_cache_dir, "train_datadict.pt"))
            self.train_data = TensorDataset(train_datadict["train"])
            self.val_data = TensorDataset(train_datadict["validation"])
        elif stage == "predict":
            predict_datadict = torch.load(os.path.join(self.data_cache_dir, "predict_datadict.pt"))
            for split in predict_datadict.keys():
                predict_datadict[split] = TensorDataset(predict_datadict[split]) 
            self.predict_data = predict_datadict
            self.idx2split = {i: key for i, key in enumerate(sorted(predict_datadict.keys()))}
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=len(self.train_data), shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False)
    
    def predict_dataloader(self):
        return [
            DataLoader(self.predict_data[self.idx2split[idx]], batch_size=len(self.predict_data[self.idx2split[idx]]), shuffle=False) \
            for idx in range(len(self.idx2split))
        ]