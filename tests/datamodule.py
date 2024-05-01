
import os
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./MNIST", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.test_batch_idx = 0

    def prepare_data(self):
        # Download MNIST to data_dir
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=transforms.ToTensor())
        self.mnist_predict = MNIST(self.data_dir, train=False, transform=transforms.ToTensor())
        mnist_full = MNIST(self.data_dir, train=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
    
    def state_dict(self):
        return {"train_batch_idx": self.train_batch_idx, "val_batch_idx": self.val_batch_idx, "test_batch_idx": self.test_batch_idx}
    
    def load_state_dict(self, state_dict):
        self.train_batch_idx = state_dict["train_batch_idx"]
        self.val_batch_idx = state_dict["val_batch_idx"]
        self.test_batch_idx = state_dict["test_batch_idx"]


def main(fabric: L.Fabric):

    if os.path.exists("data_checkpoint.ckpt"):
        data_module = MNISTDataModule.load_from_checkpoint("data_checkpoint.ckpt")
    else:
        data_module = MNISTDataModule()
        if fabric.global_rank == 0:
            data_module.prepare_data()
    fabric.barrier()

    data_module.setup("predict")
    train_dataloader = data_module.train_dataloader()
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    iter_dataloader = iter(train_dataloader)

    for i in range(10):
        batch = next(iter_dataloader)
        data_module.train_batch_idx = i
        print(batch[0])

    if fabric.global_rank == 0:
        fabric.save("data_checkpoint.ckpt", state = data_module)
    
    


if __name__ == "__main__":
    fabric = L.Fabric(accelerator="cpu", strategy="auto", devices=2, num_nodes=1, precision=32)
    fabric.launch(main)