
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.optim import Adam
import lightning as L

from logging import getLogger

from tqdm import tqdm

class BaseCalibrator(nn.Module):

    def __init__(self, num_features, num_classes, generator=None):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.generator = generator
        self.logger = getLogger(__name__)

    def forward(self, features):
        raise NotImplementedError
    
    def fit(
        self, 
        features, 
        labels,
        accelerator="cpu",
        num_devices=1,
        optimizer=None,
        batch_size=32,
        num_epochs=100,
        learning_rate=0.01,
        weight_decay=0,
        tolerance=1e-4,
    ):
        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()
        if optimizer is None or optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        model, optimizer = fabric.setup(self, optimizer)

        dataset = TensorDataset(features, labels)
        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=features.shape[0],
            generator=self.generator
        )
        dataloader = fabric.setup_dataloaders(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )
        last_epoch_loss = float("inf")
        num_samples = features.shape[0]

        model.train()
        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_loss = 0
            for batch_features, batch_labels in dataloader:
                batch_len = batch_features.shape[0]
                optimizer.zero_grad()
                batch_loss = self.loss(model(batch_features), batch_labels)
                fabric.backward(batch_loss)
                optimizer.step()
                epoch_loss += batch_len * batch_loss.item()
            epoch_loss /= num_samples
            if abs(epoch_loss - last_epoch_loss) < tolerance:
                break
            last_epoch_loss = epoch_loss
        if epoch == num_epochs - 1:
            self.logger.warning(f"WARNING: Calibration did not converge after {num_epochs} epochs")
        return self

    def calibrate(self, features):
        self.eval()
        with torch.no_grad():
            return self(features)
    
