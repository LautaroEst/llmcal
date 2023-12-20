
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.optim import Adam, LBFGS
import lightning as L

from logging import getLogger

from tqdm import tqdm

class BaseCalibrator(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        generator = torch.Generator(device="cpu")
        if kwargs["random_state"] is not None:
            generator = generator.manual_seed(kwargs["random_state"])
        self.num_features = kwargs["num_features"]
        self.num_classes = kwargs["num_classes"]
        self.generator = generator
        self.logger = getLogger(__name__)

    def forward(self, features):
        raise NotImplementedError

    def calibrate(self, features):
        self.eval()
        with torch.no_grad():
            return torch.log_softmax(self(features), dim=1)

    def fit(self, train_features, train_labels, validation_features, validation_labels, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(" + ", ".join([f"{k}={v}" for k, v in self.arguments.items()]) + ")"
    

class SGDCalibrator(BaseCalibrator):

    def fit(
        self, 
        train_features, 
        train_labels,
        validation_features,
        validation_labels,
        accelerator="cpu",
        num_devices=1,
        optimizer=None,
        batch_size=32,
        max_epochs=100,
        learning_rate=0.01,
        weight_decay=0,
        tolerance=1e-4,
    ):
        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()
        if optimizer is None or optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        model, optimizer = fabric.setup(self, optimizer)

        train_dataloader = self._prepare_dataloader(train_features, train_labels, fabric, batch_size=batch_size)
        validation_dataloader = self._prepare_dataloader(validation_features, validation_labels, fabric, batch_size=batch_size)

        last_epoch_val_loss = float("inf")
        for epoch in tqdm(range(max_epochs), leave=False):

            # Train
            model.train()
            for batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()
                train_loss = self.loss(model(batch_features), batch_labels)
                fabric.backward(train_loss)
                optimizer.step()

            # Validation
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch_features, batch_labels in validation_dataloader:
                    validation_loss += self.loss(model(batch_features), batch_labels).item()
            validation_loss /= len(validation_dataloader)
            if abs(validation_loss - last_epoch_val_loss) < tolerance:
                break
            last_epoch_val_loss = validation_loss
            
        if epoch == max_epochs - 1:
            self.logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        return self

    def _prepare_dataloader(self, features, labels, fabric, batch_size=32):
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
        return dataloader


class LBFGSBCalibrator(BaseCalibrator):

    def fit(
        self, 
        train_features, 
        train_labels, 
        validation_features,
        validation_labels,
        max_ls=40,
        max_epochs=100,
        tolerance=1e-4
    ):

        optimizer = LBFGS(
            self.parameters(),
            max_iter=max_ls
        )

        def closure():
            optimizer.zero_grad()
            loss = self.loss(self(train_features), train_labels)
            loss.backward()
            return loss
    
        last_epoch_val_loss = float("inf")
        for epoch in tqdm(range(max_epochs), leave=False):
            # Train
            self.train()
            optimizer.step(closure)

            # Validation
            self.eval()
            with torch.no_grad():
                validation_loss = self.loss(self(validation_features), validation_labels)
            if abs(validation_loss - last_epoch_val_loss) < tolerance:
                break
            last_epoch_val_loss = validation_loss
        
        if epoch == max_epochs - 1:
            self.logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")

