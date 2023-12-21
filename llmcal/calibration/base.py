
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.optim import Adam, LBFGS
import lightning as L
from logging import getLogger
from tqdm import tqdm

from .feature_maps import init_feature_map

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
        self.feature_map = None

    def forward(self, features):
        raise NotImplementedError

    def calibrate(self, features, batch_size=None, accelerator="cpu", num_devices=1):
        if batch_size is None:
            batch_size = features.shape[0]

        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()

        fake_labels = torch.zeros(features.shape[0], dtype=torch.long)
        dataset = TensorDataset(features, fake_labels)
        dataloader = fabric.setup_dataloaders(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )
        model = fabric.setup(self)
        ft_map = fabric.setup(self.feature_map)

        model.eval()
        with torch.no_grad():
            logits = []
            for batch, _ in dataloader:
                batch_logits = model(ft_map(batch))
                logits.append(batch_logits)
            logits = torch.cat(logits, dim=0)
        return torch.log_softmax(logits, dim=1)

    def fit(self, train_features, train_labels, validation_features, validation_labels, feature_map, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(" + ", ".join([f"{k}={v}" for k, v in self.arguments.items()]) + ")"
    

class SGDCalibrator(BaseCalibrator):

    def fit(
        self, 
        train_features, 
        train_labels,
        validation_features=None,
        validation_labels=None,
        feature_map=None,
        accelerator="cpu",
        num_devices=1,
        optimizer=None,
        batch_size=None,
        max_epochs=100,
        learning_rate=0.01,
        weight_decay=0,
        tolerance=1e-4,
    ):

        if (validation_features is None and validation_labels is not None) or (validation_features is not None and validation_labels is None):
            raise ValueError("Validation features and labels must be both None or both not None")
        elif validation_features is None or validation_labels is None:
            validation_features = train_features
            validation_labels = train_labels

        self.feature_map = feature_map

        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()
        if optimizer is None or optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        model, optimizer = fabric.setup(self, optimizer)
        ft_map = fabric.setup(feature_map)

        train_dataloader = self._prepare_dataloader(train_features, train_labels, fabric, batch_size=batch_size)
        validation_dataloader = self._prepare_dataloader(validation_features, validation_labels, fabric, batch_size=batch_size)

        last_epoch_val_loss = float("inf")
        for epoch in tqdm(range(max_epochs), leave=False):

            # Train
            model.train()
            for batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()
                train_loss = self.loss(model(ft_map(batch_features)), batch_labels)
                fabric.backward(train_loss)
                optimizer.step()

            # Validation
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch_features, batch_labels in validation_dataloader:
                    validation_loss += self.loss(model(ft_map(batch_features)), batch_labels).item()
            validation_loss /= len(validation_dataloader)
            if abs(validation_loss - last_epoch_val_loss) < tolerance:
                break
            last_epoch_val_loss = validation_loss
            
        if epoch == max_epochs - 1:
            self.logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        return self

    def _prepare_dataloader(self, features, labels, fabric, batch_size=None):
        dataset = TensorDataset(features, labels)
        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=features.shape[0],
            generator=self.generator
        )

        if batch_size is None:
            batch_size = features.shape[0]
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
        validation_features=None,
        validation_labels=None,
        feature_map=None,
        accelerator="cpu",
        num_devices=1,
        batch_size=None,
        max_ls=40,
        max_epochs=100,
        tolerance=1e-4
    ):
        
        if (validation_features is None and validation_labels is not None) or (validation_features is not None and validation_labels is None):
            raise ValueError("Validation features and labels must be both None or both not None")
        elif validation_features is None or validation_labels is None:
            validation_features = train_features
            validation_labels = train_labels

        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()

        optimizer = LBFGS(
            self.parameters(),
            max_iter=max_ls
        )
        model, optimizer = fabric.setup(self, optimizer)
        ft_map = fabric.setup(feature_map)

        if batch_size is None:
            batch_size = train_features.shape[0]
        
        self.feature_map = feature_map

        train_dataloader, validation_dataloader = fabric.setup_dataloaders(
            DataLoader(
                TensorDataset(train_features, train_labels),
                batch_size=batch_size,
                shuffle=False
            ),
            DataLoader(
                TensorDataset(validation_features, validation_labels),
                batch_size=batch_size,
                shuffle=False
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )

        def closure():
            optimizer.zero_grad()
            logits = []
            labels = []
            for batch_features, batch_labels in train_dataloader:
                batch_logits = model(ft_map(batch_features))
                logits.append(batch_logits)
                labels.append(batch_labels)
            logits = torch.cat(logits, dim=0)
            labels = torch.cat(labels, dim=0)
            loss = self.loss(logits, labels)
            fabric.backward(loss)
            return loss
    
        last_epoch_val_loss = float("inf")
        for epoch in tqdm(range(max_epochs), leave=False):
            # Train
            self.train()
            optimizer.step(closure)

            # Validation
            self.eval()
            with torch.no_grad():
                logits = []
                labels = []
                for batch_features, batch_labels in validation_dataloader:
                    batch_logits = model(ft_map(batch_features))
                    logits.append(batch_logits)
                    labels.append(batch_labels)
                logits = torch.cat(logits, dim=0)
                labels = torch.cat(labels, dim=0)
                validation_loss = self.loss(logits, labels).item()
            if abs(validation_loss - last_epoch_val_loss) < tolerance:
                break
            last_epoch_val_loss = validation_loss
        
        if epoch == max_epochs - 1:
            self.logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        return self

