
from logging import getLogger
import os
import lightning as L
import torch
from torch.optim import Adam, SGD, LBFGS
from tqdm import tqdm
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


class SGDMixin:

    def fit(
        self, 
        train_features, 
        train_labels,
        validation_features=None,
        validation_labels=None,
        accelerator="cpu",
        num_devices=1,
        optimizer=None,
        batch_size=None,
        max_epochs=100,
        learning_rate=0.01,
        weight_decay=0,
        tolerance=1e-4,
        model_checkpoint_dir=None
    ):

        logger = getLogger(__name__)

        if (validation_features is None and validation_labels is not None) or (validation_features is not None and validation_labels is None):
            raise ValueError("Validation features and labels must be both None or both not None")
        elif validation_features is None or validation_labels is None:
            perform_validation = False
        else:
            perform_validation = True

        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()
        if optimizer is None or optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "mini-batch-GD":
            optimizer = SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        model, optimizer = fabric.setup(self, optimizer)

        train_dataloader = self._prepare_dataloader(train_features, train_labels, fabric, batch_size=batch_size)
        if perform_validation:
            validation_dataloader = self._prepare_dataloader(validation_features, validation_labels, fabric, batch_size=batch_size)

        epochs_bar = tqdm(range(max_epochs), leave=False)
        loss_history = []
        last_epoch_loss = float("inf")
        for epoch in epochs_bar:

            epochs_bar.set_description(f"Loss: {last_epoch_loss:.4f}")

            # Train
            model.train()
            loss = 0
            for batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()
                batch_loss = self.loss(model(batch_features), batch_labels)
                fabric.backward(batch_loss)
                optimizer.step()
                loss += batch_loss.item() * batch_features.shape[0]
            loss /= len(train_dataloader.dataset)

            # Validation
            if perform_validation:
                model.eval()
                loss = 0
                with torch.no_grad():
                    for batch_features, batch_labels in validation_dataloader:
                        loss += self.loss(model(batch_features), batch_labels).item() * batch_features.shape[0]
                    loss /= len(validation_dataloader.dataset)
            if abs(loss - last_epoch_loss) / max([abs(loss), last_epoch_loss, 1]) <= tolerance:
                break

            if loss < last_epoch_loss:
                torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, "best_model_state_dict.pt"))

            loss_history.append(loss)
            last_epoch_loss = loss
            
        self.loss_history = loss_history
        if epoch == max_epochs - 1:
            logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        
        self.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, "best_model_state_dict.pt")))
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


class LBFGSBMixin:

    def fit(
        self, 
        train_features, 
        train_labels, 
        accelerator="cpu",
        num_devices=1,
        batch_size=None,
        learning_rate=1,
        max_ls=40,
        max_epochs=100,
        tolerance=1e-4
    ):
        
        logger = getLogger(__name__)

        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()

        optimizer = LBFGS(
            self.parameters(),
            lr=learning_rate,
            max_iter=max_ls
        )
        model, optimizer = fabric.setup(self, optimizer)

        if batch_size is None:
            batch_size = train_features.shape[0]
        
        train_dataloader = fabric.setup_dataloaders(
            DataLoader(
                TensorDataset(train_features, train_labels),
                batch_size=batch_size,
                shuffle=False
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )

        def closure():
            optimizer.zero_grad()
            for batch_features, batch_labels in train_dataloader:
                batch_logits = model(batch_features)
                loss = self.loss(batch_logits, batch_labels) * batch_features.shape[0] / len(train_dataloader.dataset)
                fabric.backward(loss)
            return loss
    
        epochs_bar = tqdm(range(max_epochs), leave=False)
        loss_history = []
        last_epoch_loss = float("inf")
        for epoch in epochs_bar:

            epochs_bar.set_description(f"Loss: {last_epoch_loss:.4f}")

            # Train
            self.train()
            loss = optimizer.step(closure).item()

            if abs(loss - last_epoch_loss) / max([abs(loss), last_epoch_loss, 1]) <= tolerance:
                break
            loss_history.append(loss)
            last_epoch_loss = loss
        
        self.loss_history = loss_history
        if epoch == max_epochs - 1:
            logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        return self