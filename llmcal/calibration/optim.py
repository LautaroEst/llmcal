
from logging import getLogger
import os
from typing import Literal, Union
import lightning as L
import torch
from torch.optim import Adam, SGD, LBFGS
from tqdm import tqdm
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


class SGDMixin:

    def fit(
        self, 
        train_features: torch.Tensor, 
        train_labels: torch.Tensor,
        validation_features: torch.Tensor = None,
        validation_labels: torch.Tensor = None,
        optimizer: Literal["Adam", "SGD"] = "Adam",
        batch_size: Union[int,None] = None,
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        patience: int = 10,
        model_checkpoint_dir: str = None,
        **kwargs
    ):
        """
        Fit the calibrator to the training data using a gradient-based algorithm.

        Parameters
        ----------
        train_features : torch.Tensor(shape=(num_samples, num_features))
            Training feature vector.
        train_labels : torch.Tensor(shape=(num_samples,))
            Training labels.
        validation_features : torch.Tensor(shape=(num_samples, num_features)), optional
            Validation feature vector. If not passed, use the training features.
        validation_labels : torch.Tensor(shape=(num_samples,)), optional
            Validation labels. If not passed, use the training labels.
        optimizer : {"Adam", "SGD"}, optional
            Optimizer to use, by default "Adam"
        batch_size : int, optional
            Batch size to use, by default None
        max_epochs : int, optional
            Maximum number of epochs, by default 100
        learning_rate : float, optional
            Learning rate to use, by default 0.001
        weight_decay : float, optional
            Weight decay to use, by default 0
        tolerance : float, optional
            Tolerance to use for convergence, by default 1e-4
        patience : int, optional
            Patience to use for convergence, by default 10
        model_checkpoint_dir : str
            Directory to save the best model
        **kwargs : dict
            Fabric initialization arguments. Check https://lightning.ai/docs/fabric/stable/api/fabric_args.html for more information.

        Returns
        -------
        self
            Fitted calibrator.
        """
        logger = getLogger(__name__) # Initialize the logger

        # Check the inputs
        if (validation_features is None and validation_labels is not None) or (validation_features is not None and validation_labels is None):
            raise ValueError("Validation features and labels must be both None or both not None")
        elif validation_features is None or validation_labels is None:
            perform_validation = False
        else:
            perform_validation = True

        # Initialize the accelerator
        fabric = L.Fabric(**kwargs)
        fabric.launch()

        # Initialize the optimizer
        if optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "SGD":
            optimizer = SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        model, optimizer = fabric.setup(self, optimizer)

        # Initialize the dataloaders
        train_dataloader = self._prepare_dataloader(train_features, train_labels, fabric, batch_size=batch_size)
        if perform_validation:
            validation_dataloader = self._prepare_dataloader(validation_features, validation_labels, fabric, batch_size=batch_size)

        # Prepare history
        train_loss_history = []
        best_epoch_loss = train_loss = val_loss = float("inf")
        if perform_validation:
            val_loss_history = []

        # Start training
        p_counter = patience
        epochs_bar = tqdm(range(max_epochs), leave=False)
        for epoch in epochs_bar:

            if perform_validation:
                epochs_bar.set_description(f"Train Loss: {train_loss:.4f} / Validation Loss: {val_loss:.4f}")
            else:
                epochs_bar.set_description(f"Train Loss: {train_loss:.4f}")

            # Train
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()
                batch_logits = model(batch_features)
                batch_loss = self.loss(batch_logits, batch_labels)
                train_loss += batch_loss.item() * batch_features.shape[0]
                fabric.backward(batch_loss)
                optimizer.step()
            train_loss /= len(train_dataloader.dataset)
            train_loss_history.append(train_loss)
            loss = train_loss

            # Validation
            if perform_validation:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_features, batch_labels in validation_dataloader:
                        batch_logits = model(batch_features)
                        val_loss += self.loss(batch_logits, batch_labels).item() * batch_features.shape[0]
                    val_loss /= len(validation_dataloader.dataset)
                val_loss_history.append(val_loss)
                loss = val_loss

            # Check for convergence
            if loss > best_epoch_loss:
                if p_counter == 0:
                    logger.info(f"Converged after {epoch} epochs")
                    break
                else:
                    p_counter -= 1
            else:
                p_counter = patience

            # Save the best model
            if loss < best_epoch_loss:
                torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, "best_model_state_dict.pt"))
                best_epoch_loss = loss
            
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history if perform_validation else None
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
        train_features: torch.Tensor, 
        train_labels: torch.Tensor,
        validation_features: torch.Tensor = None,
        validation_labels: torch.Tensor = None,
        batch_size: Union[int,None] = None,
        learning_rate: float = 1,
        max_ls: int = 40,
        max_epochs: int = 100,
        tolerance: float = 1e-4,
        model_checkpoint_dir: str = None,
        **kwargs
    ):
        """
        Fit the calibrator to the training data using LBFGS-B algorithm.

        Parameters
        ----------
        train_features : torch.Tensor(shape=(num_samples, num_features))
            Training feature vector.
        train_labels : torch.Tensor(shape=(num_samples,))
            Training labels.
        validation_features : None
            Validation feature vector. Not used, only for compatibility with other optimizers.
        validation_labels : None
            Validation labels. Not used, only for compatibility with other optimizers.
        batch_size : int, optional
            Batch size to use, by default None
        learning_rate : float, optional
            Learning rate to use, by default 1
        max_ls : int, optional
            Maximum number of line search steps, by default 40
        max_epochs : int, optional
            Maximum number of epochs, by default 100
        tolerance : float, optional
            Tolerance to use for convergence, by default 1e-4
        **kwargs : dict
            Fabric initialization arguments. Check https://lightning.ai/docs/fabric/stable/api/fabric_args.html for more information.

        Returns
        -------
        self
            Fitted calibrator.
        """
        
        logger = getLogger(__name__)

        # Check the inputs
        if (validation_features is None and validation_labels is not None) or (validation_features is not None and validation_labels is None):
            raise ValueError("Validation features and labels must be both None or both not None")
        elif validation_features is None or validation_labels is None:
            perform_validation = False
        else:
            perform_validation = True

        fabric = L.Fabric(**kwargs)
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
        validation_dataloader = fabric.setup_dataloaders(
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
            for batch_features, batch_labels in train_dataloader:
                batch_logits = model(batch_features)
                loss = self.loss(batch_logits, batch_labels) * batch_features.shape[0] / len(train_dataloader.dataset)
                fabric.backward(loss)
            return loss
    
        # Prepare history
        train_loss_history = []
        best_epoch_loss = train_loss = val_loss = float("inf")
        if perform_validation:
            val_loss_history = []

        # Start training
        epochs_bar = tqdm(range(max_epochs), leave=False)
        for epoch in epochs_bar:

            if perform_validation:
                epochs_bar.set_description(f"Train Loss: {train_loss:.4f} / Validation Loss: {val_loss:.4f}")
            else:
                epochs_bar.set_description(f"Train Loss: {train_loss:.4f}")

            # Train
            model.train()
            train_loss = optimizer.step(closure)
            train_loss_history.append(train_loss.item())
            loss = train_loss

            # Validation
            if perform_validation:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_features, batch_labels in validation_dataloader:
                        batch_logits = model(batch_features)
                        val_loss += self.loss(batch_logits, batch_labels).item() * batch_features.shape[0]
                    val_loss /= len(validation_dataloader.dataset)
                val_loss_history.append(val_loss)
                loss = val_loss

            # Check for convergence
            if abs(loss - best_epoch_loss) / max([1, loss, best_epoch_loss]) <= tolerance:
                logger.info(f"Converged after {epoch} epochs")
                break

            # Save the best model
            if loss < best_epoch_loss:
                torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, "best_model_state_dict.pt"))
                best_epoch_loss = loss
            
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history if perform_validation else None
        if epoch == max_epochs - 1:
            logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        
        self.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, "best_model_state_dict.pt")))
        return self