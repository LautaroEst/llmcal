
import json
import os
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from tqdm import tqdm


class LBFGSMixin:

    def fit(
        self,
        train_features: torch.FloatTensor,
        train_labels: torch.LongTensor,
        validation_features: torch.FloatTensor = None,
        validation_labels: torch.LongTensor = None,
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        max_ls: int = 40,
        tolerance: float = 1e-4,
        model_checkpoint_dir: str = None,
        **kwargs
    ):
        """
        Fits the calibrator to the given features and labels.

        Parameters
        ----------
        train_features : torch.FloatTensor
            Train features.
        train_labels : torch.LongTensor
            Train labels.
        validation_features : torch.FloatTensor, optional
            Validation features, by default None
        validation_labels : torch.LongTensor, optional
            Validation labels, by default None
        max_epochs : int, optional
            Maximum number of epochs, by default 100
        batch_size : int, optional
            Batch size, by default 32
        learning_rate : float, optional
            Learning rate, by default 0.001
        weight_decay : float, optional
            Weight decay, by default 0.0
        max_ls : int, optional
            Maximum number of line search iterations, by default 40
        tolerance : float, optional
            Tolerance for convergence, by default 1e-4
        model_checkpoint_dir : str, optional
            Directory to save the model checkpoints, by default None
        """

        # Setup fabric
        fabric = L.Fabric(**kwargs)
        fabric.launch()

        # Setup logging
        hparams = {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_ls": max_ls,
            "tolerance": tolerance,
        }

        # Setup optimizer
        optimizer = optim.LBFGS(
            self.parameters(),
            lr=learning_rate,
            max_iter=max_ls
        )
        model, optimizer = fabric.setup(self, optimizer)

        # Setup dataloaders
        if batch_size is None:
            batch_size = train_features.shape[0]

        train_dataset = TensorDataset(train_features, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        history = {"train_loss": [], "epoch": []}

        if validation_features is not None and validation_labels is not None:
            validation_dataset = TensorDataset(validation_features, validation_labels)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
            train_dataloader, validation_dataloader = fabric.setup_dataloaders(
                train_dataloader, 
                validation_dataloader,
                use_distributed_sampler=True,
                move_to_device=True
            )
            history["validation_loss"] = []
        else:
            train_dataloader = fabric.setup_dataloaders(train_dataloader, use_distributed_sampler=True, move_to_device=True)
            validation_dataloader = None

        # Define closure for training
        def closure():
            optimizer.zero_grad()
            loss_history = []
            for batch_features, batch_labels in train_dataloader:
                batch_logits = model(batch_features)
                loss = self.loss(batch_logits, batch_labels) * batch_features.shape[0] / len(train_dataloader.dataset)
                fabric.backward(loss)
                loss_history.append(loss.item())
            params_norm = torch.tensor(0.0, device=fabric.device)
            for param in model.parameters():
                params_norm += torch.norm(param)
            loss = weight_decay * params_norm
            fabric.backward(loss)
            loss = sum(loss_history) + loss.item()
            return loss
        
        # Start training
        try:
            os.makedirs(model_checkpoint_dir, exist_ok=True)
            model.train()
            epochs_bar = tqdm(range(max_epochs), leave=False)
            best_val_loss = float("inf")
            for epoch in epochs_bar:

                # Train
                loss = optimizer.step(closure)
                epochs_bar.set_description(f"Epoch {epoch + 1} | Loss: {loss:.4f}")
                history["train_loss"].append(loss)
                
                # Validate
                if validation_dataloader is not None:
                    model.eval()
                    with torch.no_grad():
                        loss = 0
                        for batch_features, batch_labels in validation_dataloader:
                            batch_logits = model(batch_features)
                            loss += self.loss(batch_logits, batch_labels).item() * batch_features.shape[0] / len(validation_dataloader.dataset)
                        params_norm = torch.tensor(0.0, device=fabric.device)
                        for param in model.parameters():
                            params_norm += torch.norm(param)
                        loss += weight_decay * params_norm
                        history["validation_loss"].append(loss.item())
                    model.train()
                    if loss < best_val_loss:
                        best_val_loss = loss
                        torch.save(self.state_dict(), os.path.join(model_checkpoint_dir, "model.pt"))
            
                history["epoch"].append(epoch)
        
        except KeyboardInterrupt:
            print("Training interrupted. Exiting...")

        # Load best model
        self.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, "model.pt")))

        # Save history
        with open(os.path.join(model_checkpoint_dir, "history.json"), "w") as f:
            json.dump(history, f)
        