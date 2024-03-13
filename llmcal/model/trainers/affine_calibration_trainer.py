
from collections import defaultdict
import json
import os
from typing import Literal
import torch
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L


class AffineCalibratorTrainer:

    def __init__(
        self,
        fabric: L.Fabric,
        val_batch_size = 8, 
        random_state = 0,
        learning_rate: float = 1,
        max_ls: int = 40,
        max_epochs: int = 100,
        tolerance: float = 1e-4,
        loss: Literal["cross_entropy", "mse"] = "cross_entropy",
        model_checkpoint_dir: str = None,
        **kwargs
    ):
        self.fabric = fabric
        self.val_batch_size = val_batch_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_ls = max_ls
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.model_checkpoint_dir = model_checkpoint_dir
        self.kwargs = kwargs

        if loss == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
            raise ValueError(f"Invalid loss: {loss}")

    def create_dataloader(self, dataset, shuffle=True, random_state=None):

        if shuffle:
            generator = torch.Generator()
            if random_state is not None:
                generator.manual_seed(random_state)
        else:
            generator = None

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=shuffle,
            generator=generator,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    def fit(self, model, train_dataset, validation_dataset):
        model.init_params(self.fabric)

        # Prepare the optimizer
        optimizer = LBFGS(
            self.model.parameters(),
            lr=self.learning_rate,
            max_iter=self.max_ls
        )
        optimizer = self.fabric.setup_optimizers(optimizer)

        train_dataset = train_dataset.flatten().select_columns(["input","target"]).with_format("torch")
        validation_dataset = validation_dataset.flatten().select_columns(["input","target"]).with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset, shuffle=True, random_state=self.random_state)
        validation_dataloader = self.create_dataloader(validation_dataset, shuffle=False)

        def closure():
            optimizer.zero_grad()
            batch = next(iter(train_dataloader))
            inputs, targets = batch["input"], batch["target"]
            logits = model(inputs)
            loss = self.loss(logits, targets)
            self.fabric.backward(loss)
            return loss

        # Start training
        history = defaultdict(list)
        try:
            os.makedirs(self.model_checkpoint_dir, exist_ok=True)
            epochs_bar = tqdm(range(self.max_epochs), leave=False)
            best_val_loss = float("inf")
            last_train_loss = float("inf")
            for epoch in epochs_bar:

                # Train
                model.train()
                train_loss = optimizer.step(closure).item()
                
                # Validate
                if validation_dataloader is not None:
                    
                    model.eval()
                    with torch.no_grad():
                        features, labels = next(iter(validation_dataloader))
                        logits = model(features)
                        val_loss = model.loss(logits, labels).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.state_dict(), os.path.join(self.model_checkpoint_dir, "best_model.pt"))
                    epochs_bar.set_description(f"Epoch {epoch + 1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

                else:
                    epochs_bar.set_description(f"Epoch {epoch + 1} | Train loss: {train_loss:.4f}")

                # Save history
                history["train_loss"].append(train_loss)
                history["train_step"].append(epoch)
                history["epoch"].append(epoch)
                if validation_dataloader is not None:
                    history["validation_loss"].append(val_loss)
                    history["validation_step"].append(epoch)

                # Save history
                with open(os.path.join(self.model_checkpoint_dir, "history.json"), "w") as f:
                    json.dump(history, f)

                # Check for convergence
                if abs(train_loss - last_train_loss) / max([1, train_loss, last_train_loss]) <= self.tolerance:
                    break
                last_train_loss = train_loss

        except KeyboardInterrupt:
            print("Training interrupted. Exiting...")

        torch.save(self.state_dict(), os.path.join(self.model_checkpoint_dir, "last_model.pt"))
        if validation_dataloader is not None:
            self.load_state_dict(torch.load(os.path.join(self.model_checkpoint_dir, "best_model.pt")))

        return self

        