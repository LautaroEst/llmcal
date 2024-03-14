
import os
import time
from typing import Literal
import torch
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from .callbacks import TBLogger, ModelCheckpoint


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

        self.hyperparams = {
            "learning_rate": self.learning_rate,
            "max_ls": self.max_ls,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
        }

        if loss == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
            raise ValueError(f"Invalid loss: {loss}")
        
        self.checkpoint_callback = ModelCheckpoint(fabric, model_checkpoint_dir, self.hyperparams)

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

        # Prepare the data
        train_dataset = train_dataset.flatten().select_columns(["input","target"]).with_format("torch")
        validation_dataset = validation_dataset.flatten().select_columns(["input","target"]).with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset, shuffle=False)
        validation_dataloader = self.create_dataloader(validation_dataset, shuffle=False)

        # Prepare the optimizer
        def closure():
            optimizer.zero_grad()
            batch = next(iter(train_dataloader))
            inputs, targets = batch["input"], batch["target"]
            logits = model(inputs)
            loss = self.loss(logits, targets)
            self.fabric.backward(loss)
            return loss
        optimizer = LBFGS(
            model.parameters(),
            lr=self.learning_rate,
            max_iter=self.max_ls
        )
        optimizer = self.fabric.setup_optimizers(optimizer)

        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        state = {"model": model, "optimizer": optimizer, "epoch": 0, "step": 0, "offset_time": 0.}
        state, version = self.checkpoint_callback.find_last_version(state)
        model = state["model"]
        optimizer = state["optimizer"]
        epochs_bar = tqdm(
            range(state["epoch"], self.max_epochs), 
            dynamic_ncols=True,
            leave=False,
            initial=state["epoch"],
            total=self.max_epochs
        )
        logger = TBLogger(root_dir=self.model_checkpoint_dir, version=version)
        logger.log_hyperparams(self.hyperparams)

        start_time = time.time()
        try:
            best_val_loss = float("inf")
            last_train_loss = float("inf")
            for epoch in epochs_bar:

                # Train
                model.train()
                train_loss = optimizer.step(closure).item()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    batch = next(iter(validation_dataloader))
                    inputs, targets = batch["input"], batch["target"]
                    logits = model(inputs)
                    val_loss = self.loss(logits, targets).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state["model"] = model
                    state["optimizer"] = optimizer
                    state["epoch"] = epoch
                    state["step"] = epoch
                    self.fabric.save(os.path.join(self.model_checkpoint_dir, version, "best_model.ckpt"), state)
                epochs_bar.set_description(f"Epoch {epoch + 1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

                # Save history
                logger.log_metrics({
                    "loss/train": train_loss,
                    "loss/validation": val_loss,
                }, step=epoch)

                # Check for convergence
                if abs(train_loss - last_train_loss) / max([1, train_loss, last_train_loss]) <= self.tolerance:
                    break
                last_train_loss = train_loss

            end_time = time.time()
            logger.finalize("success", time = state["offset_time"] + end_time - start_time)

        except KeyboardInterrupt:
            print("Training interrupted. Exiting...")
            end_time = time.time()
            logger.finalize("interrupted", time = state["offset_time"] + end_time - start_time)

        state["model"] = model
        state["optimizer"] = optimizer
        state["epoch"] = epoch
        state["step"] = epoch
        state["offset_time"] += end_time - start_time
        self.fabric.save(os.path.join(self.model_checkpoint_dir, version, "last_model.ckpt"), state)

        return self

    def predict(self, model, dataset):

        # TODO: Implement this method
        pass
        