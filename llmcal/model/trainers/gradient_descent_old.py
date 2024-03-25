
import os
import time
from typing import Literal
import torch
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from .utils import TBLogger
from datasets import Dataset

class GradientDescentTrainer:

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
        
    def create_dataloader(self, dataset, batch_size=None):
        dataset = dataset.select_columns(["input","target"]).with_format("torch")
        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    def fit(self, model, train_dataset, validation_dataset):

        # Prepare the optimizer
        optimizer = LBFGS(
            model.parameters(),
            lr=self.learning_rate,
            max_iter=self.max_ls
        )
        optimizer = self.fabric.setup_optimizers(optimizer)

        # Find checkpoint and resume from there
        state = {"model": model, "optimizer": optimizer, "epoch": 0, "step": 0}
        if os.path.exists(os.path.join(self.model_checkpoint_dir, "training.success")):
            print("Found a successful training. Loading checkpoint...")
            self.fabric.load(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)
            return self
        if os.path.exists(os.path.join(self.model_checkpoint_dir, "training.interrupted")):
            print("Resuming training from last checkpoint...")
            self.fabric.load(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)
            with open(os.path.join(self.model_checkpoint_dir, "training.interrupted"), "r") as f:
                offset_time = float(f.read())
        else:
            offset_time = 0

        # Prepare the data
        
        validation_dataset = validation_dataset.select_columns(["input","target"]).with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset)
        validation_dataloader = self.create_dataloader(validation_dataset)

        def closure():
            optimizer.zero_grad()
            batch = next(iter(train_dataloader))
            targets = batch.pop("target")
            output = model(**batch["input"])
            loss = self.loss(output["logits"], targets)
            self.fabric.backward(loss)
            return loss

        # Configure training loop
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        model = state["model"]
        optimizer = state["optimizer"]
        epochs_bar = tqdm(
            range(state["epoch"], self.max_epochs), 
            dynamic_ncols=True,
            leave=False,
            initial=state["epoch"],
            total=self.max_epochs
        )
        logger = TBLogger(root_dir=self.model_checkpoint_dir)
        logger.log_hyperparams(self.hyperparams)
        start_time = time.time()
        best_val_loss = float("inf")
        last_train_loss = float("inf")
        
        # Start training
        try:
            for epoch in epochs_bar:

                # Train
                model.train()
                train_loss = optimizer.step(closure).item()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    batch = next(iter(validation_dataloader))
                    targets = batch.pop("target")
                    output = model(**batch["input"])
                    val_loss = self.loss(output["logits"], targets).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state["model"] = model
                    state["optimizer"] = optimizer
                    state["epoch"] = epoch
                    state["step"] = epoch
                    self.fabric.save(os.path.join(self.model_checkpoint_dir, "best_model.ckpt"), state)
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
            logger.finalize("success", time = offset_time + end_time - start_time)

        except KeyboardInterrupt:
            print("Training interrupted. Exiting...")
            end_time = time.time()
            logger.finalize("interrupted", time = offset_time + end_time - start_time)

        state["model"] = model
        state["optimizer"] = optimizer
        state["epoch"] = epoch
        state["step"] = epoch
        self.fabric.save(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)

        return self

    def predict(self, model, dataset):

        model.eval()
        dataset = dataset.select_columns(["input","target"]).with_format("torch")
        dataloader = self.create_dataloader(dataset, batch_size=self.val_batch_size)
        outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_size = len(batch.pop("target"))
                output = model(**batch["input"])
                for key, value in output.items():
                    value = value.cpu()
                    if torch.is_floating_point(value):
                        value = value.type(torch.float32)
                    output[key] = value.numpy()
                outputs.extend([{key: output[key][i] for key in output.keys()} for i in range(batch_size)])
        dataset = dataset.add_column("output", outputs)
        return dataset        