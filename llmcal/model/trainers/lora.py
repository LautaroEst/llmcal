
import os
import time
from typing import Literal
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from .utils import TBLogger
from datasets import Dataset

class LoRATrainer:

    def __init__(
        self,
        fabric: L.Fabric,
        val_batch_size = 8, 
        random_state = 0,
        learning_rate: float = 1,
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
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.model_checkpoint_dir = model_checkpoint_dir
        self.kwargs = kwargs

        self.hyperparams = {
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
        }

        if loss == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
            raise ValueError(f"Invalid loss: {loss}")
        
    def create_dataloader(self, dataset, batch_size=32, shuffle=True, random_state=None):
        if batch_size is None:
            batch_size = len(dataset)

        if shuffle:
            generator = torch.Generator()
            if random_state is not None:
                generator.manual_seed(random_state)
        else:
            generator = None                

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    def fit(self, model, train_dataset, validation_dataset):

        # Prepare the optimizer
        optimizer = AdamW(
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
            state["model"].init_params(self.fabric)
            offset_time = 0

        # Prepare the data
        train_dataset = train_dataset.select_columns(["input","target"]).flatten().with_format("torch")
        validation_dataset = validation_dataset.select_columns(["input","target"]).flatten().with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset)
        validation_dataloader = self.create_dataloader(validation_dataset)

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
        model.train()

        # Start training
        try:
            for epoch in epochs_bar:

                # Train
                for i, batch in enumerate(train_dataloader):
                    inputs = {k: batch[f"input.{k}"] for k in batch if f"input.{k}" in batch}
                    targets = batch["target"]
                    logits = model(**inputs)
                    train_loss = self.loss(logits, targets)
                    self.fabric.backward(train_loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    epochs_bar.set_description(f"Epoch {epoch + 1} | Train loss: {train_loss.item():.4f} | Val loss: {val_loss:.4f}")

                # Validate
                val_loss = self._validate(self.fabric, model, validation_dataloader, self.loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state["model"] = model
                    state["optimizer"] = optimizer
                    state["epoch"] = epoch
                    state["step"] = epoch
                    self.fabric.save(os.path.join(self.model_checkpoint_dir, "best_model.ckpt"), state)
                
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
    
    @staticmethod
    @torch.no_grad()
    def _validate(fabric, model, dataloader, loss):
        model.eval()
        val_loss = 0
        for i, batch in enumerate(dataloader):
            inputs = {k: batch[f"input.{k}"] for k in batch if f"input.{k}" in batch}
            targets = batch["target"]
            logits = model(**inputs)
            val_loss += loss(logits, targets).item() * len(targets)
        val_loss /= len(dataloader.dataset)
        return val_loss.item()
            

    def predict(self, model, dataset):

        model.eval()
        dataset = dataset.flatten().select_columns(["idx", "input","target"]).with_format("torch")
        dataloader = self.create_dataloader(dataset, batch_size=self.val_batch_size)
        outputs = {"idx": [], "input": [], "target": [], "output": []}
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs, targets = batch["input"], batch["target"]
                logits = model(inputs)
                outputs["idx"].append(batch["idx"].cpu())
                outputs["input"].append(batch["input"].cpu())
                outputs["target"].append(targets.cpu())
                outputs["output"].append(logits.cpu())

        outputs["idx"] = torch.cat(outputs["idx"], dim=0)
        outputs["input"] = torch.cat(outputs["input"], dim=0)
        outputs["target"] = torch.cat(outputs["target"], dim=0)
        outputs["output"] = torch.cat(outputs["output"], dim=0)
        outputs = Dataset.from_dict(dict(outputs))
        return outputs
        