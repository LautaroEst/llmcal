
from functools import partial
import os
import pickle
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
        batch_size = 8,
        learning_rate: float = 1,
        weight_decay: float = 0,
        max_epochs: int = 100,
        max_ls: int = 40,
        random_state = 0,
        loss: Literal["cross_entropy"] = "cross_entropy",
        val_interval: int = 1,
        checkpoint_interval: int = 1000,
        model_checkpoint_dir: str = None,
    ):
        self.fabric = fabric
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.max_ls = max_ls
        self.random_state = random_state

        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval
        self.model_checkpoint_dir = model_checkpoint_dir


        if loss == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss function {loss} not implemented")

        self.hyperparams = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "max_ls": self.max_ls,
        }

        
    def fit(self, model, train_dataset, validation_dataset):

        logger = TBLogger(root_dir=self.model_checkpoint_dir)

        if self.max_epochs == 0:
            logger.finalize("success", time = 0)
            return self
        
        # Find checkpoint and resume from there
        trainable_parameters = model.get_trainable_parameters()
        optimizer = LBFGS(
            trainable_parameters,
            lr=self.learning_rate,
            max_iter=self.max_ls
        )
        state = {
            "model": model, 
            "optimizer": optimizer, 
            "epoch": 0,
            "global_step": 0,
            "best_val_loss": float("inf"), 
            "last_train_loss": float("inf"), 
            "last_val_loss": float("inf"), 
        }
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
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
        train_dataset = train_dataset.select_columns(["input","target"]).with_format("torch")
        validation_dataset = validation_dataset.select_columns(["input","target"]).with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset)
        validation_dataloader = self.create_dataloader(validation_dataset)

        # Setup logging, model and otpiumizer
        logger.log_hyperparams(self.hyperparams)
        model = self.fabric.setup_module(state["model"])
        optimizer = self.fabric.setup_optimizers(state["optimizer"])

        # Epoch and step bars
        global step_count
        step_count = state["global_step"]
        epochs_bar = tqdm(
            range(state["epoch"], self.max_epochs), 
            dynamic_ncols=True,
            leave=False,
            initial=state["epoch"],
            total=self.max_epochs,
            desc="Epochs"
        )

        # Closure for the LBFGS optimizer
        def closure():
            global step_count
            optimizer.zero_grad()
            batch = next(iter(train_dataloader))
            targets = batch.pop("target")
            output = model(**batch["input"])
            loss = self.loss(output["logits"], targets)
            logger.log_metrics({"loss/train": loss.item()}, step=step_count)
            self.fabric.backward(loss)
            step_count += 1
            return loss

        # Start training
        train_loss_last_epoch = train_loss = state["last_train_loss"]
        val_loss = state["last_val_loss"]
        best_val_loss = state["best_val_loss"]
        model.train()
        start_time = time.time()
        try:
            for epoch in epochs_bar:

                # Forward, backward and step
                train_loss = optimizer.step(closure).item()
                metrics = {}
                
                # Validate
                if epoch % self.val_interval == 0:
                    val_loss = self._validate(model, validation_dataloader, self.loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        metrics["loss/validation"] = val_loss
                        if epoch % self.checkpoint_interval == 0:
                            self.fabric.save(os.path.join(self.model_checkpoint_dir, "best_model.ckpt"), {
                                "model": model,
                                "optimizer": optimizer,
                                "epoch": epoch,
                                "global_step": step_count,
                                "best_val_loss": best_val_loss,
                                "last_train_loss": train_loss,
                                "last_val_loss": val_loss,
                            })

                # Log
                epochs_bar.set_description(f"Epochs | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                logger.log_metrics(metrics, step=step_count)

                # Save checkpoint
                if epoch % self.checkpoint_interval == 0:
                    state["model"] = model
                    state["optimizer"] = optimizer
                    state["epoch"] = epoch
                    state["global_step"] = step_count
                    state["best_val_loss"] = best_val_loss
                    state["train_loss"] = train_loss
                    state["last_val_loss"] = val_loss
                    self.fabric.save(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)
                
                if train_loss_last_epoch == train_loss:
                    break

            end_time = time.time()
            epochs_bar.close()
            logger.finalize("success", time = offset_time + end_time - start_time)
            state["model"] = model
            state["optimizer"] = optimizer
            state["epoch"] = epoch
            state["global_step"] = step_count
            state["best_val_loss"] = best_val_loss
            state["last_train_loss"] = train_loss
            state["last_val_loss"] = val_loss
            self.fabric.save(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)

        except KeyboardInterrupt:
            print("Training interrupted. Exiting...")
            end_time = time.time()
            epochs_bar.close()
            logger.finalize("interrupted", time = offset_time + end_time - start_time)

        return self
    
    @staticmethod
    @torch.inference_mode()
    def _validate(model, dataloader, loss):
        model.eval()
        batch = next(iter(dataloader))
        targets = batch.pop("target")
        output = model(**batch["input"])
        val_loss = loss(output["logits"], targets).item()
        model.train()
        return val_loss
            

    def predict(self, model, dataset: Dataset, prefix: str = "") -> Dataset:
        model.eval()
        dataset = dataset.select_columns(["input","target"]).with_format("torch")
        dataloader = self.create_dataloader(dataset, batch_size=self.batch_size)

        try:
            batch_idx = 0
            progress_bar = iter(tqdm(dataloader, leave=False, dynamic_ncols=True))
            if os.path.exists(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.interrupted")):
                with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_predictions.pkl"), "rb") as f:
                    outputs = pickle.load(f)
                with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.interrupted"), "r") as f:
                    last_batch_idx = int(f.read())
                while batch_idx < last_batch_idx:
                    next(progress_bar)
                    batch_idx += 1
            else:
                outputs = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(progress_bar, start=batch_idx):
                    batch_size = len(batch["target"])
                    output = model(**batch["input"])
                    for key, value in output.items():
                        value = value.cpu()
                        if torch.is_floating_point(value):
                            value = value.type(torch.float32)
                        output[key] = value.numpy()
                    outputs.extend([{key: output[key][i] for key in output.keys()} for i in range(batch_size)])
                progress_bar.close()
            dataset = dataset.add_column("output", outputs)

            if os.path.exists(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.interrupted")):
                os.remove(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.interrupted"))
            if os.path.exists(os.path.join(self.model_checkpoint_dir, f"{prefix}_predictions.pkl")):
                os.remove(os.path.join(self.model_checkpoint_dir, f"{prefix}_predictions.pkl"))
            
            with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.success"), "w") as f:
                f.write("")
            
        except KeyboardInterrupt:
            print("Prediction interrupted. Exiting...")
            outputs = outputs[:self.batch_size * batch_idx]
            os.makedirs(os.path.join(self.model_checkpoint_dir), exist_ok=True)
            with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_predictions.pkl"), "wb") as f:
                pickle.dump(outputs, f)
            with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.interrupted"), "w") as f:
                f.write(str(batch_idx))

        return dataset
        
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