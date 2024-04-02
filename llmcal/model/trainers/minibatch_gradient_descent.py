
from functools import partial
from math import ceil
import os
import pickle
import time
from typing import Literal
import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from .utils import TBLogger
from datasets import Dataset

class MiniBatchGDTrainer:

    def __init__(
        self,
        fabric: L.Fabric,
        batch_size = 8,
        micro_batch_size = 2,
        learning_rate: float = 1,
        weight_decay: float = 0,
        max_epochs: int = 100,
        max_steps: int = 100000,
        warmup_steps: int = None,
        random_state = 0,
        loss: Literal["cross_entropy"] = "cross_entropy",
        val_interval: int = 1,
        checkpoint_interval: int = 1000,
        model_checkpoint_dir: str = None,
    ):
        self.fabric = fabric
        self.global_batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = (self.global_batch_size // fabric.world_size) // micro_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.random_state = random_state

        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval
        self.model_checkpoint_dir = model_checkpoint_dir
        self.loss = loss

        self.hyperparams = {
            "batch_size": batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
        }

    def get_lr_scheduler(self, optimizer, max_steps):
        # linear warmup followed by cosine annealing
        if self.warmup_steps is None:
            return None
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / self.warmup_steps)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - self.warmup_steps))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[self.warmup_steps])
            
    def fit(self, model, train_dataset, validation_dataset):

        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        logger = TBLogger(root_dir=self.model_checkpoint_dir)

        if self.max_epochs == 0:
            logger.finalize("success", time = 0)
            return self
        
        # Prepare the data
        self.set_collate_function(model, model.tokenizer)
        train_dataset = train_dataset.select_columns(["input","target"]).with_format("torch")
        validation_dataset = validation_dataset.select_columns(["input","target"]).with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset, batch_size=self.micro_batch_size, max_seq_length=model.max_seq_length, shuffle=True)
        validation_dataloader = self.create_dataloader(validation_dataset, batch_size=self.micro_batch_size, max_seq_length=model.max_seq_length, shuffle=False)

        # Find checkpoint and resume from there
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self.get_lr_scheduler(
            optimizer, 
            max_steps = min(self.max_epochs * (len(train_dataloader) // self.gradient_accumulation_steps), (self.max_steps or float("inf")))
        )
        state = {
            "model": model,
            "optimizer": optimizer, 
            "scheduler": scheduler,
            "epoch": 0,
            "global_step": 0,
            "step_inside_epoch": 0,
            "best_val_loss": float("inf"), 
            "last_train_loss": 0, 
            "last_val_loss": 0, 
        }
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
        
        # Setup logging, model and otpiumizer
        logger.log_hyperparams(self.hyperparams)
        model = self.fabric.setup_module(state["model"])
        optimizer = self.fabric.setup_optimizers(state["optimizer"])
        scheduler = state["scheduler"]
        
        # Epoch and step bars
        global_step_count = state["global_step"]
        epoch_step_count = state["step_inside_epoch"]
        iter_dataloader = iter(train_dataloader)
        iter_num = 0
        while iter_num < self.gradient_accumulation_steps * state["step_inside_epoch"]: # Advance to the last step
            next(iter_dataloader)
            iter_num += 1
        first_epoch = state["epoch"]
        max_steps = ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        
        # Start training
        last_train_loss = state["last_train_loss"]
        last_val_loss = state["last_val_loss"]
        best_val_loss = state["best_val_loss"]
        cum_train_loss = 0.
        cum_num_samples = 0
        model.train()
        start_time = time.time()
        try:
            for epoch in range(first_epoch, self.max_epochs):
                while iter_num < len(train_dataloader) and global_step_count < self.max_steps:
                    is_accumulating = (iter_num + 1) % self.gradient_accumulation_steps != 0
                    batch = next(iter_dataloader)
                    
                    # Forward-backward pass
                    inputs, targets = batch["input"], batch["target"]
                    inputs["labels"] = targets
                    batch_size = len(targets)
                    with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                        train_loss = model.train_step(**inputs, loss_fn=self.loss)
                        self.fabric.backward(train_loss)
                    cum_train_loss += train_loss.item() * batch_size
                    cum_num_samples += batch_size
                    
                    if not is_accumulating or iter_num == len(train_dataloader) - 1:

                        # Metrics
                        last_train_loss = cum_train_loss / cum_num_samples
                        metrics = {"loss/train": last_train_loss}

                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                        # Validate
                        if global_step_count % self.val_interval == 0:
                            last_val_loss = self._validate(model, validation_dataloader, self.loss)
                            metrics["loss/validation"] = last_val_loss
                            if last_val_loss < best_val_loss:
                                best_val_loss = last_val_loss
                                if global_step_count % self.checkpoint_interval == 0:
                                    self.fabric.save(os.path.join(self.model_checkpoint_dir, "best_model.ckpt"), {
                                        "model": model,
                                        "optimizer": optimizer,
                                        "scheduler": scheduler,
                                        "epoch": epoch,
                                        "global_step": global_step_count,
                                        "step_inside_epoch": epoch_step_count if epoch_step_count + 1 < ceil(len(train_dataloader) / self.gradient_accumulation_steps) else 0,
                                        "best_val_loss": best_val_loss,
                                        "last_train_loss": last_train_loss,
                                        "last_val_loss": last_val_loss,
                                    })

                        # Log
                        self.fabric.print(
                            f"Epoch {epoch:02}/{self.max_epochs-1:02} | "
                            f"Step {epoch_step_count:02}/{max_steps-1:02} | "
                            f"Train loss: {last_train_loss:.4f} | "
                            f"Val loss: {last_val_loss:.4f}"
                        )
                        logger.log_metrics(metrics, step=global_step_count)

                        # Save checkpoint
                        if global_step_count % self.checkpoint_interval == 0:
                            state["model"] = model
                            state["optimizer"] = optimizer
                            state["scheduler"] = scheduler
                            state["epoch"] = epoch
                            state["global_step"] = global_step_count + 1
                            state["step_inside_epoch"] = epoch_step_count + 1 if epoch_step_count + 1 < max_steps else 0
                            state["best_val_loss"] = best_val_loss
                            state["last_train_loss"] = last_train_loss
                            state["last_val_loss"] = last_val_loss
                            self.fabric.save(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)

                        global_step_count += 1
                        epoch_step_count += 1
                        cum_train_loss = 0.
                        cum_num_samples = 0
                    
                    iter_num += 1

                if epoch < self.max_epochs - 1:
                    iter_dataloader = iter(train_dataloader)
                    epoch_step_count = 0
                    iter_num = 0

            end_time = time.time()
            logger.finalize("success", time = offset_time + end_time - start_time)
            state["model"] = model
            state["optimizer"] = optimizer
            state["scheduler"] = scheduler
            state["epoch"] = epoch
            state["global_step"] = global_step_count
            state["step_inside_epoch"] = 0
            state["best_val_loss"] = best_val_loss
            state["last_train_loss"] = last_train_loss
            state["last_val_loss"] = last_val_loss
            self.fabric.save(os.path.join(self.model_checkpoint_dir, "last_model.ckpt"), state)

        except KeyboardInterrupt:
            print("Training interrupted. Exiting...")
            end_time = time.time()
            logger.finalize("interrupted", time = offset_time + end_time - start_time)

        return self
    
    @staticmethod
    @torch.inference_mode()
    def _validate(model, dataloader, loss):
        
        if loss == "cross_entropy":
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss function {loss} not implemented")
        
        model.eval()
        val_loss = 0
        progress_bar = tqdm(dataloader, leave=False, dynamic_ncols=True, desc="Validation")
        for i, batch in enumerate(progress_bar):
            inputs, targets = batch["input"], batch["target"]
            output = model.predict_step(**inputs)
            val_loss += loss(output["logits"], targets).item() * len(targets)
        val_loss /= len(dataloader.dataset)
        progress_bar.close()
        model.train()
        return val_loss
            

    def predict(self, model, dataset: Dataset, prefix: str = "") -> Dataset:
        self.set_collate_function(model, model.tokenizer)
        dataloader = self.create_dataloader(dataset, batch_size=self.micro_batch_size, max_seq_length=model.max_seq_length, shuffle=False)
        os.makedirs(os.path.join(self.model_checkpoint_dir), exist_ok=True)
        
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
                while batch_idx < len(dataloader):
                    batch = next(progress_bar)
                    batch_size = len(batch["target"])
                    output = model.predict_step(**batch["input"])
                    for key, value in output.items():
                        value = value.cpu()
                        if torch.is_floating_point(value):
                            value = value.type(torch.float32)
                        output[key] = value.numpy()
                    outputs.extend([{key: output[key][i] for key in output.keys()} for i in range(batch_size)])
                    batch_idx += 1
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
            outputs = outputs[:self.micro_batch_size * batch_idx]
            with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_predictions.pkl"), "wb") as f:
                pickle.dump(outputs, f)
            with open(os.path.join(self.model_checkpoint_dir, f"{prefix}_prediction.interrupted"), "w") as f:
                f.write(str(batch_idx))

        return dataset
        
    def set_collate_function(self, model, tokenizer):
        if model.__class__.__name__ in ["LoRALitGPTLanguageModel", "LitGPTLanguageModel"]:
            self._collate_function = partial(self._prompt_lm_collate_function, tokenizer=tokenizer)
        elif model.__class__.__name__ in ["LoRALitGPTPromptClassification", "LitGPTPromptClassification"]:
            self._collate_function = partial(self._prompt_classifier_collate_function, tokenizer=tokenizer)
        elif model.__class__.__name__ in ["LoRALitGPTSequenceClassification", "LitGPTSequenceClassification"]:
            self._collate_function = partial(self._prompt_sequence_classification_collate_function, tokenizer=tokenizer)
        else:
            raise ValueError(f"Model class {model.__class__.__name__} not supported")
        
    def create_dataloader(self, dataset, batch_size, max_seq_length, shuffle=False):

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=partial(self._collate_function, max_seq_length=max_seq_length),
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    def _prompt_classifier_collate_function(self, batch, tokenizer, max_seq_length):
        prompts = []
        answers_ids = []
        targets = []
        max_ans_length = 0
        for sample in batch:
            prompts.append(sample["input"]["prompt"])
            ans_ids = [tokenizer([answer])["input_ids"][0,1:] for answer in sample["input"]["answers"]]
            max_ans_length = max([max([len(ans_id) for ans_id in ans_ids]), max_ans_length])
            answers_ids.append(ans_ids)
            targets.append(sample["target"])
        tokenized_prompt = tokenizer(prompts, max_seq_length=max_seq_length-max_ans_length)
        return {
            "input": {
                "prompt_ids": tokenized_prompt["input_ids"],
                "prompt_mask": tokenized_prompt["attention_mask"],
                "answers_ids": answers_ids,
            },
            "target": torch.tensor(targets, dtype=torch.long)
        }
    
    def _prompt_lm_collate_function(self, batch):
        raise NotImplementedError
    
    def _prompt_sequence_classification_collate_function(self, batch, tokenizer, max_seq_length):
        prompts = []
        targets = []
        for sample in batch:
            prompts.append(sample["input"]["prompt"])
            targets.append(sample["target"])
        tokenized_prompt = tokenizer(prompts, max_seq_length=max_seq_length)
        return {
            "input": {
                "prompt_ids": tokenized_prompt["input_ids"],
                "prompt_mask": tokenized_prompt["attention_mask"],
            },
            "target": torch.tensor(targets, dtype=torch.long)
        }