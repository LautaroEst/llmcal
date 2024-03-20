
from functools import partial
import os
import time
from typing import Literal
import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from .utils import TBLogger
from datasets import Dataset

class FullFinetuningTrainer:

    def __init__(
        self,
        fabric: L.Fabric,
        batch_size = 8,
        micro_batch_size = 2,
        learning_rate: float = 1,
        weight_decay: float = 0,
        max_epochs: int = 100,
        tolerance: float = 1e-4,
        random_state = 0,
        loss: Literal["cross_entropy", "mse"] = "cross_entropy",
        log_interval: int = 1,
        val_interval: int = 1,
        model_checkpoint_dir: str = None,
    ):
        self.fabric = fabric
        self.global_batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = (self.global_batch_size // fabric.world_size) // micro_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.random_state = random_state

        self.log_interval = log_interval
        self.val_interval = val_interval
        self.model_checkpoint_dir = model_checkpoint_dir

        self.hyperparams = {
            "batch_size": batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
        }

        if loss == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
            raise ValueError(f"Invalid loss: {loss}")
        
    def fit(self, model, train_dataset, validation_dataset):

        if self.max_epochs == 0:
            return self

        # Prepare the optimizer
        optimizer = SGD(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

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
        self.set_collate_function(model, model.tokenizer)
        train_dataset = train_dataset.select_columns(["input","target"]).with_format("torch")
        validation_dataset = validation_dataset.select_columns(["input","target"]).with_format("torch")
        train_dataloader = self.create_dataloader(train_dataset, batch_size=self.micro_batch_size, max_seq_length=model.max_seq_length, shuffle=True, random_state=self.random_state)
        validation_dataloader = self.create_dataloader(validation_dataset, batch_size=self.micro_batch_size, max_seq_length=model.max_seq_length, shuffle=False)

        # Configure training loop
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        model = self.fabric.setup_module(state["model"])
        optimizer = self.fabric.setup_optimizers(state["optimizer"])
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
        step_count = 0
        best_val_loss = float("inf")
        last_train_loss = float("inf")
        model.train()

        # Start training
        try:
            for epoch in epochs_bar:

                # Train
                for i, batch in enumerate(train_dataloader):
                    is_accumulating = (i + 1) % self.gradient_accumulation_steps != 0
                    inputs, targets = batch["input"], batch["target"]
                    
                    with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                        output = model(**inputs)
                        train_loss = self.loss(output["logits"], targets)
                        self.fabric.backward(train_loss / self.gradient_accumulation_steps)

                    # TODO: register loss

                    if not is_accumulating:
                        optimizer.step()
                        optimizer.zero_grad()
                        step_count += 1

                    # Validate
                    if step_count % self.val_interval == 0:
                        val_loss = self._validate(model, validation_dataloader, self.loss)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            state["model"] = model
                            state["optimizer"] = optimizer
                            state["epoch"] = epoch
                            state["step"] = epoch
                            self.fabric.save(os.path.join(self.model_checkpoint_dir, "best_model.ckpt"), state)

                    # Log
                    if step_count % self.log_interval == 0:
                        epochs_bar.set_description(f"Epoch {epoch + 1} | Step {step_count + 1} | Train loss: {train_loss.item():.4f} | Val loss: {val_loss:.4f}")
                    logger.log_metrics({
                        "loss/train": train_loss,
                        "loss/validation": val_loss,
                    }, step=step_count)

                    # Check for convergence
                    if not is_accumulating:
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
    def _validate(model, dataloader, loss):
        model.eval()
        val_loss = 0
        for i, batch in enumerate(dataloader):
            inputs, targets = batch["input"], batch["target"]
            output = model(**inputs)
            val_loss += loss(output["logits"], targets).item() * len(targets)
        val_loss /= len(dataloader.dataset)
        return val_loss
            

    def predict(self, model, dataset: Dataset) -> Dataset:
        self.set_collate_function(model, model.tokenizer)
        dataloader = self.create_dataloader(dataset, batch_size=self.micro_batch_size, max_seq_length=model.max_seq_length, shuffle=False)

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
        
    def set_collate_function(self, model, tokenizer):
        if model.__class__.__name__ == "LitGPTLanguageModel":
            self._collate_function = partial(self._prompt_lm_collate_function, tokenizer=tokenizer)
        elif model.__class__.__name__ == "LitGPTPromptClassifier":
            self._collate_function = partial(self._prompt_classifier_collate_function, tokenizer=tokenizer)
        elif model.__class__.__name__ == "LitGPTSequenceClassification":
            self._collate_function = partial(self._prompt_sequence_classification_collate_function, tokenizer=tokenizer)
        else:
            raise ValueError(f"Model class {model.__class__.__name__} not supported")

    def create_dataloader(self, dataset, batch_size, max_seq_length, shuffle=False, random_state=None):

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
            collate_fn=partial(self._collate_function, max_seq_length=max_seq_length),
            generator=generator,
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