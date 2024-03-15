
from typing import Dict, Optional
import torch
from torch import nn, optim

from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset


class PromptClassificationTrainer:

    def __init__(
        self, 
        fabric, 
        val_batch_size = 8, 
        random_state = 0,
        model_checkpoint_dir: Optional[str] = None,
        batch_size: Optional[int] = 8,
        learning_rate: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 0.0,
        warmup_steps: Optional[int] = 0,
        max_epochs: Optional[int] = 1000,
        gradient_accumulation_steps: Optional[int] = 1,
        lr_scheduler: Optional[str] = "linear",
        lr_scheduler_params: Optional[Dict] = {},
        optimizer: Optional[str] = "adam",
        optimizer_params: Optional[Dict] = {},
        **kwargs,
    ):
        self.fabric = fabric
        self.val_batch_size = val_batch_size
        self.random_state = random_state
        self.model_checkpoint_dir = model_checkpoint_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.kwargs = kwargs


    def fit(self, model, train_dataset, validation_dataset):
        model.init_params(self.fabric)
        self.tokenizer = model.tokenizer

        if self.max_epochs == 0:
            return self
        
        # TODO: implement fit method


    def predict(self, model, dataset: Dataset) -> Dataset:
        dataloader = self.create_dataloader(dataset, batch_size=self.val_batch_size, shuffle=False)

        outputs = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                output = model(prompt_ids=batch["prompt_ids"], prompt_mask=batch["prompt_mask"], answers_ids=batch["answers_ids"])
                for key, value in output.items():
                    value = value.cpu()
                    if torch.is_floating_point(value):
                        value = value.type(torch.float32)
                    output[key] = value
                outputs["output"].extend([{key: output[key][i] for key in output.keys()} for i in range(len(batch["idx"]))])
                outputs["idx"].append(batch["idx"].cpu())
                outputs["target"].append(batch["target"].cpu())
                outputs["input"].extend(batch["input"])
            
        outputs["idx"] = torch.cat(outputs["idx"], dim=0)
        outputs["target"] = torch.cat(outputs["target"], dim=0)
        outputs = Dataset.from_dict(dict(outputs))
        return outputs

    def create_dataloader(self, dataset, batch_size, shuffle=False, random_state=None):

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
            collate_fn=self._collate_function,
            generator=generator,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    def _collate_function(self, batch):
        keys = batch[0].keys()
        new_batch = defaultdict(list)
        for sample in batch:
            for key in keys:
                if key == "input":
                    new_batch["prompt"].append(sample["input"]["prompt"])
                    new_batch["answers_ids"].append([self.tokenizer([answer])["input_ids"][0,1:] for answer in sample["input"]["answers"]])
                    new_batch["input"].append(sample["input"])
                else:
                    new_batch[key].append(sample[key])
        tokenized_prompt = self.tokenizer(new_batch["prompt"])
        new_batch.pop("prompt")
        new_batch["prompt_ids"] = tokenized_prompt["input_ids"]
        new_batch["prompt_mask"] = tokenized_prompt["attention_mask"]
        new_batch["idx"] = torch.tensor(new_batch["idx"], dtype=torch.long)
        new_batch["target"] = torch.tensor(new_batch["target"], dtype=torch.long)
        return dict(new_batch)
    