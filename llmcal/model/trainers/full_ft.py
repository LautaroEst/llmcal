
from typing import Dict, Optional
import torch
from torch import nn, optim

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset


class FullFinetuningTrainer:

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
        self.tokenizer = model.tokenizer

        if self.max_epochs == 0:
            return self
        
        # TODO: implement fit method
        self.set_collate_function(model)


    def predict(self, model, dataset: Dataset) -> Dataset:
        self.set_collate_function(model)
        dataloader = self.create_dataloader(dataset, batch_size=self.val_batch_size, shuffle=False)

        outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_size = len(batch.pop("target"))
                output = model(**batch)
                for key, value in output.items():
                    value = value.cpu()
                    if torch.is_floating_point(value):
                        value = value.type(torch.float32)
                    output[key] = value.numpy()
                outputs.extend([{key: output[key][i] for key in output.keys()} for i in range(batch_size)])
        dataset = dataset.add_column("output", outputs)
        return dataset

    def set_collate_function(self, model):
        if model.__class__.__name__ == "LitGPTLanguageModel":
            self._collate_function = self._prompt_lm_collate_function
        elif model.__class__.__name__ == "LitGPTPromptClassifier":
            self._collate_function = self._prompt_classifier_collate_function
        elif model.__class__.__name__ == "LitGPTSequenceClassification":
            self._collate_function = self._prompt_sequence_classification_collate_function
        else:
            raise ValueError(f"Model class {model.__class__.__name__} not supported")

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

    def _prompt_classifier_collate_function(self, batch):
        prompts = []
        answers_ids = []
        targets = []
        for sample in batch:
            prompts.append(sample["input"]["prompt"])
            answers_ids.append([self.tokenizer([answer])["input_ids"][0,1:] for answer in sample["input"]["answers"]])
            targets.append(sample["target"])
        tokenized_prompt = self.tokenizer(prompts)
        return {
            "prompt_ids": tokenized_prompt["input_ids"],
            "prompt_mask": tokenized_prompt["attention_mask"],
            "answers_ids": answers_ids,
            "target": torch.tensor(targets, dtype=torch.long)
        }
    
    def _prompt_lm_collate_function(self, batch):
        raise NotImplementedError
    
    def _prompt_sequence_classification_collate_function(self, batch):
        raise NotImplementedError
    