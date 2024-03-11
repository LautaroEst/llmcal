
import torch
from torch import nn, optim

from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset


class PromptClassifierPredictor:

    def __init__(self, fabric, val_batch_size = 8, random_state = 0):
        self.fabric = fabric
        self.val_batch_size = val_batch_size
        self.random_state = random_state

    def fit(self, model, prompt, train_dataset, validation_dataset):
        self.prompt = prompt
        model.init_params(self.fabric)
        self.model = model
        self.tokenizer = model.tokenizer

    def predict(self, dataset: Dataset) -> Dataset:
        dataloader = self.create_dataloader(dataset)

        with torch.no_grad():
            outputs = defaultdict(list)
            for batch in tqdm(dataloader):
                output = self.model(prompt_ids=batch["prompt_ids"], prompt_mask=batch["prompt_mask"], answer_ids=batch["answer_ids"])
                for key, value in output.items():
                    value = value.cpu()
                    if torch.is_floating_point(value):
                        value = value.type(torch.float32)
                    outputs[key].append(value)
                for key in batch:
                    if key != "idx":
                        outputs[key].extend(batch[key])
                outputs["idx"].append(batch["idx"])
            for key in output:
                outputs[key] = torch.cat(outputs[key], dim=0)
            outputs["idx"] = torch.cat(outputs["idx"], dim=0)

        outputs = Dataset.from_dict(dict(outputs))
        return outputs

    def create_dataloader(self, dataset):
        
        def collate_function(batch):
            output = {}
            batch = {k: [sample[k] for sample in batch] for k in batch[0].keys()}
            prompt = self.prompt.transform(batch)
            tokenized_prompt = self.tokenizer(prompt["prompt"])
            output["idx"] = torch.tensor(batch["idx"])
            output["prompt_ids"] = tokenized_prompt["input_ids"]
            output["prompt_mask"] = tokenized_prompt["attention_mask"]
            output["answer_ids"] = [[self.tokenizer([answer])["input_ids"][0,1:] for answer in answers] for answers in prompt["answers"]]
            for key in batch:
                if key != "idx":
                    output[key] = batch[key]
            return output

        dataloader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_function,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    