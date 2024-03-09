
import torch
from torch import nn, optim

from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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

    def predict(self, dataset):
        dataloader = self.create_dataloader(dataset)

        with torch.no_grad():
            outputs = defaultdict(list)
            for batch in tqdm(dataloader):
                idx = batch.pop("idx")
                output = self.model(**batch)
                for key, value in output.items():
                    outputs[key].append(value.cpu())
                outputs["idx"].append(idx.cpu())
            for key in output:
                outputs[key] = torch.cat(outputs[key], dim=0)

        return outputs

    def create_dataloader(self, dataset):
        
        def collate_function(batch):
            output = {}
            prompt = self.prompt.transform(batch)
            tokenized_prompt = self.tokenizer(prompt["prompt"])
            output["idx"] = batch["idx"]
            output["prompt_ids"] = tokenized_prompt["input_ids"]
            output["prompt_mask"] = tokenized_prompt["attention_mask"]
            output["answer_ids"] = [[self.tokenizer(answer)["input_ids"] for answer in answers] for answers in prompt["answers"]]
            return output

        dataloader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_function,
        )
        dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    