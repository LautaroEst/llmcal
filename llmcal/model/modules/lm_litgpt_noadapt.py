
import os
from pathlib import Path
from typing import Callable, List
import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from .lit_tokenizer import LitGPTTokenizer
from litgpt import Config
from ...prompt import PrefixPrompt


class DynamicPaddingCollator:

    def __init__(self, pad_token_id, max_seq_len):
        # batch = {"idx": ..., "prompt_ids": ..., "answers_ids": ...}
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        prompts_ids = []
        prompt_masks = []
        answers_ids = []
        max_ans_len = max([max([ans.shape[0] for ans in sample["answers_ids"]]) for sample in batch])
        max_prompt_len = min(self.max_seq_len - max_ans_len, max([sample["prompt_ids"].shape[0] for sample in batch]))
        for sample in batch:
            prompts_ids.append(torch.cat([torch.ones(max_prompt_len - sample["prompt_ids"].shape[0], dtype=torch.long) * self.pad_token_id, sample["prompt_ids"]]))
            prompt_masks.append(torch.cat([torch.zeros(max_prompt_len - sample["prompt_ids"].shape[0], dtype=torch.long), torch.ones(sample["prompt_ids"].shape[0], dtype=torch.long)]))
            answers_ids.append(sample["answers_ids"])
        return {
            "prompt_ids": torch.stack(prompts_ids),
            "prompt_masks": torch.stack(prompt_masks),
            "answers_ids": answers_ids
        }


class LanguageModelLitGPTNoAdaptation(L.LightningModule):

    def __init__(
        self,
        checkpoint_dir: str,
        preshots_template: str, 
        shots_template: str,
        postshots_template: str,
        shots_separator: str,
        answers_templates: List[str],
        data_load_fn: Callable,
        data_cache_dir: str,
        batch_size: int,
    ):
        super().__init__()
        self.checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
        self.config = Config.from_checkpoint(self.checkpoint_dir)
        self.tokenizer = LitGPTTokenizer(self.checkpoint_dir)
        self.prompt = PrefixPrompt(preshots_template, shots_template, postshots_template, shots_separator, answers_templates)
        self.data_load_fn = data_load_fn
        self.data_cache_dir = data_cache_dir
        self.batch_size = batch_size

    def prepare_data(self):

        if os.path.exists(self.data_cache_dir):
            return
        
        # Create the cache directory
        os.makedirs(self.data_cache_dir, exist_ok=True)

        # Download the dataset
        datadict, shots = self.data_load_fn()

        # Fill the prompt and tokenize
        self.prompt.fit(shots)

        def transform(sample):
            prompt = self.prompt.transform(**sample)
            prompt_ids = self.tokenizer([prompt["prompt"]])["input_ids"][0,:]
            answers_ids = [self.tokenizer([ans])["input_ids"][0,1:] for ans in prompt["answers"]]
            return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}

        # Process the dataset
        datadict["train"] = datadict["train"].map(transform)
        datadict["validation"] = datadict["validation"].map(transform)
        datadict["test"] = datadict["test"].map(transform)

        # Save to disk
        datadict.save_to_disk(self.data_cache_dir)

    def setup(self, stage):
        datadict = load_from_disk(self.data_cache_dir)
        if stage == "fit":
            self.train_data = datadict["train"].with_format("torch")
            self.val_data = datadict["validation"].with_format("torch")
        elif stage == "test":
            self.test_data = datadict["test"].with_format("torch")
        elif stage == "predict":
            self.predict_data = {"val": datadict["test"].with_format("torch"), "test": datadict["test"].with_format("torch")}
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collator)
    
    def val_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
    
    def test_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
    
    def predict_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return {
            split: DataLoader(self.predict_data[split], batch_size=self.batch_size, shuffle=False, collate_fn=collator) \
            for split in ["val", "test"]
        }
    
    def configure_model(self):
        # Init model here
        with self.trainer.init_module():
            self.l1 = torch.nn.Linear(28, 28)
        return