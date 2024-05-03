
import os
from pathlib import Path
from typing import Dict, List

import lightning as L
from torch.utils.data import DataLoader
from datasets import load_from_disk
from .utils import DynamicPaddingCollator
from ..prompt import PrefixPrompt
from ..datasets.utils import SUPPORTED_DATASETS, load_dataset
from .utils import LitGPTTokenizer
from litgpt import Config


class LanguageModelLitGPTFineTuningDataModule(L.LightningDataModule):

    def __init__(
        self,
        dataset: SUPPORTED_DATASETS,
        data_dir: str,
        data_cache_dir: str,
        tokenizer_dir: str,
        num_train_samples: int,
        num_val_samples: int,
        num_shots: int,
        preshots_template: str, 
        shots_template: str,
        postshots_template: str,
        shots_separator: str,
        answers_templates: List[str],
        batch_size: int = 32,
        random_state: int = 0,
    ):
        super().__init__()
        self.dataset_name = dataset
        self.data_dir = data_dir
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_shots = num_shots
        self.random_state = random_state

        # Init prompt
        self.prompt = PrefixPrompt(preshots_template, shots_template, postshots_template, shots_separator, answers_templates)
        self.data_cache_dir = data_cache_dir
        self.batch_size = batch_size

        # Init config and tokenizer
        if not os.path.exists(tokenizer_dir):
            if not os.path.exists(os.path.join(os.getenv("LIT_CHECKPOINTS"), tokenizer_dir)):
                raise FileNotFoundError(f"Checkpoint directory {tokenizer_dir} not found")
            self.tokenizer_dir = Path(os.getenv("LIT_CHECKPOINTS")) / tokenizer_dir
        else:
            self.tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer = LitGPTTokenizer(self.tokenizer_dir)
        self.config = Config.from_checkpoint(self.tokenizer_dir)


    def prepare_data(self):

        if os.path.exists(self.data_cache_dir):
            return
        
        # Create the cache directory
        os.makedirs(os.path.join(self.data_cache_dir, "train_data"), exist_ok=True)
        os.makedirs(os.path.join(self.data_cache_dir, "predict_data"), exist_ok=True)

        # Download the dataset
        train_datadict, predict_datadict, shots = load_dataset(
            dataset_name=self.dataset_name,
            data_dir=self.data_dir, 
            num_train_samples=self.num_train_samples, 
            num_val_samples=self.num_val_samples, 
            num_shots=self.num_shots, 
            random_state=self.random_state
        )

        # Fill the prompt and tokenize
        self.prompt.fit(shots)

        def transform(sample):
            prompt = self.prompt.transform(**sample)
            prompt_ids = self.tokenizer([prompt["prompt"]])["input_ids"][0,:]
            answers_ids = [self.tokenizer([ans])["input_ids"][0,1:] for ans in prompt["answers"]]
            return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}

        # Process the train dataset
        train_datadict["train"] = train_datadict["train"].map(transform)
        train_datadict["validation"] = train_datadict["validation"].map(transform)
        train_datadict.save_to_disk(os.path.join(self.data_cache_dir, "train_data"))

        # Process the original dataset
        for split in predict_datadict.keys():
            predict_datadict[split] = predict_datadict[split].map(transform)
        predict_datadict.save_to_disk(os.path.join(self.data_cache_dir, "predict_data"))
        

    def setup(self, stage):
        if stage == "fit":
            train_datadict = load_from_disk(os.path.join(self.data_cache_dir, "train_data"))
            self.train_data = train_datadict["train"].with_format("torch")
            self.val_data = train_datadict["validation"].with_format("torch")
        elif stage == "predict":
            predict_datadict = load_from_disk(os.path.join(self.data_cache_dir, "predict_data"))
            self.predict_data = {split: predict_datadict[split].with_format("torch") for split in predict_datadict.keys()}
            self.idx2split = {i: key for i, key in enumerate(sorted(predict_datadict.keys()))}
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collator)
    
    def val_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
    
    def predict_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return [
            DataLoader(self.predict_data[self.idx2split[idx]], batch_size=self.batch_size, shuffle=False, collate_fn=collator) \
            for idx in range(len(self.idx2split))
        ]