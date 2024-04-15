

import os
import numpy as np
from datasets import load_dataset, load_from_disk
from ..prompt import Prompt, DynamicPaddingCollator
from torch.utils.data import DataLoader


class SST2Dataset:

    def __init__(self, cache_dir, prompt_template, answers_templates, tokenizer_dir, max_seq_len):
        self.cache_dir = cache_dir
        self.prompt = Prompt(prompt_template, tokenizer_dir, max_seq_len, answers_templates)

    def _prepare_data_per_split(self, datadict, split):
        if os.path.exists(os.path.join(self.cache_dir, split)):
            return
        os.makedirs(os.path.join(self.cache_dir, split), exist_ok=True)
        dataset = datadict[split] if split != "val" else datadict["validation"]
        dataset = dataset.map(lambda sample: {"input": self.prompt.fill_and_tokenize(**{"sentence": sample["sentence"]})})
        dataset = dataset.remove_columns(["sentence"])
        dataset.save_to_disk(os.path.join(self.cache_dir, split))

    def prepare_data(self):
        datadict = load_dataset("nyu-mll/glue", "sst2")
        self._prepare_data_per_split(datadict, "train")
        self._prepare_data_per_split(datadict, "val")
        self._prepare_data_per_split(datadict, "test")

    def create_dataloader(self, split, batch_size, num_samples=None, shuffle=True, random_state=None):
        dataset = load_from_disk(os.path.join(self.cache_dir, split)).with_format("torch")
        if shuffle:
            rs = np.random.RandomState(random_state)
            if num_samples is None:
                num_samples = len(dataset)
            idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
            dataset = dataset.select(idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=DynamicPaddingCollator(self.prompt.tokenizer.pad_token_id))
        return dataloader
        