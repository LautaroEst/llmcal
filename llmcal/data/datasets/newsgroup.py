
import os
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from ..prompt import Prompt



class TwentyNewsGroupDataset:

    def __init__(self, cache_dir, max_seq_len):
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len

    def _prepare_data_per_split(self, datadict, split, prompt):
        if split == "test":
            dataset = datadict["test"]
            dataset = dataset.add_column("idx", list(range(len(dataset))))
            dataset = dataset.select(list(range(100))) # TEMP
        elif split in ["train", "val"]:
            dataset = datadict["train"]
            dataset = dataset.add_column("idx", list(range(len(dataset))))
            rs = np.random.RandomState(78)
            idx = rs.permutation(len(dataset))
            if split == "val":
                # idx = idx[:1000]
                idx = idx[:100] # TEMP
            elif split == "train":
                # idx = idx[1000:11000]
                idx = idx[100:200] # TEMP
            dataset = dataset.select(idx)
        dataset = dataset.rename_column("label","target")
        dataset = dataset.map(lambda sample: {"input": prompt.fill_and_tokenize(sample["text"])})
        dataset = dataset.remove_columns(["text","label_text"])
        dataset.save_to_disk(os.path.join(self.cache_dir, split))

    def prepare_data(self, prompt_template, answers_templates, tokenizer_dir, max_seq_len):
        datadict = load_dataset("SetFit/20_newsgroups")
        prompt = Prompt(prompt_template, answers_templates, tokenizer_dir, max_seq_len)
        self._prepare_data_per_split(datadict, "train", prompt)
        self._prepare_data_per_split(datadict, "val", prompt)
        self._prepare_data_per_split(datadict, "test", prompt)

    def create_dataloader(self, split, batch_size, num_samples=None, shuffle=True, random_state=None):
        dataset = load_from_disk(os.path.join(self.cache_dir, split))
        if shuffle:
            rs = np.random.RandomState(random_state)
            if num_samples is None:
                num_samples = len(dataset)
            idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
            dataset = dataset.select(idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader
        



        



