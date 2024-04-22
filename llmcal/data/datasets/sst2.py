import numpy as np
from datasets import load_dataset, load_from_disk
from ..prompt import Prompt, DynamicPaddingCollator
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from transformers import AutoTokenizer


def _sample_and_shuffle(dataset, num_samples, random_state):
    rs = np.random.RandomState(random_state)
    if num_samples is None:
        num_samples = len(dataset)
    idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
    dataset = dataset.select(idx)
    return dataset


class SST2Dataset(LightningDataModule):

    def __init__(self, prompt_template, model_name_or_path, num_train_samples, num_val_samples, batch_size, cache_dir, random_state = 0):
        self.prompt = Prompt(prompt_template)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.random_state = random_state

    def prepare_data(self):
        datadict = load_dataset("nyu-mll/glue", "sst2")
        datadict["train"] = _sample_and_shuffle(datadict["train"], self.num_train_samples, self.random_state)
        datadict["validation"] = _sample_and_shuffle(datadict["validation"], self.num_val_samples, self.random_state)
        self.prompt.fit(datadict["train"])
        for split in ["train", "validation", "test"]:
            datadict[split] = datadict[split].map(lambda sample: {"input": self.tokenizer(self.prompt.transform(**{"sentence": sample["sentence"]}))})
        datadict.save_to_disk(self.cache_dir)

    def setup(self, stage):
        datadict = load_from_disk(self.cache_dir)
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
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=DynamicPaddingCollator(self.tokenizer.pad_token_id))
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=DynamicPaddingCollator(self.tokenizer.pad_token_id))
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=DynamicPaddingCollator(self.tokenizer.pad_token_id))
    
    def predict_dataloader(self):
        return {
            split: DataLoader(self.predict_data[split], batch_size=self.batch_size, shuffle=False, collate_fn=DynamicPaddingCollator(self.tokenizer.pad_token_id)) \
            for split in ["val", "test"]
        }

        

    
        