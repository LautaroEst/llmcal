
from functools import partial
import os
from pathlib import Path
from lightning.pytorch.core.hooks import DataHooks
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from .sst2 import load_sst2
from typing import Literal, List, Callable
from ...prompt import PrefixPrompt
from ...model.modules.lit_tokenizer import LitGPTTokenizer

SUPPORTED_DATASETS = Literal["sst2", "20newsgroup", "medical_abstracts", "dbpedia", "banking77"]


def _prepare_data_for_lm_litgpt(
    preshots_template: str, 
    shots_template: str,
    postshots_template: str,
    shots_separator: str,
    answers_templates: List[str],
    data_load_fn: Callable,
    data_cache_dir: str,
    tokenizer_dir: str
) -> None:
    if os.path.exists(data_cache_dir):
        return
    
    # Create the cache directory
    os.makedirs(data_cache_dir, exist_ok=True)

    # Download the dataset
    datadict, shots = data_load_fn()

    # Fill the prompt
    prompt = PrefixPrompt(preshots_template, shots_template, postshots_template, shots_separator, answers_templates)
    prompt.fit(shots)

    # Tokenizer
    if not os.path.exists(tokenizer_dir):
        if not os.path.exists(os.path.join(os.getenv("LIT_CHECKPOINTS"), tokenizer_dir)):
            raise FileNotFoundError(f"Checkpoint directory {tokenizer_dir} not found")
        tokenizer = LitGPTTokenizer(Path(os.getenv("LIT_CHECKPOINTS")) / tokenizer_dir)
    else:
        tokenizer = LitGPTTokenizer(Path(tokenizer_dir))

    def transform(sample):
        prompt = prompt.transform(**sample)
        prompt_ids = tokenizer([prompt["prompt"]])["input_ids"][0,:]
        answers_ids = [tokenizer([ans])["input_ids"][0,1:] for ans in prompt["answers"]]
        return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}

    # Process the dataset
    datadict["train"] = datadict["train"].map(transform)
    datadict["validation"] = datadict["validation"].map(transform)
    datadict["test"] = datadict["test"].map(transform)

    # Save to disk
    datadict.save_to_disk(data_cache_dir)
    shots.save_to_disk(os.path.join(data_cache_dir,"shots"))


class DataPreparation(DataHooks):

    def __init__(
        self, 
        dataset_name: SUPPORTED_DATASETS,
        data_dir: str, 
        num_train_samples, 
        num_val_samples, 
        num_shots, 
        random_state,
        task,
        model_type,
        method,
        **kwargs
    ):
        super().__init__()
        self.prepare_data_per_node = False

        if dataset_name == "sst2":
            self.data_load_fn = partial(
                load_sst2, 
                data_dir=data_dir, 
                num_train_samples=num_train_samples,
                num_val_samples=num_val_samples,
                num_shots=num_shots,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Invalid dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_shots = num_shots
        self.random_state = random_state

        self.task = task
        self.model_type = model_type
        self.method = method

        self.kwargs = kwargs

    def prepare_data(self) -> None:
        if (
            self.task == "language_model" 
            and self.model_type == "litgpt" 
            and self.method in ["no_adaptation", "full_ft", "lora"]
        ):
            _prepare_data_for_lm_litgpt(
                preshots_template=self.kwargs["preshots_template"],
                shots_template=self.kwargs["shots_template"],
                postshots_template=self.kwargs["postshots_template"],
                shots_separator=self.kwargs["shots_separator"],
                answers_templates=self.kwargs["answers_templates"],
                data_load_fn=self.data_load_fn,
                data_cache_dir=self.data_dir,
                tokenizer_dir=self.kwargs["checkpoint_dir"],
            )
        else:
            raise ValueError(f"Invalid combination of task ({self.task}), model_type ({self.model_type}) and method ({self.method})")