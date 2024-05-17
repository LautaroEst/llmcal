

import os
from pathlib import Path
from typing import List

import torch
from llmcal.data.datasets.utils import SUPPORTED_DATASETS, load_dataset
from llmcal.data.utils import LitGPTCollator
from llmcal.prompt.litgpt import LitGPTPrompt
from litgpt import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from litgpt.lora import Config

from torch.utils.data import DataLoader
import lightning as L

def parse_checkpoint_dir(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        if not os.path.exists(os.path.join(os.getenv("LIT_CHECKPOINTS"), checkpoint_dir)):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
        checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
    else:
        checkpoint_dir = Path(checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    return checkpoint_dir



def main(
    dataset: SUPPORTED_DATASETS,
    train_samples: int,
    val_samples: int,
    num_shots: int,
    random_state: int,
    preshots_template: str, 
    shots_template: str,
    postshots_template: str,
    shots_separator: str,
    answers_templates: List[str],
    checkpoint_dir: str,
    batch_size: int = 32,
):

    # Load dataset
    train_datadict, _, shots = load_dataset(dataset, train_samples, val_samples, num_shots, random_state)

    # Load and train prompt
    prompt = LitGPTPrompt(preshots_template, shots_template, postshots_template, shots_separator, answers_templates)
    prompt.fit(shots)
    
    # Load tokenizer and config
    checkpoint_dir = parse_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_checkpoint(checkpoint_dir)

    def transform_train(sample):
        """Transform a sample from the dataset to a format that can be processed by the model."""
        filled_prompt = prompt.transform(**sample)
        prompt_with_answer = f"{filled_prompt['prompt']} {filled_prompt['answers'][sample['label']]}"
        prompt_ids = tokenizer.encode(prompt_with_answer, bos=True)
        answers_ids = [torch.tensor([],dtype=torch.int) for _ in filled_prompt["answers"]]
        return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}
    
    def transform_val(sample):
        """Transform a sample from the dataset to a format that can be processed by the model."""
        filled_prompt = prompt.transform(**sample)
        prompt_ids = tokenizer.encode(filled_prompt["prompt"], bos=True)
        answers_ids = [tokenizer.encode(ans, bos=True)[1:] for ans in filled_prompt["answers"]]
        return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}
    
    # Process the train dataset
    train_loader = DataLoader(
        train_datadict["train"].map(transform_train).with_format("torch"), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=LitGPTCollator(0, config.block_size)
    )
    val_loader = DataLoader(
        train_datadict["validation"].map(transform_val).with_format("torch"), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=LitGPTCollator(0, config.block_size)
    )


    
if __name__ == "__main__":
    main(
        dataset="dbpedia",
        train_samples=100,
        val_samples=100,
        num_shots=2,
        random_state=42,
        preshots_template="",
        shots_template="Article: \"{content}\"\nCategory: {answer}",
        shots_separator="\n",
        postshots_template="\nArticle: \"{content}\"\nCategory:",
        answers_templates=[
            "Company", "Educational Institution", "Artist", "Athlete", 
            "Office Holder", "Mean Of Transportation", "Building", 
            "Natural Place", "Village", "Animal", "Plant", "Album", 
            "Film", "Written Work"
        ],
        checkpoint_dir="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        batch_size=32,
    )
