

import os
from pathlib import Path
import sys
from typing import List, Union

import torch
from llmcal.data.datasets.utils import SUPPORTED_DATASETS, load_dataset
from llmcal.data.utils import LitGPTCollator
from llmcal.prompt.litgpt import LitGPTPrompt
from llmcal.loggers import setup_loggers
from llmcal.models.litgpt_full_ft import LanguageModelLitGPTFullFT, LitGPTFullFT
from llmcal.utils import save_yaml

from litgpt import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from litgpt import Config
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import TrainerStatus

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
    output_dir: str,
    dataset: SUPPORTED_DATASETS,
    train_samples: int,
    val_samples: int,
    num_shots: int,
    random_state: int,
    preshots_template: str, 
    shots_template: str,
    postshots_template: str,
    answers_templates: List[str],
    checkpoint_dir: str,
    batch_size: int = 32,
    accelerator: str = "cpu",
    strategy: str = "auto",
    devices: int = 1,
    num_nodes: int = 1,
    precision: Union[int,str] = 32,
):
    torch.set_float32_matmul_precision("high")
    os.makedirs(output_dir, exist_ok=True)
    setup_loggers(output_dir)
    save_yaml(locals(), os.path.join(output_dir, "params.yaml"))

    # Load dataset
    _, prediction_datadict, shots = load_dataset(dataset, train_samples, val_samples, num_shots, random_state)

    # Load and train prompt
    prompt = LitGPTPrompt(preshots_template, shots_template, postshots_template, answers_templates)
    prompt.fit(shots)
    
    # Load tokenizer and config
    checkpoint_dir = parse_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_checkpoint(checkpoint_dir)

    # Init trainer
    trainer = L.Trainer(
        accelerator = accelerator,
        strategy = strategy,
        devices = devices,
        num_nodes = num_nodes,
        precision = precision,
        logger = False,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        deterministic = True,
        profiler = "simple",
        default_root_dir = output_dir,
    )

    # Init base model
    with trainer.init_module(empty_init=True):
        gpt = LitGPTFullFT(config)
        gpt.set_kv_cache(batch_size=batch_size)
    
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint = lazy_load(checkpoint_path)
    gpt.load_state_dict(checkpoint, strict=False)
    model = LanguageModelLitGPTFullFT(gpt)

    # -----------------
    # Evaluate the model
    # -----------------
    def transform(sample):
        """Transform a sample from the dataset to a format that can be processed by the model."""
        filled_prompt = prompt.transform(**sample)
        prompt_ids = tokenizer.encode(filled_prompt["prompt"], bos=True)
        answers_ids = [tokenizer.encode(ans, bos=True)[1:] for ans in filled_prompt["answers"]]
        return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}
    
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    for split, dataset in prediction_datadict.items():
        if os.path.exists(os.path.join(output_dir, f"predictions/{split}")):
            continue
        dataloader = DataLoader(
            dataset.map(transform).with_format("torch"), 
            shuffle=False, 
            batch_size=batch_size, 
            num_workers=4, 
            collate_fn=LitGPTCollator(0, config.block_size)
        )
        trainer.predict(model, dataloaders=dataloader)
        if trainer.state.status == TrainerStatus.INTERRUPTED:
            sys.exit("Prediction interrupted.")
        model.predict_outputs.save_to_disk(os.path.join(output_dir, f"predictions/{split}"))


    
if __name__ == "__main__":
    main(
        output_dir="./test",
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
        batch_size=1,
        accumulate_grad_batches=32,
        max_epochs=1,
        accelerator="gpu",
        val_check_interval=64,
    )
