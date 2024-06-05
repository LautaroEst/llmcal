

import os
from pathlib import Path
import shutil
import sys
from typing import List, Union

import numpy as np
import torch
from llmcal.data.datasets.utils import SUPPORTED_DATASETS, load_dataset_for_xval
from llmcal.data.utils import LitGPTCollator
from llmcal.prompt.litgpt import LitGPTPrompt
from llmcal.loggers import TBLogger, CSVLogger, setup_loggers
from llmcal.models.litgpt_lora import LanguageModelLitGPTLoRA, LitGPTLoRA, init_lora_linear_modules
from llmcal.utils import save_yaml

from litgpt import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from litgpt.lora import Config, mark_only_lora_as_trainable, lora_filter, merge_lora_weights
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import TrainerStatus
from datasets import load_from_disk, concatenate_datasets

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
    total_train_samples: int,
    nfolds: int,
    num_shots: int,
    val_prop: float,
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
    max_epochs: int = 1000,
    max_steps: int = -1,
    accumulate_grad_batches: int = 1,
    use_lora_checkpoint: bool = False,
    lora_r: int = 1,
    lora_alpha: float = 0.5,
    lora_dropout: float = 0.1,
    lora_query: bool = True,
    lora_key: bool = True,
    lora_value: bool = True,
    lora_projection: bool = True,
    lora_mlp: bool = True,
    lora_head: bool = True,
    optimizer: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
):
    torch.set_float32_matmul_precision("high")
    os.makedirs(output_dir, exist_ok=True)
    setup_loggers(output_dir)
    save_yaml(locals(), os.path.join(output_dir, "params.yaml"))

    # Load dataset
    folds = load_dataset_for_xval(dataset, total_train_samples, nfolds, random_state)

    # Load and train prompt
    prompt = LitGPTPrompt(preshots_template, shots_template, postshots_template, answers_templates)
    prompt.fit()
    
    # Load tokenizer and config
    checkpoint_dir = parse_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_checkpoint(
        checkpoint_dir,
        lora_r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        lora_query = lora_query,
        lora_key = lora_key,
        lora_value = lora_value,
        lora_projection = lora_projection,
        lora_mlp = lora_mlp,
        lora_head = lora_head,
    )

    def transform_train(sample):
        """Transform a sample from the dataset to a format that can be processed by the model."""
        filled_prompt = prompt.transform(**sample)
        prompt_with_answer = f"{filled_prompt['prompt']} {filled_prompt['answers'][sample['label']]}"
        prompt_ids = tokenizer.encode(prompt_with_answer, bos=True)
        answers_ids = [torch.tensor([],dtype=torch.int) for _ in filled_prompt["answers"]]
        return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}
    
    def transform_predict(sample):
        """Transform a sample from the dataset to a format that can be processed by the model."""
        filled_prompt = prompt.transform(**sample)
        prompt_ids = tokenizer.encode(filled_prompt["prompt"], bos=True)
        answers_ids = [tokenizer.encode(ans, bos=True)[1:] for ans in filled_prompt["answers"]]
        return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}
    
    for i, fold in enumerate(folds):
        train_data, prediction_data = fold["train"], fold["validation"]
        train_data = train_data.map(transform_train).with_format("torch")
        prediction_data = prediction_data.map(transform_predict).with_format("torch")
        fold_output_dir = os.path.join(output_dir, f"fold_{i}")
        train_and_predict(train_data, prediction_data, config, checkpoint_dir, fold_output_dir, batch_size, accumulate_grad_batches, max_epochs, max_steps, accelerator, strategy, devices, num_nodes, precision, optimizer, learning_rate, weight_decay, use_lora_checkpoint)

    all_predictions = []
    for i in range(len(folds)):
        dataset = load_from_disk(os.path.join(output_dir, f"fold_{i}", "predictions"))
        all_predictions.append(dataset)

    all_predictions = concatenate_datasets(all_predictions)
    train_samples = int(total_train_samples * (1 - val_prop))
    val_samples = total_train_samples - train_samples
    rs = np.random.RandomState(random_state)
    idx = rs.permutation(len(all_predictions))
    train_idx = idx[:train_samples]
    val_idx = idx[train_samples:train_samples + val_samples]
    datadict = {
        "train": all_predictions.select(train_idx),
        "validation": all_predictions.select(val_idx),
    }

    for split, dataset in datadict.items():
        os.makedirs(os.path.join(output_dir, "predictions", split), exist_ok=True)
        dataset.save_to_disk(os.path.join(output_dir, "predictions", split))

def train_and_predict(
    train_data, 
    prediction_data, 
    config, 
    checkpoint_dir, 
    output_dir, 
    batch_size, 
    accumulate_grad_batches, 
    max_epochs, 
    max_steps, 
    accelerator, 
    strategy, 
    devices, 
    num_nodes, 
    precision, 
    optimizer, 
    learning_rate, 
    weight_decay, 
    use_lora_checkpoint
):
    
    if os.path.exists(os.path.join(output_dir, f"predictions/state.json")):
        print(f"Fold already trained and predicted. Skipping.")
        return

    # Process the train dataset
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=LitGPTCollator(0, config.block_size)
    )

    # Init trainer
    trainer = L.Trainer(
        accelerator = accelerator,
        strategy = strategy,
        devices = devices,
        num_nodes = num_nodes,
        precision = precision,
        logger = [
            TBLogger(save_dir=os.path.join(output_dir,"logs")),
            CSVLogger(save_dir=os.path.join(output_dir,"logs"))
        ],
        max_epochs = max_epochs,
        max_steps = max_steps,
        limit_val_batches = 0.0,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        accumulate_grad_batches = accumulate_grad_batches,
        deterministic = True,
        profiler = "simple",
        default_root_dir = output_dir,
    )

    # Init base model
    with trainer.init_module(empty_init=True):
        gpt = LitGPTLoRA(config)
        # gpt.set_kv_cache(batch_size=1)
    mark_only_lora_as_trainable(gpt)
    
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint_path_lora = checkpoint_dir / "lit_model.pth.lora"  if use_lora_checkpoint else None
    checkpoint = lazy_load(checkpoint_path)
    gpt.load_state_dict(checkpoint, strict=False)
    if checkpoint_path_lora is not None:
        checkpoint = lazy_load(checkpoint_path_lora)
        gpt.load_state_dict(checkpoint, strict=False)
    else:
        init_lora_linear_modules(gpt)

    model = LanguageModelLitGPTLoRA(
        gpt = gpt,
        optimizer = optimizer,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
    )

    # -----------------
    # Fit the model
    # -----------------
    last_checkpoint_path = os.path.join(output_dir, "last.ckpt") if os.path.exists(os.path.join(output_dir, "last.ckpt")) else None
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=last_checkpoint_path)
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        sys.exit("Training interrupted.")

    # Save best checkpoint and config files
    os.makedirs(os.path.join(output_dir,"checkpoint"), exist_ok=True)
    torch.save({k: v for k, v in model.gpt.state_dict().items() if lora_filter(k,v)}, Path(output_dir) / "checkpoint" / "lit_model.pth.lora")
    if not os.path.exists(os.path.join(output_dir, "checkpoint/lit_model.pth")):
        os.symlink(checkpoint_path, os.path.join(output_dir,"checkpoint/lit_model.pth")) # symlink to the "lit_model.pth" file
    config_files = ["config.json", "generation_config.json", "model_config.yaml"]
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    for file_name in config_files + tokenizer_files:
        src_path = checkpoint_dir / file_name
        tgt_path = Path(output_dir) / "checkpoint" / file_name
        if src_path.exists() and not tgt_path.exists():
            shutil.copy(src_path, os.path.join(output_dir,"checkpoint"))

    # Merge LoRA weights for inference
    merge_lora_weights(model.gpt)
    with trainer.init_module():
        model.gpt.set_kv_cache(batch_size=batch_size)

    # -----------------
    # Evaluate the model
    # -----------------
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
        
    dataloader = DataLoader(
        prediction_data, 
        shuffle=False, 
        batch_size=batch_size, 
        num_workers=4, 
        collate_fn=LitGPTCollator(0, config.block_size)
    )
    trainer.predict(model, dataloaders=dataloader)
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        sys.exit("Prediction interrupted.")
    model.predict_outputs.save_to_disk(os.path.join(output_dir, f"predictions"))


    
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
