

import os
from pathlib import Path
import shutil
import sys
from typing import List, Union

import torch
from llmcal.data.datasets.utils import SUPPORTED_DATASETS, load_dataset
from llmcal.loggers import TBLogger, CSVLogger, setup_loggers
from llmcal.models.encoder_lm import EncoderLanguageModel
from llmcal.utils import save_yaml

from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import TrainerStatus
from time import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(
    output_dir: str,
    dataset: SUPPORTED_DATASETS,
    total_train_samples: int,
    val_prop: float,
    sentence_field: str,
    num_shots: int,
    num_classes: int,
    random_state: int,
    checkpoint_dir: str,
    batch_size: int = 32,
    accelerator: str = "cpu",
    strategy: str = "auto",
    devices: int = 1,
    num_nodes: int = 1,
    precision: Union[int,str] = 32,
    max_epochs: int = 1000,
    max_steps: int = -1,
    val_check_interval: int = 1,
    accumulate_grad_batches: int = 1,
    optimizer: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    timing: bool = False,
):
    torch.set_float32_matmul_precision("high")
    os.makedirs(output_dir, exist_ok=True)
    setup_loggers(output_dir)
    save_yaml(locals(), os.path.join(output_dir, "params.yaml"))

    # Load dataset
    train_datadict, prediction_datadict, shots = load_dataset(dataset, total_train_samples, val_prop, num_shots, random_state, timing=timing)

    # Load tokenizer and config
    checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    config.num_labels = num_classes

    # Process the train dataset
    train_loader = DataLoader(
        train_datadict["train"].map(tokenizer, input_columns=sentence_field, remove_columns=sentence_field).select_columns(["idx","input_ids","attention_mask","label"]).with_format("torch"), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=DataCollatorWithPadding(tokenizer, padding=True)
    )
    val_loader = DataLoader(
        train_datadict["validation"].map(tokenizer, input_columns=sentence_field, remove_columns=sentence_field).select_columns(["idx","input_ids","attention_mask","label"]).with_format("torch"), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=DataCollatorWithPadding(tokenizer, padding=True)
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
        val_check_interval = val_check_interval,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        accumulate_grad_batches = accumulate_grad_batches,
        deterministic = True,
        profiler = "simple",
        default_root_dir = output_dir,
    )

    if timing:
        start_time = time()

    # Init base model
    with trainer.init_module(empty_init=True):
        base_model = AutoModelForSequenceClassification.from_config(config)
    for param in base_model.parameters():
        param.requires_grad = True

    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    checkpoint = torch.load(checkpoint_path, map_location=trainer.strategy.root_device)
    base_model.load_state_dict(checkpoint, strict=False)
    model = EncoderLanguageModel(
        base_model = base_model,
        optimizer = optimizer,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
    )
    # -----------------
    # Fit the model
    # -----------------
    last_checkpoint_path = os.path.join(output_dir, "last.ckpt") if os.path.exists(os.path.join(output_dir, "last.ckpt")) else None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=last_checkpoint_path)
    if timing:
        end_time = time()
        with open(os.path.join(output_dir, "timing_full_ft.txt"), "w") as f:
            f.write(f"{end_time - start_time}")
        sys.exit("Finish timing")
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        sys.exit("Training interrupted.")
    trainer.validate(model, dataloaders=val_loader)
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        sys.exit("Training interrupted.")

    # Save best checkpoint and config files
    checkpoint = lazy_load(os.path.join(output_dir, "best.ckpt"))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    os.makedirs(os.path.join(output_dir,"checkpoint"), exist_ok=True)
    torch.save(model.base_model.state_dict(), Path(output_dir) / "checkpoint" / "pytorch_model.bin")
    config_files = ["config.json"]
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
    for file_name in config_files + tokenizer_files:
        src_path = checkpoint_dir / file_name
        tgt_path = Path(output_dir) / "checkpoint" / file_name
        if src_path.exists() and not tgt_path.exists():
            shutil.copy(src_path, os.path.join(output_dir,"checkpoint"))

    # -----------------
    # Evaluate the model
    # -----------------
    predictions_dir = os.path.join(output_dir, "predictions")
    best_ckpt_path = os.path.join(output_dir, "best.ckpt")
    if os.path.exists(predictions_dir) and os.path.getmtime(best_ckpt_path) > os.path.getmtime(predictions_dir):
        version = 1
        while os.path.exists(predictions_dir):
            predictions_dir = predictions_dir.split(".v")[0] + f".v{version}"
            version += 1
        os.rename(os.path.join(output_dir, "predictions"), predictions_dir)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        
    for split, dataset in prediction_datadict.items():
        if os.path.exists(os.path.join(output_dir, f"predictions/{split}")):
            continue
        dataloader = DataLoader(
            dataset.map(tokenizer, input_columns=sentence_field, remove_columns=sentence_field).select_columns(["idx","input_ids","attention_mask","label"]).with_format("torch"),
            shuffle=False, 
            batch_size=batch_size, 
            num_workers=4, 
            collate_fn=DataCollatorWithPadding(tokenizer, padding=True)
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
