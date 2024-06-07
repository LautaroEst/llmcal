

import os
import sys
from typing import Literal

import torch
from llmcal.data.datasets.utils import load_dataset
from llmcal.loggers import TBLogger, CSVLogger, setup_loggers
from llmcal.utils import save_yaml
from llmcal.models.affine_calibration_no_es import AffineCalibration

from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import TrainerStatus

AFFINE_METHODS_NO_ES = ["affine_matrix_no_es", "affine_vector_no_es", "affine_scalar_no_es", "temp_scaling_no_es", "bias_only_no_es"]


def main(
    output_dir: str,
    total_train_samples: int,
    val_prop: int,
    random_state: int,
    alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
    beta: bool = True,
    max_ls: int = 40,
    learning_rate: float = 0.001,
    accelerator: str = "cpu",
    max_epochs: int = 1000,
):
    torch.set_float32_matmul_precision("high")
    os.makedirs(output_dir, exist_ok=True)
    setup_loggers(output_dir)
    save_yaml(locals(), os.path.join(output_dir, "params.yaml"))

    # Load dataset
    data_dir = os.path.join("/".join(output_dir.split("/")[:-1]),".cache/predictions")
    train_datadict, prediction_datadict, _ = load_dataset(data_dir, total_train_samples, 0, 0, random_state)

    # Process the train dataset
    train_loader = DataLoader(
        train_datadict["train"].with_format("torch"), 
        batch_size=len(train_datadict["train"]), 
        shuffle=True, 
        num_workers=4, 
    )
    # val_loader = DataLoader(
    #     train_datadict["validation"].with_format("torch"), 
    #     batch_size=len(train_datadict["validation"]), 
    #     shuffle=False, 
    #     num_workers=4, 
    # )

    # Init trainer
    trainer = L.Trainer(
        accelerator = accelerator,
        strategy = "auto",
        devices = 1,
        num_nodes = 1,
        precision = 32,
        logger = [
            TBLogger(save_dir=os.path.join(output_dir,"logs")),
            CSVLogger(save_dir=os.path.join(output_dir,"logs")),
        ],
        max_epochs = max_epochs,
        check_val_every_n_epoch = 1,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        deterministic = True,
        profiler = "simple",
        default_root_dir = output_dir,
    )

    # Init base model
    model_params = dict(
        num_classes = len(train_datadict["train"][0]["logits"]),
        alpha = alpha,
        beta = beta,
        max_ls = max_ls,
        learning_rate = learning_rate,
    )
    with trainer.init_module():
        model = AffineCalibration(**model_params)
            
    # -------------------
    # Fit the model
    # -------------------
    last_checkpoint_path = os.path.join(output_dir, "last.ckpt") if os.path.exists(os.path.join(output_dir, "last.ckpt")) else None
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=last_checkpoint_path)
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=last_checkpoint_path)
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        sys.exit("Training interrupted.")

    if not os.path.exists(os.path.join(output_dir,"checkpoint")):
        os.makedirs(os.path.join(output_dir,"checkpoint"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir,"checkpoint","model.pth"))
    save_yaml(model_params,os.path.join(output_dir,"checkpoint","model_params.yaml"))

    # -------------------
    # Evaluate the model
    # -------------------
    predictions_dir = os.path.join(output_dir, "predictions")
    # best_ckpt_path = os.path.join(output_dir, "best.ckpt")
    best_ckpt_path = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(predictions_dir) and os.path.getmtime(best_ckpt_path) > os.path.getmtime(predictions_dir):
        version = 1
        while os.path.exists(predictions_dir):
            predictions_dir = predictions_dir.split(".v")[0] + f".v{version}"
            version += 1
        os.rename(os.path.join(output_dir, "predictions"), predictions_dir)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    checkpoint = lazy_load(best_ckpt_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for split, dataset in prediction_datadict.items():
        if os.path.exists(os.path.join(output_dir, f"predictions/{split}")):
            continue
        dataloader = DataLoader(
            dataset.with_format("torch"), 
            shuffle=False, 
            batch_size=len(dataset), 
            num_workers=4, 
        )
        trainer.predict(model, dataloaders=dataloader)
        if trainer.state.status == TrainerStatus.INTERRUPTED:
            sys.exit("Prediction interrupted.")
        model.predict_outputs.save_to_disk(os.path.join(output_dir, f"predictions/{split}"))


if __name__ == "__main__":
    main(
        output_dir="experiments/agnews_8_639/basic_agnews_0-shot_litgpt/lm_tinyllama/no_adaptation_bf16/test",
        total_train_samples=32,
        random_state=42,
        alpha="scalar",
        beta=True,
        max_ls=40,
        learning_rate=1.,
        max_epochs=100,
        accelerator="cpu",
    )
