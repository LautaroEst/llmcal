
import os
from pathlib import Path
from typing import Literal
import lightning as L
from llmcal.utils import load_yaml
from llmcal.model.utils import check_model_type
from llmcal.model.modules import *
from llmcal.data.datasets import *
from time import perf_counter
from functools import partial
from llmcal.model import Trainer
from llmcal.model.loggers import TBLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from llmcal.data.datasets.base import SUPPORTED_DATASETS


def main(
    dataset: SUPPORTED_DATASETS,
    prompt: str,
    data_fold: str,
    model: str,
    method: str,
):
    model_config = load_yaml(f"configs/model/{model}.yaml")
    prompt_config = load_yaml(f"configs/prompt/{prompt}.yaml")
    data_fold_config = load_yaml(f"configs/fold/{data_fold}.yaml")
    method_config = load_yaml(f"configs/method/{method}.yaml")
    results_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/{method}"

    # ---------------------
    # Dataset
    # ---------------------
    data_dir = f"experiments/{dataset}/.cache"
    if dataset == "sst2":
        data_load_fn = partial(
            load_sst2, 
            data_dir=data_dir, 
            num_train_samples=data_fold_config["train_samples"], 
            num_val_samples=data_fold_config["val_samples"], 
            num_shots=data_fold_config["shots_samples"], 
            random_state=data_fold_config["random_state"]
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    # ---------------------
    # Trainer
    # ---------------------
    trainer = L.Trainer(
        accelerator = model_config.get("accelerator", "auto"),
        strategy = model_config.get("strategy", "auto"),
        devices = model_config.get("devices", "auto"),
        num_nodes = model_config.get("num_nodes", 1),
        precision = model_config.get("precision", "32-true"),
        logger = TBLogger(save_dir=results_dir),
        callbacks = [
            ModelCheckpoint(
                dirpath=results_dir,
                filename="checkpoint",
                monitor="val_loss",
                save_last=False,
                save_top_k=1,
                save_weights_only=False,
                mode="min",
                every_n_train_steps=method_config.get("checkpoint_frequency", 1),
            )
        ],
        fast_dev_run = method_config.get("fast_dev_run", False),
        max_epochs = method_config.get("max_epochs", 1000),
        min_epochs = method_config.get("min_epochs", 1),
        max_steps = method_config.get("max_steps", -1),
        min_steps = method_config.get("min_steps", 1),
        max_time = method_config.get("max_time", None),
        limit_train_batches = method_config.get("limit_train_batches", None),
        limit_val_batches = method_config.get("limit_val_batches", None),
        limit_test_batches = method_config.get("limit_test_batches", None),
        limit_predict_batches = method_config.get("limit_predict_batches", None),
        overfit_batches = method_config.get("overfit_batches", 0),
        val_check_interval = method_config.get("val_check_interval", 1),
        check_val_every_n_epoch = method_config.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps = method_config.get("num_sanity_val_steps", None),
        log_every_n_steps = None,
        enable_checkpointing = True,
        enable_progress_bar = True,
        enable_model_summary = True,
        accumulate_grad_batches = method_config.get("accumulate_grad_batches", 1),
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
        deterministic = True,
        benchmark = None,
        use_distributed_sampler = True,
        profiler = None,
        detect_anomaly = False,
        barebones = False,
        plugins = None,
        sync_batchnorm = False,
        reload_dataloaders_every_n_epochs = 0,
        default_root_dir = results_dir,
    )

    # ---------------------
    # Model
    # ---------------------
    task = model_config.pop("task", None)
    method = method_config.pop("method", None)
    model_type = check_model_type(model_config["checkpoint_dir"], method)
    if task == "language_model" and model_type == "litgpt" and method == "full_ft":
        model_cls = LanguageModelLitGPTFullFT
        model_init_args = {
            "checkpoint_dir": model_config["checkpoint_dir"],
            "embedding_pooling": model_config["embedding_pooling"],
            "preshots_template": prompt_config["preshots_template"],
            "shots_template": prompt_config["shots_template"],
            "postshots_template": prompt_config["postshots_template"],
            "shots_separator": prompt_config["shots_separator"],
            "answers_templates": prompt_config["answers_templates"],
            "data_load_fn": data_load_fn,
            "data_cache_dir": f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache",
            "batch_size": method_config["batch_size"],
            "loss_fn": method_config["loss_fn"],
            "optimizer": method_config["optimizer"],
            "learning_rate": method_config["learning_rate"],
            "weight_decay": method_config["weight_decay"],
        }
    elif task == "language_model" and model_type == "litgpt" and method == "no_adaptation":
        model_cls = LanguageModelLitGPTNoAdaptation
        model_init_args = {
            "checkpoint_dir": model_config["checkpoint_dir"],
            "preshots_template": prompt_config["preshots_template"],
            "shots_template": prompt_config["shots_template"],
            "postshots_template": prompt_config["postshots_template"],
            "shots_separator": prompt_config["shots_separator"],
            "answers_templates": prompt_config["answers_templates"],
            "embedding_pooling": model_config["embedding_pooling"],
            "batch_size": method_config["batch_size"],
            "data_load_fn": data_load_fn,
            "data_cache_dir": f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache",
        }
    elif task == "language_model" and model_type == "litgpt" and method == "lora":
        model_cls = LanguageModelLitGPTLoRA
        model_init_args = {
            "checkpoint_dir": model_config["checkpoint_dir"],
            "preshots_template": prompt_config["preshots_template"],
            "shots_template": prompt_config["shots_template"],
            "postshots_template": prompt_config["postshots_template"],
            "shots_separator": prompt_config["shots_separator"],
            "answers_templates": prompt_config["answers_templates"],
            "embedding_pooling": model_config["embedding_pooling"],
            "batch_size": method_config["batch_size"],
            "data_load_fn": data_load_fn,
            "data_cache_dir": f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache",
        }
    elif task == "language_model" and model_type == "litgpt" and method == "affine_calibration":
        model_cls = LanguageModelLitGPTAffineCalibration
        model_init_args = {
            "data_cache_dir": f"experiments/{dataset}/{data_fold}/{prompt}/{model}/no_adaptation/",
            "alpha": method_config["alpha"],
            "beta": method_config["beta"],
            "batch_size": method_config["batch_size"],
            "max_ls": method_config["max_ls"],
        }
    else:
        raise ValueError(f"Invalid combination of task ({task}), checkpoint_dir ({model_config['checkpoint_dir']}) and method ({method})")
    with trainer.init_module():
        model = model_cls(**model_init_args)

    # ---------------------
    # Fit the model
    # ---------------------
    if os.path.exists(os.path.join(results_dir, "checkpoint.ckpt")):
        trainer.fit(model, ckpt_path=os.path.join(results_dir, "checkpoint.ckpt"))
    else:
        trainer.fit(model)

    # ---------------------
    # Predict
    # ---------------------
    trainer.predict(model)


if __name__ == "__main__":
    from fire import Fire
    import torch
    torch.set_float32_matmul_precision("high")
    Fire(main)