import os
import pickle
from copy import copy
from typing import Any, List, Literal, Optional, Union
import torch
import lightning as L

from llmcal.utils import load_yaml, save_yaml
from llmcal.models.utils import load_model_from_checkpoint
from llmcal.data.datasets import *
from litgpt.utils import CycleIterator

GLOBAL_SEED = 8239

def train(
    fabric: L.Fabric,
    state: dict,
    train_iterator: CycleIterator,
    val_dataloader: torch.utils.data.DataLoader,
):
    try:
        # Training loop here
        state["training_has_finished"] = True
    except KeyboardInterrupt:
        state["training_has_finished"] = False

    return state
    

def predict_from_batch_idx(
    fabric: L.Fabric,
    model: L.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    batch_idx: int,
):
    batch_idx = len(dataloader)
    outputs = []
    return outputs, batch_idx


def setup(
    dataset: Literal["sst2", "20newsgroup", "medical_abstracts", "dbpedia", "banking77"],
    prompt: str,
    train_list: str,
    base_model: str,
    method: str,
):
    
    # Init experiment
    results_dir = f"experiments/{dataset}/{prompt}/{train_list}/{base_model}/{method}"
    if os.path.exists(os.path.join(results_dir, "config.yaml")):
        print(f"Experiment {results_dir} already run.")
        return
    print("Running experiment")
    print("Dataset:", dataset)
    print("Prompt:", prompt)
    print("Train list:", train_list)
    print("Base model:", base_model)
    print("Method:", method)
    os.makedirs(results_dir, exist_ok=True)
    save_yaml({
        "dataset_args": dataset_args,
        "prompt_args": prompt_args,
        "base_model_checkpoint": base_model_checkpoint,
        "method_args": method_args,
    }, os.path.join(results_dir, "config.yaml"))
    
    # Dataset:
    train_list_config = load_yaml(f"configs/fold/{train_list}.yaml")
    dataset_args = {
        "dataset": dataset,
        "nshots": train_list_config.get("nshots", 0),
        "train_samples": train_list_config.get("train_samples", 0),
        "val_samples": train_list_config.get("val_samples", 0),
        "random_state": train_list_config.get("random_state", 0),
        "cache_dir": f"experiments/{dataset}/{prompt}/{train_list}/.cache"
    }

    # Prompt:
    prompt_config = load_yaml(f"configs/prompt/{prompt}.yaml")
    prompt_args = {
        "preshot": prompt_config.get("preshot", ""),
        "shot": prompt_config.get("shot", ""),
        "postshot": prompt_config.get("postshot", ""),
        "answers": prompt_config.get("answers", ""),
    }

    # Base model:
    base_model_config = load_yaml(f"configs/base_model/{base_model}.yaml")
    fabric = L.Fabric(
        accelerator=base_model_config.get("accelerator", "cpu"),
        strategy=base_model_config.get("strategy", "auto"),
        devices=base_model_config.get("devices", 1),
        num_nodes=base_model_config.get("num_nodes", 1),
        precision=base_model_config.get("precision", 32),
        plugins=base_model_config.get("plugins", None),
        callbacks=base_model_config.get("callbacks", None),
        loggers=base_model_config.get("loggers", None),
    )
    base_model_checkpoint = base_model_config["checkpoint_dir"]

    # Method:
    method_config = load_yaml(f"configs/method/{method}.yaml")
    method_args = {
        "method": method,
        **method_config
    }
    
    # TODO: add modifications on dict and dict name

    fabric.launch(
        main,
        dataset_args=dataset_args,
        prompt_args=prompt_args,
        base_model_checkpoint=base_model_checkpoint,
        method=method,
        method_args=method_args,
        results_dir=results_dir
    )
    
    

def main(
    fabric: L.Fabric,
    dataset_args: dict = dict(
        dataset="sst2",
        nshots=0,
        train_samples=0,
        val_samples=0,
        random_state=0,
        cache_dir=""
    ),
    prompt_args: dict = dict(
        preshot="",
        shot="",
        postshot="",
        answers="",
    ),
    base_model_checkpoint: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    method: Literal["no_adaptation", "affine_calibration", "full_ft", "lora"] = "no_adaptation",
    method_args: dict = {},
    results_dir: str = ""
):
    
    checkpoint_dir = base_model_kwargs["checkpoint_dir"]
    model_type = base_model_kwargs["model_type"]
    method = method_kwargs.pop("method", "no_adaptation")
    prompt_template = prompt_kwargs["prompt_template"]
    answers_templates = prompt_kwargs["answers_templates"]

    # Set the seed
    fabric.seed_everything(GLOBAL_SEED)

    # Try to load checkpoint
    state = {
        "model": None, 
        "optimizer": None, 
        "train_batch_idx": 0,
        "val_batch_idx": 0,
        "test_batch_idx": 0,
        "epoch": 0,
        "global_step": 0,
        "iter_num": 0,
        "training_has_finished": False,
    }
    if os.path.exists(os.path.join(results_dir, "checkpoint.ckpt")):
        fabric.print(f"Loading checkpoint from {results_dir}...")
        fabric.load(os.path.join(results_dir, "checkpoint.ckpt"), state=state, strict=False)
    else:
        fabric.print("Initializing model...")
        state["model"] = load_model_from_checkpoint(fabric, checkpoint_dir, model_type, method, **method_kwargs)
        state["optimizer"] = state["model"].configure_optimizers()
    if state["optimizer"] is None:
        state["model"] = fabric.setup(state["model"])
    else:
        state["model"], state["optimizer"] = fabric.setup(state["model"], state["optimizer"])

    # Prepare dataloaders
    fabric.print("Preparing dataloaders...")
    if method == "affine_calibration":
        logits_dir = os.path.join("/".join(results_dir.split("/")[:-2]), "no_adaptation/all")
        if not os.path.exists(logits_dir):
            raise ValueError("Affine calibration requires the no_adaptation method to be run first on the entire dataset.")
        dataset = TensorDataset(logits_dir, "logits")
    else:
        if dataset == "sst2":
            dataset = SST2Dataset(data_cache_dir, prompt_template, answers_templates, checkpoint_dir, state["model"].base_model.max_seq_length)
        elif dataset == "20newsgroup":
            dataset = TwentyNewsGroupDataset(cache_dir=data_cache_dir)
        else:
            raise NotImplementedError(f"Dataset {dataset} is not supported.")
        if fabric.global_rank == 0:
            dataset.prepare_data()
    fabric.barrier()
    train_dataloader = dataset.create_dataloader("train", batch_size=method_kwargs["micro_batch_size"], num_samples=train_fold_kwargs["train_samples"], shuffle=True, random_state=train_fold_kwargs["random_state"])
    val_dataloader = dataset.create_dataloader("val", batch_size=method_kwargs["micro_batch_size"], num_samples=train_fold_kwargs["val_samples"], shuffle=False, random_state=train_fold_kwargs["random_state"])
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    eval_dataloaders = {}
    for split in ["train", "val", "test"]:
        if eval_fold_kwargs[f"{split}_samples"] > 0:
            dl = dataset.create_dataloader(split, batch_size=method_kwargs["micro_batch_size"], num_samples=eval_fold_kwargs[f"{split}_samples"], shuffle=False, random_state=eval_fold_kwargs["random_state"])
            eval_dataloaders[split] = fabric.setup_dataloaders(dl)
        else:
            eval_dataloaders[split] = None
    train_iterator = CycleIterator(train_dataloader)
    for _ in range(state["iter_num"]):
        next(train_iterator)
    fabric.barrier()

    # Train
    if not state["training_has_finished"]:
        if method != "no_adaptation":
            state = train(fabric, state, train_iterator, val_dataloader)
        else:
            fabric.print("No adaptation method selected. Skipping training.")
            state["training_has_finished"] = True
    
    # Predict
    for split in ["train", "val", "test"]:
        if eval_fold_kwargs[f"{split}_samples"] > 0:
            if state["training_has_finished"] and state[f"{split}_batch_idx"] < len(eval_dataloaders[split]):
                fabric.print(f"Predicting on {split} set...")
                outputs, batch_idx = predict_from_batch_idx(fabric, state["model"], eval_dataloaders[split], batch_idx=state[f"{split}_batch_idx"])
                state[f"{split}_batch_idx"] = batch_idx
                if fabric.global_rank == 0:
                    with open(os.path.join(results_dir, f"{split}_outputs.pkl"), "wb") as f:
                        pickle.dump(outputs, f)
    fabric.barrier()

    if fabric.global_rank == 0:
        fabric.print("Saving checkpoint...")
        fabric.save(os.path.join(results_dir, "checkpoint.ckpt"), state)
    
    fabric.print("Done!")



if __name__ == "__main__":
    from fire import Fire
    torch.set_float32_matmul_precision("high")
    Fire(setup)

