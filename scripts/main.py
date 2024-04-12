import os
import pickle
from typing import Any, List, Literal, Optional, Union
import torch
import lightning as L

from llmcal.utils import load_yaml, save_yaml
from llmcal.models.utils import load_model_from_checkpoint
from llmcal.data.datasets import TwentyNewsGroupDataset, TensorDataset
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
    base_model: str,
    dataset: Literal["20newsgroup", "medical_abstracts", "dbpedia", "banking77"],
    prompt: str,
    method: str,
    fold: str,
):
    
    # Parse config arguments
    base_model_config = load_yaml(f"configs/base_model/{base_model}.yaml")
    prompt_config = load_yaml(f"configs/prompt/{prompt}.yaml")
    method_config = load_yaml(f"configs/method/{method}.yaml")
    fold_config = load_yaml(f"configs/fold/{fold}.yaml")
    data_cache_dir = f"experiments/{base_model}/{dataset}/{prompt}/.cache"
    results_dir = f"experiments/{base_model}/{dataset}/{prompt}/{method}/{fold}/" 
    
    # TODO: add modifications on dict and dict name

    if os.path.exists(os.path.join(results_dir, "config.yaml")):
        print(f"Experiment {base_model}/{dataset}/{prompt}/{method}/{fold} already exists.")
        return

    # Init fabric
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
    fabric.launch(main, base_model_config, dataset, prompt_config, method_config, fold_config, data_cache_dir, results_dir)
    save_yaml({**base_model_config, **prompt_config, **method_config, **fold_config, "dataset": dataset}, os.path.join(results_dir, "config.yaml"))
    print("Done!")


def main(
    fabric: L.Fabric,
    base_model_kwargs: dict,
    dataset: str,
    prompt_kwargs: dict,
    method_kwargs: dict,
    fold_kwargs: dict,
    data_cache_dir: str,
    results_dir: str,
):
    
    checkpoint_dir = base_model_kwargs["checkpoint_dir"]
    model_type = base_model_kwargs["model_type"]
    method = method_kwargs.pop("method", "no_adaptation")
    prompt_template = prompt_kwargs["prompt_template"]
    answers_templates = prompt_kwargs["answers_templates"]

    # Set the seed
    fabric.seed_everything(GLOBAL_SEED)

    # Try to load checkpoint
    fabric.print(f"Loading checkpoint from {results_dir}...")
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
        fabric.load(os.path.join(results_dir, "checkpoint.ckpt"), state=state, strict=False)
    else:
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
        if dataset == "20newsgroup":
            dataset = TwentyNewsGroupDataset(cache_dir=data_cache_dir)
        else:
            raise NotImplementedError(f"Dataset {dataset} is not supported.")
        if fabric.global_rank == 0:
            dataset.prepare_data(prompt_template=prompt_template, answers_template=answers_templates, tokenizer_dir=checkpoint_dir, max_seq_len=state["model"].base_model.max_seq_len)
        fabric.barrier()        
    train_dataloader = dataset.create_dataloader("train", batch_size=method_kwargs["micro_batch_size"], num_samples=fold_kwargs["train_samples"], shuffle=True, random_state=fold_kwargs["random_state"])
    val_dataloader = dataset.create_dataloader("val", batch_size=method_kwargs["micro_batch_size"], num_samples=fold_kwargs["val_samples"], shuffle=False, random_state=fold_kwargs["random_state"])
    test_dataloader = dataset.create_dataloader("test", batch_size=method_kwargs["micro_batch_size"], num_samples=fold_kwargs["test_samples"], shuffle=False, random_state=fold_kwargs["random_state"])
    train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader, test_dataloader)
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
    if state["training_has_finished"] and state["train_batch_idx"] < len(train_dataloader):
        fabric.print("Predicting on train set...")
        outputs, batch_idx = predict_from_batch_idx(fabric, state["model"], train_dataloader, batch_idx=state["train_batch_idx"])
        state["train_batch_idx"] = batch_idx
        if fabric.global_rank == 0:
            with open(os.path.join(results_dir, "train_outputs.pkl"), "wb") as f:
                pickle.dump(outputs, f)

    if state["training_has_finished"] and state["val_batch_idx"] < len(val_dataloader):
        fabric.print("Predicting on val set...")
        outputs, batch_idx = predict_from_batch_idx(fabric, state["model"], val_dataloader, batch_idx=state["val_batch_idx"])
        state["val_batch_idx"] = batch_idx
        if fabric.global_rank == 0:
            with open(os.path.join(results_dir, "val_outputs.pkl"), "wb") as f:
                pickle.dump(outputs, f)

    if state["training_has_finished"] and state["test_batch_idx"] < len(test_dataloader):
        fabric.print("Predicting on test set...")
        outputs, batch_idx = predict_from_batch_idx(fabric, state["model"], test_dataloader, batch_idx=state["test_batch_idx"])
        state["test_batch_idx"] = batch_idx
        if fabric.global_rank == 0:
            with open(os.path.join(results_dir, "test_outputs.pkl"), "wb") as f:
                pickle.dump(outputs, f)
    
    if fabric.global_rank == 0:
        fabric.print("Saving checkpoint...")
        fabric.save(state, os.path.join(results_dir, "checkpoint.ckpt"))



if __name__ == "__main__":
    from fire import Fire
    torch.set_float32_matmul_precision("high")
    Fire(setup)

