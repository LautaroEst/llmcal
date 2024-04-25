
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

def main(
    dataset: Literal["sst2", "20newsgroup", "medical_abstracts", "dbpedia", "banking77"],
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

    if os.path.exists(os.path.join(results_dir, "done.txt")):
        print("Experiment already done. Exiting...")
        return
    t_start = perf_counter()

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
    # Model
    # ---------------------
    task = model_config.pop("task", None)
    model_type = check_model_type(model_config["checkpoint_dir"])
    method = method_config.pop("method", None)
    if task == "language_model" and model_type == "litgpt" and method == "no_adaptation":
        model_cls = LanguageModelLitGPTNoAdaptation
        model_init_args = {
            "checkpoint_dir": model_config["checkpoint_dir"],
            "preshots_template": prompt_config["preshots_template"],
            "shots_template": prompt_config["shots_template"],
            "postshots_template": prompt_config["postshots_template"],
            "shots_separator": prompt_config["shots_separator"],
            "answers_templates": prompt_config["answers_templates"],
            "batch_size": method_config["batch_size"],
        }
    elif task == "language_model" and model_type == "litgpt" and method == "full_ft":
        model_cls = LanguageModelLitGPTFullFT
        model_init_args = {
            "checkpoint_dir": model_config["checkpoint_dir"],
            "preshots_template": prompt_config["preshots_template"],
            "shots_template": prompt_config["shots_template"],
            "postshots_template": prompt_config["postshots_template"],
            "shots_separator": prompt_config["shots_separator"],
            "answers_templates": prompt_config["answers_templates"],
            "batch_size": method_config["batch_size"],
        }
    else:
        raise ValueError(f"Invalid combination of task ({task}), checkpoint_dir ({model_config['checkpoint_dir']}) and method ({method})")
    model_init_args["data_load_fn"] = data_load_fn
    model_init_args["data_cache_dir"] = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache"
    model = model_cls(**model_init_args)

    # ---------------------
    # Trainer
    # ---------------------
    trainer = Trainer(
        accelerator = model_config.get("accelerator", "auto"),
        strategy = model_config.get("strategy", "auto"),
        devices = model_config.get("devices", "auto"),
        precision = model_config.get("precision", "32-true"),
        plugins = None,
        callbacks = None,
        max_epochs = method_config.get("max_epochs", 1000),
        max_steps = method_config.get("max_steps", None),
        grad_accum_steps = method_config.get("grad_accum_steps", 1),
        limit_train_batches = method_config.get("limit_train_batches", float("inf")),
        limit_val_batches = method_config.get("limit_val_batches", float("inf")),
        validation_frequency = method_config.get("validation_frequency", 1),
        use_distributed_sampler = True,
        checkpoint_dir = results_dir,
        checkpoint_frequency = method_config.get("checkpoint_frequency", 1),
        random_state = method_config.get("random_state", 42),
    )

    # ---------------------
    # Fit the model
    # ---------------------
    trainer.fit(model)
    return
    # ---------------------
    # Predict
    # ---------------------
    trainer.predict(model)

    # ---------------------
    # Save results
    # ---------------------
    t_end = perf_counter()
    with open(os.path.join(results_dir, "done.txt"), "w") as f:
        f.write(f"Execution time: {t_end - t_start} seconds")


if __name__ == "__main__":
    from fire import Fire
    import torch
    torch.set_float32_matmul_precision("high")
    Fire(main)