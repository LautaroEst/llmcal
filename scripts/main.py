
import os
from typing import Literal
import lightning as L
from llmcal.utils import load_yaml
from llmcal.model.utils import check_model_type
from llmcal.model.modules import *
from llmcal.data.datasets import *
from time import perf_counter
from functools import partial

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
        data_load_fn = partial(load_sst2, data_dir=data_dir, num_train_samples=data_fold_config["train_samples"], num_val_samples=data_fold_config["val_samples"], num_shots=data_fold_config["shots_samples"], random_state=data_fold_config["random_state"])
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
    else:
        raise ValueError(f"Invalid combination of task ({task}), checkpoint_dir ({model_config['checkpoint_dir']}) and method ({method})")
    model_init_args["data_load_fn"] = data_load_fn
    model_init_args["data_cache_dir"] = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache"
    model = model_cls(**model_init_args)

    # ---------------------
    # Trainer
    # ---------------------
    trainer = L.Trainer(
        accelerator=model_config.get("accelerator", "cpu"),
        strategy=model_config.get("strategy", "auto"),
        devices=model_config.get("devices", 1),
        num_nodes=model_config.get("num_nodes", 1),
        precision=model_config.get("precision", None),
    )

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

    # ---------------------
    # Save results
    # ---------------------
    t_end = perf_counter()
    with open(os.path.join(results_dir, "done.txt"), "w") as f:
        f.write(f"Execution time: {t_end - t_start} seconds")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)