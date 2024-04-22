
from typing import Literal
import lightning as L
from llmcal.data.datasets import SST2Dataset
from llmcal.utils import load_yaml

L.Trainer

def main(
    dataset: Literal["sst2", "20newsgroup", "medical_abstracts", "dbpedia", "banking77"],
    prompt: str,
    data_fold: str,
    base_model: str,
    method: str,
):
    prompt_config = load_yaml(f"configs/prompt/{prompt}.yaml")
    data_fold_config = load_yaml(f"configs/fold/{data_fold}.yaml")
    base_model_config = load_yaml(f"configs/base_model/{base_model}.yaml")
    method_config = load_yaml(f"configs/method/{method}.yaml")

    # Dataset
    dataset_cache_dir = f"experiments/{dataset}/{prompt}/{data_fold}/{base_model}/.data_cache"
    if dataset == "sst2":
        dataset = SST2Dataset(prompt_config["prompt_template"], base_model_config["tokenizer"], data_fold_config["train_samples"], data_fold_config["val_samples"], method_config["batch_size"], dataset_cache_dir, data_fold_config["random_state"])
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # Model
    # model = ...
    
    # Trainer
    # trainer = ...
    
    # Fit the model
    # trainer.fit(model, dataset)        

    # Predict
    # trainer.predict(model, dataset)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)