
import os
from typing import Optional
from llmcal.utils import load_yaml
from llmcal.data.utils import load_dataset
from llmcal.prompt.utils import load_prompt
from llmcal.model.utils import load_model

def main(
    dataset: str,
    model: str,
    prompt: Optional[str] = None,
):
    
    # Prepare results directory
    if prompt is not None:
        results_dir = f"experiments/{dataset}/{prompt}/{model}"
    else:
        results_dir = f"experiments/{dataset}/no_prompt/{model}"
    os.makedirs(results_dir, exist_ok=True)

    # Load the dataset
    print("Loading the dataset...")
    dataset_args = load_yaml(f"configs/dataset/{dataset}.yaml")
    dataset = load_dataset(**dataset_args)

    # Init prompt and train it
    if prompt is not None:
        print("Training the prompt...")
        prompt_args = load_yaml(f"configs/prompt/{prompt}.yaml")
        prompt = load_prompt(prompt_args)
        prompt.fit(dataset["train"])

    # Init model and trainer
    print("Loading the model...")
    model_args = load_yaml(f"configs/model/{model}.yaml")
    model, trainer = load_model(model_args)
    
    # Fit the model to the dataset
    print("Training the model...")
    trainer.fit(model, prompt, dataset["train"], dataset["validation"])
    
    # Predict on all data
    for split in ["train", "validation", "test"]:
        print(f"Predicting on {split} set...")
        results = trainer.predict(dataset[split])
        results.save_to_disk(f"{results_dir}/{split}")



if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    