
import os
from typing import Optional
from llmcal.utils import load_yaml
from llmcal.data.utils import load_dataset
from llmcal.prompt.utils import load_prompt
from llmcal.model.utils import load_model

experiment_name = __file__.split("/")[-1].split(".")[0]

def main(
    dataset: str,
    model: str,
    prompt: Optional[str] = "no_prompt",
):
    
    # Prepare results directory
    results_dir = f"results/{experiment_name}/{dataset}/{model}/{prompt}"
    os.makedirs(results_dir, exist_ok=True)

    # Read the config files
    dataset_args = load_yaml(f"configs/{experiment_name}/dataset/{dataset}.yaml")
    model_args = load_yaml(f"configs/{experiment_name}/model/{model}.yaml")
    prompt_args = load_yaml(f"configs/{experiment_name}/prompt/{prompt}.yaml")

    # Load the dataset
    dataset = load_dataset(**dataset_args)

    # Init prompt and train it
    if prompt_args["class_name"] is not None:
        prompt = load_prompt(prompt_args)
        prompt.fit(dataset["train"])
    else:
        prompt = None

    # Init model and trainer
    model, trainer = load_model(model_args)
    
    # Fit the model to the dataset
    trainer.fit(model, prompt, dataset["train"], dataset["validation"])
    
    # Predict on all data
    results = {}
    for split in ["train", "validation", "test"]:
        results[split] = trainer.predict(dataset[split])



if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    