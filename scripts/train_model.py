
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
    # Read the config files
    dataset_args = load_yaml(f"configs/dataset/{dataset}.yaml")
    model_args = load_yaml(f"configs/model/{model}.yaml")

    # Load the dataset
    dataset = load_dataset(dataset_args)

    # Init prompt and train it
    if prompt is not None:
        prompt_args = load_yaml(f"configs/prompt/{prompt}.yaml")
        prompt = load_prompt(prompt_args)
        prompt.fit(dataset["train"])

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
    