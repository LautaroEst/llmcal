
from llmcal.utils import load_yaml

def main(
    data: str,
    model: str,
    prompt: str,
):
    # Load data
    data_args = load_yaml(data)
    dataset = load_dataset(data_args)

    # Load prompt
    prompt_args = load_yaml(prompt)
    prompt, answer = load_prompt(prompt_args)
    dataloaders = create_dataloader(dataset, prompt, answer)

    # Load model and trainer
    model_args = load_yaml(model)
    fabric = init_fabric(model_args)
    model = load_model(model_args)
    trainer = load_trainer(model_args)

    # Train model and predict
    trainer.fit(model, dataloaders["train"], dataloaders["validation"])
    trainer.predict(model, dataloaders["train"])
    trainer.predict(model, dataloaders["validation"])
    trainer.predict(model, dataloaders["test"])



if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    