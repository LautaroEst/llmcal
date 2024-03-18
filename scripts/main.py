
import os
from llmcal.utils import load_yaml, save_yaml
from llmcal.model.utils import load_model
from llmcal.data.utils import load_dataset_and_cast_task

def main(
    model: str,
    task: str,
    splits: str, 
):
    print("=" * 20)
    print("Starting the experiment...")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Splits: {splits}")
    print()

    # Parse arguments:
    args = {
        "model": load_yaml(f"configs/model/{model}.yaml"),
        "task": load_yaml(f"configs/task/{task}.yaml"),
        "splits": load_yaml(f"configs/splits/{splits}.yaml"),
    }

    # Load the datasets and cast to the task
    print("Loading the data...")
    train_dataset, train_cast = load_dataset_and_cast_task(
        dataset=args["task"]["task"], 
        split="train",
        n_samples=args["splits"]["train_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=args["task"]["casting"],
    )
    args["splits"]["train_samples"] = len(train_dataset)
    val_dataset, _ = load_dataset_and_cast_task(
        dataset=args["task"]["task"], 
        split="validation",
        n_samples=args["splits"]["validation_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=train_cast,
    )
    args["splits"]["validation_samples"] = len(val_dataset)
    test_dataset, _ = load_dataset_and_cast_task(
        dataset=args["task"]["task"], 
        split="test",
        n_samples=args["splits"]["test_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=train_cast,
    )
    args["splits"]["test_samples"] = len(test_dataset)

    # {split}_dataset is dataset with columns [idx, input, target]
    # input could be (prompt, anwsers) or numpy array of features
    # target is the output to predict, could be a string or an int

    # Prepare results directory
    results_dir = f"experiments/{task}/{model}/{splits}"
    os.makedirs(results_dir, exist_ok=True)

    # Init model and trainer
    print("Loading the model...")
    model, trainer = load_model(args["model"], model_checkpoint_dir=os.path.join(results_dir,".cache"))
    # model is nn.Module and trainer is a trainer for LM, LMClassification or Classification
    # or a do-nothing trainer for feature extraction
    
    # Fit the model to the dataset
    print("Training the model...")
    trainer.fit(model, train_dataset, val_dataset)
    
    # Predict on all data
    print(f"Predicting on the train set...")
    results = trainer.predict(model, train_dataset) # results is a dataset with columns [idx, input, target, output]
    results.save_to_disk(f"{results_dir}/train")
    
    print(f"Predicting on the validation set...")
    results = trainer.predict(model, val_dataset)
    results.save_to_disk(f"{results_dir}/validation")

    print(f"Predicting on the test set...")
    results = trainer.predict(model, test_dataset)
    results.save_to_disk(f"{results_dir}/test")

    save_yaml(args, f"{results_dir}/config.yaml")

    print("Done!")
    print("=" * 20)
    print()


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    
    # TODO:
    #   - Add logging with logger
    #   - Replace datasets for a custom class that reads metadata and doesn't copy all the dataset
    #   - Replace the configuration to gin-style