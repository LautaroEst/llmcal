
import os
from typing import Optional, List
from llmcal.utils import load_yaml, perform_modifications
from llmcal.model.utils import load_model
from llmcal.data.utils import load_dataset_and_cast_task

def main(
    model: str,
    train_task: str,
    test_task: str,
    splits: str, 
    mods: Optional[List[str]] = [],
):
    print("=" * 20)
    print("Starting the experiment...")
    print(f"Model: {model}")
    print(f"Train task: {train_task}")
    print(f"Test task: {test_task}")
    print(f"Splits: {splits}")
    print(f"Modifications: {mods}")
    print()

    # Parse arguments:
    model_args = load_yaml(f"configs/model/{model}.yaml")
    train_task_args = load_yaml(f"configs/task/{train_task}.yaml")
    eval_task_args = load_yaml(f"configs/task/{test_task}.yaml")
    split_config = load_yaml(f"configs/splits/{splits}.yaml")
    args = perform_modifications(
        {"model": model_args, "train_task": train_task_args, "eval_task": eval_task_args, "splits": split_config}, mods
    )

    # Load the train and test dataset:
    print("Loading the data...")
    train_dataset, train_cast = load_dataset_and_cast_task(
        dataset=args["train_task"]["task"], 
        split="train",
        n_samples=args["splits"]["train_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=args["train_task"]["casting"],
    )
    val_dataset, _ = load_dataset_and_cast_task(
        dataset=args["train_task"]["task"], 
        split="validation",
        n_samples=args["splits"]["validation_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=train_cast,
    )
    test_dataset, test_cast = load_dataset_and_cast_task(
        dataset=args["eval_task"]["task"], 
        split="test",
        n_samples=args["splits"]["test_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=args["eval_task"]["casting"],
    )
    # {split}_dataset is dataset with columns [idx, input, target]
    # input could be (prompt, anwsers) or numpy array of features
    # target is the output to predict, could be a string or an int

    # Prepare results directory
    results_dir = f"experiments/{model}/{train_task}/{test_task}/{splits}"
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

    print("Done!")
    print("=" * 20)
    print()


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    