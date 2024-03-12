
import os
from llmcal.utils import load_yaml
from llmcal.model.utils import load_model

def main(
    model: str,
    train_task: str,
    test_task: str,
    splits: str, 
):

    # Load the train and test dataset:
    print("Loading the data...")
    split_config = load_yaml(f"configs/split/{splits}.yaml")
    train_task_args = load_yaml(f"configs/dataset/{train_task}.yaml")
    train_dataset = load_dataset_and_cast_task(
        dataset=train_task_args["task"], 
        prompt_config=train_task_args["prompt"],
        split="train",
        n_samples=split_config["train_samples"],
        random_state=split_config["random_state"]
    )
    val_dataset = load_dataset_and_cast_task(
        dataset=train_task_args["task"], 
        prompt_config=train_task_args["prompt"],
        split="validation",
        n_samples=split_config["validation_samples"],
        random_state=split_config["random_state"]
    )
    eval_task_args = load_yaml(f"configs/dataset/{test_task}.yaml")
    test_dataset = load_dataset_and_cast_task(
        dataset=eval_task_args["task"], 
        prompt_config=eval_task_args["prompt"],
        split="test",
        n_samples=split_config["test_samples"],
        random_state=split_config["random_state"]
    )
    # {split}_dataset is dataset with columns [idx, input, target]
    # input could be (prompt, anwsers) or numpy array of features
    # target is the output to predict, could be a string or an int

    # Prepare results directory
    results_dir = f"experiments/{model}/{train_task}/{test_task}/{splits}"
    os.makedirs(results_dir, exist_ok=True)

    # Init model and trainer
    print("Loading the model...")
    model_args = load_yaml(f"configs/model/{model}.yaml")
    model, trainer = load_model(model_args)
    # model is nn.Module and trainer is a trainer for LM, LMClassification or Classification
    # or a do-nothing trainer for feature extraction
    
    # Fit the model to the dataset
    print("Training the model...")
    trainer.fit(model, train_dataset, val_dataset)
    
    # Predict on all data
    print(f"Predicting on the train set...")
    results = trainer.predict(train_dataset)
    results.save_to_disk(f"{results_dir}/train")
    
    print(f"Predicting on the validation set...")
    results = trainer.predict(val_dataset)
    results.save_to_disk(f"{results_dir}/validation")

    print(f"Predicting on the test set...")
    results = trainer.predict(test_dataset)
    results.save_to_disk(f"{results_dir}/test")



if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    