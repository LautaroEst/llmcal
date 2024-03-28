
import os
from llmcal.utils import load_yaml, save_yaml
from llmcal.model.utils import load_model
from llmcal.data.utils import load_dataset_and_cast_task

def main(
    model: str,
    task: str,
    splits: str,
    **mods,
):
    
    # Parse arguments:
    args = {
        "model": load_yaml(os.path.join("configs/model",f"{model}.yaml")),
        "task": load_yaml(os.path.join("configs/task",f"{task}.yaml")),
        "splits": load_yaml(os.path.join("configs/splits",f"{splits}.yaml")),
    }

    model_with_mods_name = model
    for mod in mods:
        if "." in mod:
            k_1, k_2 = mod.split(".")
            args["model"][k_1][k_2] = mods[mod]
        else:
            args["model"][mod] = mods[mod]
        model_with_mods_name += f"_{mod}={mods[mod]}"
    results_dir = f"experiments/{task}/{model_with_mods_name}/{splits}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 20)
    print("Experiment configuration:\n")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Splits: {splits}")
    print()

    if os.path.exists(os.path.join(results_dir, "config.yaml")):
        print("Experiment already done.\n" + "=" * 20 + "\n")
        return

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

    # Init model and trainer
    print("Loading the model...")
    model, trainer = load_model(args["model"], model_checkpoint_dir=os.path.join(results_dir,".cache"))
    # model is nn.Module and trainer is a trainer for LM, LMClassification or Classification
    # or a do-nothing trainer for feature extraction
    
    # Fit the model to the dataset
    print("Training the model...")
    trainer.fit(model, train_dataset, val_dataset)
    if os.path.exists(os.path.join(results_dir,".cache","training.interrupted")):
        print("=" * 20 + "\n")
        return
    
    # Predict on all data
    if not os.path.exists(os.path.join(results_dir,".cache","train_prediction.success")):
        print(f"Predicting on the train set...")
        results = trainer.predict(model, train_dataset, prefix="train") # results is a dataset with columns [idx, input, target, output]
        results.save_to_disk(f"{results_dir}/train")
    
    if not os.path.exists(os.path.join(results_dir,".cache","validation_prediction.success")):
        print(f"Predicting on the validation set...")
        results = trainer.predict(model, val_dataset, prefix="validation")
        results.save_to_disk(f"{results_dir}/validation")

    if not os.path.exists(os.path.join(results_dir,".cache","test_prediction.success")):
        print(f"Predicting on the test set...")
        results = trainer.predict(model, test_dataset, prefix="test")
        results.save_to_disk(f"{results_dir}/test")

    interrupted_files = [f for f in os.listdir(os.path.join(results_dir,".cache")) if f.endswith(".interrupted")]
    if not interrupted_files:
        save_yaml(args, f"{results_dir}/config.yaml")

    print("Done!\n" + "=" * 20 + "\n")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    
    # TODO:
    #   - Add logging with logger
    #   - Replace datasets for a custom class that reads metadata and doesn't copy all the dataset
    #   - Replace the configuration to gin-style