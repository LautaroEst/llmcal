import os
from llmcal.utils import load_yaml, save_yaml
from llmcal.model.utils import load_model
from llmcal.data.utils import load_dataset_and_cast_task


def main():
    args = {
        "model": load_yaml(os.path.join("configs/model/tinyllama.yaml")),
        "task": load_yaml(os.path.join("configs/task/refind_inst_0-shot_prompt.yaml")),
        "splits": load_yaml(os.path.join("configs/splits/all.yaml")),
    }

    print("Loading the data...")
    train_dataset, train_cast = load_dataset_and_cast_task(
        dataset=args["task"]["task"], 
        split="train",
        n_samples=args["splits"]["train_samples"],
        random_state=args["splits"]["random_state"],
        cast_obj_or_config=args["task"]["casting"],
    )

    print(train_dataset[0]["target"])    
    train_dataset = train_dataset.select_columns(["input","target"]).flatten().with_format("torch")
    print(train_dataset[0]["target"])    

    # print("Loading the model...")
    # model, trainer = load_model(args["model"], model_checkpoint_dir=".cache")
    # data = train_dataset.map(lambda x: {"length": model.tokenizer([x["input"]["prompt"]])["input_ids"].size(1)})
    # data = data.filter(lambda x: x["length"] >= 2048)
    # import pdb; pdb.set_trace()
    # trainer.predict(model, data)
    # encoded_prompt = model.tokenizer([data[0]["input"]["prompt"]])
    # prompt_ids = encoded_prompt["input_ids"].to(trainer.fabric.device)#[:,:2048]
    # prompt_mask = encoded_prompt["attention_mask"].to(trainer.fabric.device)#[:,:2048]
    # answers_ids = [[model.tokenizer([answer])["input_ids"][0,1:].to(trainer.fabric.device) for answer in data[0]["input"]["answers"]]]
    # output = model(prompt_ids=prompt_ids, prompt_mask=prompt_mask, answers_ids=answers_ids)

if __name__ == "__main__":
    main()