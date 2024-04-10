import os

import torch
from llmcal.utils import load_yaml, save_yaml
from llmcal.model.utils import load_model
from llmcal.data.utils import load_dataset_and_cast_task


def main():
    args = {
        "model": load_yaml(os.path.join("configs/model/tinyllama_3T_bf16.yaml")),
        "task": load_yaml(os.path.join("configs/task/medical_abstracts.yaml")),
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

    train_dataset = train_dataset.select_columns(["input","target"]).with_format("torch")

    print("Loading the model...")
    model, trainer = load_model(args["model"], model_checkpoint_dir=".cache")
    # data = train_dataset.map(lambda x: {"length": model.tokenizer([x["input"]["prompt"]])["input_ids"].size(1)})
    # data = data.filter(lambda x: x["length"] >= 2048)
    data = train_dataset#.select(range(20))
    
    # trainer.predict(model, data)
    
    idx = 839
    print(data[idx])
    encoded_prompt = model.tokenizer([data[idx]["input"]["prompt"]])
    prompt_ids = encoded_prompt["input_ids"].to(trainer.fabric.device)#[:,:2048]
    prompt_mask = encoded_prompt["attention_mask"].to(trainer.fabric.device)#[:,:2048]
    answers_ids = [[model.tokenizer([answer])["input_ids"][0,1:].to(trainer.fabric.device) for answer in data[idx]["input"]["answers"]]]
    output = model.predict_step(prompt_ids=prompt_ids, prompt_mask=prompt_mask, answers_ids=answers_ids)
    print(output)
    output = super(model.__class__,model).forward(prompt_ids)
    index = output["logits"][0,-1,:].sort(descending=True).indices[:10]
    print([model.tokenizer.tokenizer.decode(torch.tensor([i])) for i in index])


if __name__ == "__main__":
    main()