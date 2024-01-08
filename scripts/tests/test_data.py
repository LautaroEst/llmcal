
import torch
from llmcal.models import load_model_and_tokenizer
from llmcal.data import load_dataset, LoaderWithTemplate


def main():

    model_name = "gpt2-xl"
    # model_name = "meta-llama/Llama-2-7b-hf"
    dataset_name, split = "tony_zhao/sst2", "train"
    subsample = None
    template = "Review: {sentence}\nSentiment:"
    labels = [" negative", " positive"]
    batch_size = 4
    random_state = 0


    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset = load_dataset(dataset_name, split=split, subsample=subsample, random_state=random_state, sort_by_length=True)
    loader = LoaderWithTemplate(
        dataset=dataset,
        template=template,
        labels=labels,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        random_state=random_state,
    )

    batch = next(iter(loader))
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            encoded_labels=batch["encoded_labels"],
        )
    print(output["logits"])
    for idx in batch["idx"]:
        print(dataset[idx])

if __name__ == '__main__':
    main()