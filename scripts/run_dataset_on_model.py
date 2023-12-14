
import argparse
import json
import lightning as L

from llmcal.models import LanguageModelClassifier
from llmcal.data import load_dataset, LoaderWithTemplateCollator, SubstitutionTemplate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

fabric = L.Fabric(accelerator="cpu", precision="32-true")

def predict_step(self, batch, batch_idx, dataloader_idx=0):
    _, logits = self(batch["encoded_prompt"], batch["encoded_labels"])
    logits = logits.cpu().numpy()
    labels = batch["label"].cpu().numpy()
    ids = batch["original_id"]
    prompts = batch["prompt"]
    return ids, prompts, logits, labels


def main():

    # Read command args
    args = parse_args()

    # Load model
    with fabric.init_module():
        model = LanguageModelClassifier.from_model_name(args.model_name)

    with open(args.template, "r") as f:
        template_file = json.load(f)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="validation")
    template = SubstitutionTemplate(**template_file)
    dataloader = LoaderWithTemplateCollator(
        dataset=dataset,
        template=template,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.seed
    )

    batch = next(iter(dataloader))
    print(batch)


if __name__ == '__main__':
    main()