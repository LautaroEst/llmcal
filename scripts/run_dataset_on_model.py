
import argparse
import json
import lightning as L
import numpy as np
import os
from tqdm import tqdm
import torch

from llmcal.models import LanguageModelClassifier
from llmcal.data import load_dataset, LoaderWithTemplateCollator, Template

fabric_args = {
    "accelerator": "cpu",
    "precision": 32,
}


SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--templates_path', type=str, required=True)
    parser.add_argument('--splits', type=str, required=True)
    parser.add_argument('--num_samples', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_embeddings', action="store_true")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.splits = args.splits.split(",")
    if args.num_samples is not None:
        num_samples = args.num_samples.split(",")
        num_samples = [int(s) if s != "None" else None for s in num_samples]
    else:
        num_samples = [None] * len(args.splits)
    args.num_samples = num_samples
    return args

def main():

    # Read command args
    args = parse_args()

    # Initialize fabric
    fabric = L.Fabric(**fabric_args)

    # Load model
    with fabric.init_module():
        model = LanguageModelClassifier.from_model_name(args.model_name)

    # Prepare templates
    with open(os.path.join(args.templates_path,f"{args.dataset_name}.json"), "r") as f:
        templates = json.load(f)
    
    for template_args in templates:
        template_id = template_args.pop("id")
        template = Template(**template_args)

        # Run model on dataset
        for split, num_samples in zip(args.splits, args.num_samples):

            results_dir = os.path.join(
                args.output_dir, 
                SCRIPT_NAME, 
                args.model_name, 
                args.dataset_name, 
                template_id,
                str(args.seed)
            )
            if os.path.exists(os.path.join(results_dir, f"{split}.ids.npy")):
                continue
            
            # Prepare dataloaders
            dataset = load_dataset(args.dataset_name, split=split, subsample=num_samples, random_state=args.seed, sort_by_length=True)
            dataloader = LoaderWithTemplateCollator(
                dataset=dataset,
                template=template,
                tokenizer=model.tokenizer,
                batch_size=args.batch_size,
                shuffle=False,
                random_state=args.seed
            )
            dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=True, move_to_device=True)
            dataloader = tqdm(dataloader, desc=f"{split} split", leave=False)

            # Run model on dataset
            results = run_predictions(model, dataloader, save_embeddings=args.save_embeddings)

            # Save results
            os.makedirs(results_dir, exist_ok=True)
            for k, v in results.items():
                np.save(os.path.join(results_dir, f"{split}.{k}.npy"), v)
        with open(os.path.join("/".join(results_dir.split("/")[:-1]), f"template.json"), "w") as f:
            json.dump(template_args, f)


def run_predictions(model, dataloader, save_embeddings=False):
    model.eval()
    results = {"ids": [], "prompts": [], "logits": [], "labels": []}
    if save_embeddings:
        results["embeddings"] = []
    for batch in dataloader:
        with torch.no_grad():
            encoder_output, logits_batch = model(batch["encoded_prompt"], batch["encoded_labels"], output_embeddings=save_embeddings)
        results["ids"].extend(batch["idx"])
        results["prompts"].extend(batch["prompt"])
        results["logits"].append(logits_batch.cpu().numpy())
        results["labels"].append(batch["label"].cpu().numpy())
        if save_embeddings:
            results["embeddings"].append(encoder_output["embeddings"].cpu().numpy())

    results["ids"] = np.array(results["ids"])
    results["prompts"] = np.array(results["prompts"])
    results["logits"] = np.concatenate(results["logits"], axis=0)
    results["labels"] = np.concatenate(results["labels"], axis=0)
    if save_embeddings:
        results["embeddings"] = np.concatenate(results["embeddings"], axis=0)
    return results

if __name__ == '__main__':
    main()