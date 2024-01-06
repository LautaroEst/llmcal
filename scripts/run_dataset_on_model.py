
import argparse
from collections import defaultdict
import json
import pickle
import lightning as L
import numpy as np
import os
from tqdm import tqdm
import torch

from llmcal.models import LanguageModelClassifier
from llmcal.data import load_dataset, LoaderWithTemplateCollator, Template

fabric_args = {
    "accelerator": "cpu",
    "devices": 1,
    # "precision": 16,
}

# Initialize fabric
# fabric = L.Fabric(**fabric_args)
# fabric.launch()



SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--templates', type=str, required=True)
    parser.add_argument('--splits', type=str, required=True)
    parser.add_argument('--num_samples', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_embeddings', action="store_true")
    parser.add_argument('--random_state', type=int, default=0)
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

    # Prepare templates
    with open(os.path.join(args.templates), "r") as f:
        templates = json.load(f)

    # Load model
    print( "======================================")
    print(f">>> Loading {args.model_name} model...")
    # with fabric.init_module():
        # model = LanguageModelClassifier.from_model_name(args.model_name)
    # model = fabric.setup(model, move_to_device=True)
    model = LanguageModelClassifier.from_model_name(args.model_name)
    print("Model loaded successfully!\n")
    model.eval()

    # Run model on each dataset
    for template_args in templates:
        run_model_on_dataset(model, template_args, args, random_state=args.random_state)

    print("\nDone!")
    print( "======================================\n\n")
    print()
    

def run_model_on_dataset(model, template_args, args, random_state=0):
    
    template_id = template_args.pop("id")
    batch_size = template_args.pop("batch_size")
    template = Template(**template_args)

    # Run model on dataset
    for split, num_samples in zip(args.splits, args.num_samples):

        print(f">>> Running model on dataset...")
        print(f"\t* Model: {args.model_name}")
        print(f"\t* Dataset: {args.dataset_name}")
        print(f"\t* Split: {split}")
        print(f"\t* Num samples: {num_samples}")
        print(f"\t* Template: {template_id}")

        results_dir = os.path.join(
            args.output_dir, 
            SCRIPT_NAME, 
            args.model_name, 
            args.dataset_name,
            split,
            template_id,
        )
        
        # Prepare dataloaders
        dataset = load_dataset(args.dataset_name, split=split, subsample=num_samples, random_state=random_state, sort_by_length=True)
        dataloader = LoaderWithTemplateCollator(
            dataset=dataset,
            template=template,
            tokenizer=model.tokenizer,
            batch_size=batch_size,
            shuffle=False,
            random_state=random_state
        )
        # dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=True, move_to_device=True)
        dataloader = tqdm(dataloader, leave=False)

        # Run model on dataset
        results = predict(model, dataloader, results_dir, save_embeddings=args.save_embeddings)

        # Save results
        os.makedirs(results_dir, exist_ok=True)
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                np.save(os.path.join(results_dir, f"{k}.npy"), v)
            else:
                with open(os.path.join(results_dir, f"{k}.pkl"), "wb") as f:
                    pickle.dump(v, f)
    
        with open(os.path.join(results_dir, f"template.json"), "w") as f:
            json.dump(template_args, f, indent=4, separators=(',', ': '))


def predict(model, dataloader, results_dir, save_embeddings=False):
    model.eval()
    results = {"ids": [], "logits": [], "labels": [], "features": defaultdict(list)}
    if save_embeddings:
        results["embeddings"] = []

    are_previous_results = os.path.exists(os.path.join(results_dir, f"ids.npy"))
    if are_previous_results:
        if os.path.exists(os.path.join(results_dir, f"embeddings.npy")) and save_embeddings:
            previous_embeddings = torch.from_numpy(np.load(os.path.join(results_dir, f"embeddings.npy")))
        elif save_embeddings:
            are_previous_results = False
        previous_idx = np.load(os.path.join(results_dir, f"ids.npy"))
        previous_logits = torch.from_numpy(np.load(os.path.join(results_dir, f"logits.npy")))
        
    for batch in dataloader:
        if are_previous_results:
            batch_size = len(batch["idx"])
            valid_idx = []
            encoder_output = {"embeddings": []}
            logits_batch = []
            for i in range(batch_size):
                pi = np.where(previous_idx == batch["idx"][i])[0]
                if len(pi) == 0:
                    valid_idx.append(i)
                    encoder_output["embeddings"].append(None)
                    logits_batch.append(None)
                else:
                    if save_embeddings:
                        encoder_output["embeddings"].append(previous_embeddings[pi[0]])
                    logits_batch.append(previous_logits[pi[0]])
            if len(valid_idx) != 0:
                encoded_prompt = {k: batch["encoded_prompt"][k][valid_idx,:] for k in batch["encoded_prompt"]}
                encoded_labels = {ii: {k: v[valid_idx,:] for k, v in d.items()} for ii, d in batch["encoded_labels"].items()}
                with torch.no_grad():
                    eo, lb = model(encoded_prompt, encoded_labels, output_embeddings=save_embeddings)
                counter = 0
                for i in range(batch_size):
                    if logits_batch[i] is None:
                        logits_batch[i] = lb[counter]
                        if save_embeddings:
                            encoder_output["embeddings"][i] = eo["embeddings"][counter]
                        counter += 1
            logits_batch = torch.stack(logits_batch)
            if save_embeddings:
                encoder_output["embeddings"] = torch.stack(encoder_output["embeddings"])
        else:
            with torch.no_grad():
                encoder_output, logits_batch = model(batch["encoded_prompt"], batch["encoded_labels"], output_embeddings=save_embeddings)

        results["ids"].extend(batch["idx"])
        results["logits"].append(logits_batch.cpu().numpy())
        results["labels"].append(batch["label"].cpu().numpy())
        if save_embeddings:
            results["embeddings"].append(encoder_output["embeddings"].cpu().numpy())
        for feature in batch["features"]:
            results["features"][feature].extend(batch["features"][feature])

    results["ids"] = np.array(results["ids"])
    results["logits"] = np.concatenate(results["logits"], axis=0)
    results["labels"] = np.concatenate(results["labels"], axis=0)
    if save_embeddings:
        results["embeddings"] = np.concatenate(results["embeddings"], axis=0)
    results["features"] = {k: v for k, v in results["features"].items()}
    return results

if __name__ == '__main__':
    main()