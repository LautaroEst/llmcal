
import os
import gc
import json
from pathlib import Path
from litgpt.utils import incremental_save, lazy_load
from litgpt.scripts.download import download_from_hub
import torch
import lightning as L
from transformers import AutoConfig, BertModel


def compress_binfiles(checkpoint_dir: Path) -> None:

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_dir.glob("*.bin"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")
    
    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end
        for bin_file in sorted(bin_files):
            print("Processing", bin_file)
            import pdb; pdb.set_trace()
            hf_weights = torch.load(bin_file)
            for name, param in hf_weights.items():
                print("Saving", name)
                sd[name] = saver.store_early(param)
        gc.collect()
        print("Saving converted checkpoint")
        saver.save(sd)


def main():
    model_name = "bert-base-uncased"
    # model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    # checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS"))
    # download_from_hub(model_name, dtype="float32", checkpoint_dir=checkpoint_dir, convert_checkpoint=False)
    # compress_binfiles(checkpoint_dir / model_name)

    fabric = L.Fabric(accelerator="cpu", devices=2, precision="bf16-true")
    fabric.launch()

    with fabric.init_module():
        model = BertModel.from_pretrained(model_name)
    print(model.encoder.layer[0].attention.self.query.weight)
    print(model.encoder.layer[0].attention.self.query.weight.dtype)


if __name__ == "__main__":
    main()