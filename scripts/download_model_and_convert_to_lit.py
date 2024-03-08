
import argparse
import os
import sys
import gc
import json
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import shutil

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor
from lightning_utilities.core.imports import RequirementCache

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")
_HF_TRANSFER_AVAILABLE = RequirementCache("hf_transfer")

from lit_gpt import Config
from lit_gpt.utils import incremental_save, lazy_load


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "gpt_neox.embed_in.weight": "transformer.wte.weight",
        "gpt_neox.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "gpt_neox.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "gpt_neox.layers.{}.attention.query_key_value.bias": "transformer.h.{}.attn.attn.bias",
        "gpt_neox.layers.{}.attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "gpt_neox.layers.{}.attention.dense.bias": "transformer.h.{}.attn.proj.bias",
        "gpt_neox.layers.{}.attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "gpt_neox.layers.{}.attention.rotary_emb.inv_freq": None,
        "gpt_neox.layers.{}.attention.bias": None,
        "gpt_neox.layers.{}.attention.masked_bias": None,
        "gpt_neox.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "gpt_neox.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": "transformer.h.{}.mlp.fc.bias",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": "transformer.h.{}.mlp.proj.bias",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "gpt_neox.final_layer_norm.bias": "transformer.ln_f.bias",
        "gpt_neox.final_layer_norm.weight": "transformer.ln_f.weight",
        "embed_out.weight": "lm_head.weight",
    }

    for name, param in hf_weights.items():
        if "gpt_neox.layers" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_falcon(
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "transformer.word_embeddings.weight": "transformer.wte.weight",
        "transformer.h.{}.self_attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "transformer.h.{}.self_attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "transformer.h.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "transformer.h.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if "7b" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            }
        )
    elif "40b" in model_name or "180B" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.ln_attn.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.ln_attn.weight": "transformer.h.{}.norm_1.weight",
                "transformer.h.{}.ln_mlp.bias": "transformer.h.{}.norm_2.bias",
                "transformer.h.{}.ln_mlp.weight": "transformer.h.{}.norm_2.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in hf_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{l}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{l}.norm_2.bias",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config._mlp_class == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.block_sparse_moe.gate.weight": "transformer.h.{l}.mlp.gate.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "transformer.h.{l}.mlp.experts.{e}.fc_1.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "transformer.h.{l}.mlp.experts.{e}.fc_2.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "transformer.h.{l}.mlp.experts.{e}.proj.weight",
            }
        )
    elif config._mlp_class in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{l}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": "transformer.h.{l}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l = layer_template(name, 2)
            e = None
            if "block_sparse_moe.experts" in name:
                from_name, e = layer_template(from_name, 5)
            qkv = qkv_weights.setdefault(l, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l=l, e=e)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    # convert separate q, k, v matrices into an interleaved qkv
    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype)
        k = load_param(k, f"layer {i} k", dtype)
        v = load_param(v, f"layer {i} v", dtype)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]

def copy_weights_phi(
    config: Config,
    qkv_weights: dict,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    if any(layer_name.startswith(("layers.", "transformer.")) for layer_name in hf_weights):
        raise ValueError(
            "You are using an outdated Phi checkpoint. Please reload it as described in 'tutorials/download_phi.md'"
        )

    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.dense.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.dense.bias": "transformer.h.{}.attn.proj.bias",
        "model.layers.{}.mlp.fc1.weight": "transformer.h.{}.mlp.fc.weight",
        "model.layers.{}.mlp.fc1.bias": "transformer.h.{}.mlp.fc.bias",
        "model.layers.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "model.final_layernorm.weight": "transformer.ln_f.weight",
        "model.final_layernorm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }

    for name, param in hf_weights.items():
        if name.startswith("model.layers."):
            from_name, l = layer_template(name, 2)
            qkv = qkv_weights.setdefault(l, defaultdict(dict))
            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # split across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype)
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    config_dict = asdict(config)
    print(f"Model config {config_dict}")
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config_dict, json_config)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, model_name)
    elif config._mlp_class in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    elif "phi" in model_name:
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_phi, config, qkv_weights)
    else:
        copy_fn = copy_weights_gpt_neox

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
            hf_weights = lazy_load(bin_file)
            copy_fn(sd, hf_weights, saver=saver, dtype=dtype)
        gc.collect()
        print("Saving converted checkpoint")
        saver.save(sd)


def download_from_hub(
    repo_id: Optional[str] = None,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    tokenizer_only: bool = False,
    checkpoint_dir: Path = Path("checkpoints"),
) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['hf_config']['org']}/{config['hf_config']['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    from huggingface_hub import snapshot_download, list_repo_files

    if ("meta-llama" in repo_id or "falcon-180" in repo_id) and not access_token:
        raise ValueError(
            f"{repo_id} requires authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --access_token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )

    all_files_in_repo = list_repo_files(repo_id)
    from_safetensors = not any(file.endswith(".bin") for file in all_files_in_repo)

    download_files = ["tokenizer*", "generation_config.json"]
    if not tokenizer_only:
        if from_safetensors:
            if not _SAFETENSORS_AVAILABLE:
                raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
            download_files.append("*.safetensors*")
        else:
            # covers `.bin` files and `.bin.index.json`
            download_files.append("*.bin*")

    import huggingface_hub._snapshot_download as download
    import huggingface_hub.constants as constants

    previous = constants.HF_HUB_ENABLE_HF_TRANSFER
    if _HF_TRANSFER_AVAILABLE and not previous:
        print("Setting HF_HUB_ENABLE_HF_TRANSFER=1")
        constants.HF_HUB_ENABLE_HF_TRANSFER = True
        download.HF_HUB_ENABLE_HF_TRANSFER = True

    directory = checkpoint_dir / repo_id
    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=download_files,
        token=access_token,
    )

    constants.HF_HUB_ENABLE_HF_TRANSFER = previous
    download.HF_HUB_ENABLE_HF_TRANSFER = previous

    # convert safetensors to PyTorch binaries
    if from_safetensors:
        from safetensors import SafetensorError
        from safetensors.torch import load_file as safetensors_load

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
            print(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", 
        type=str, 
        help="Repository ID"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        help="Data type of the checkpoint", 
        default=None
    )
    return parser.parse_args()


def main():

    args = parse_args()

    checkpoint_dir = os.getenv("LIT_CHECKPOINTS")
    if checkpoint_dir is None:
        # add the variable LIT_CHECKPOINTS to .bashrc
        print("Select a directory to store the checkpoints:\n>>> ", end="")
        checkpoint_dir = input()
        with open(str(Path.home() / ".bashrc"), "a") as f:
            f.write(f"\nexport LIT_CHECKPOINTS={checkpoint_dir}")
        os.environ["LIT_CHECKPOINTS"] = checkpoint_dir
        print("Models will be stored in", checkpoint_dir)

    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    elif (checkpoint_dir / args.repo_id / "lit_model.pth").exists():
        print(f"{args.repo_id} is already downloaded and converted")
        return

    download_from_hub(
        repo_id = args.repo_id,
        access_token = os.getenv("HF_TOKEN"),
        tokenizer_only = False,
        checkpoint_dir = checkpoint_dir,
    )

    convert_hf_checkpoint(
        checkpoint_dir = checkpoint_dir / args.repo_id, 
        model_name = args.repo_id, 
        dtype = args.dtype
    )

    for p in (checkpoint_dir / args.repo_id).glob("*.safetensors"):
        os.remove(p)
    for p in (checkpoint_dir / args.repo_id).glob("*.bin"):
        os.remove(p)
    


if __name__ == "__main__":
    main()