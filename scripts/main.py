from typing import Any, List, Literal, Optional, Union
import torch
import lightning as L

from llmcal.utils import load_yaml
from llmcal.models.utils import load_model_from_checkpoint

def setup(
    base_model: str,
    method: str,
    global_seed: int = 0
):
    # Parse config arguments
    base_model_config = load_yaml(f"configs/base_model/{base_model}.yaml")
    method_config = load_yaml(f"configs/method/{method}.yaml")
    kwargs = {
        "checkpoint_dir": base_model_config["checkpoint_dir"],
        "model_type": base_model_config["model_type"],
        "method": method_config.pop("method", "no_adaptation"),
        "model_kwargs": method_config,
        "seed": global_seed,
    }

    # Init fabric
    fabric = L.Fabric(
        accelerator=base_model_config.get("accelerator", "cpu"),
        strategy=base_model_config.get("strategy", "auto"),
        devices=base_model_config.get("devices", 1),
        num_nodes=base_model_config.get("num_nodes", 1),
        precision=base_model_config.get("precision", 32),
        plugins=base_model_config.get("plugins", None),
        callbacks=base_model_config.get("callbacks", None),
        loggers=base_model_config.get("loggers", None),
    )
    fabric.launch(main, **kwargs)


def main(
    fabric: L.Fabric,
    checkpoint_dir: str,
    model_type: Literal["language_model", "sequence_classifier"],
    method: Literal["no_adaptation", "full_ft", "lora", "embeddings_finetuning", "affine_calibration"],
    model_kwargs: dict,
    seed: int,
    
):
    fabric.seed_everything(seed)
    
    # Init the model
    model = load_model_from_checkpoint(fabric, checkpoint_dir, model_type, method, **model_kwargs)

    # Get the optimizer
    optimizer = model.configure_optimizers()
    optimizer.zero_grad()

    # Get the dataloader
    dataset = DummyDataset()
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    




if __name__ == "__main__":
    from fire import Fire
    torch.set_float32_matmul_precision("high")
    Fire(setup)

