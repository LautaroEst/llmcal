
import torch
import lightning as L

from llmcal.models import init_model
from llmcal.data.datasets import DummyDataset

def setup(
    accelerator="cpu",
    precision="bf16-true",
    devices=2,
    **kwargs
):
    # Initialize the fabric
    fabric = L.Fabric(accelerator=accelerator, precision=precision, devices=devices)
    fabric.launch(main, **kwargs)


def main(
    fabric,
    seed = 8392,
    model_class = "LitGPT",
    checkpoint_dir = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T-bfloat16",
    num_epochs = 10,
    batch_size = 2,
    micro_batch_size = 1,
    loss_fn = "cross_entropy",
    learning_rate = 1e-3,
    weight_decay = 0.0,
    eval_every_n_steps = 5,
):
    
    # Set the seed
    fabric.seed_everything(seed)
    
    # Initialize the model
    model = init_model(
        fabric, 
        model_class, 
        checkpoint_dir,
        loss_fn = loss_fn,
        learning_rate = learning_rate, 
        weight_decay = weight_decay, 
        batch_size = batch_size, 
        micro_batch_size = micro_batch_size
    )
    
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

    iter_train_dataloader = iter(train_dataloader)
    print(next(iter_train_dataloader))
    fabric.print("Finished OK")


if __name__ == "__main__":
    from fire import Fire
    torch.set_float32_matmul_precision("high")
    Fire(setup)

