import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn
import os
from transformers import AutoModelForSequenceClassification
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from litgpt.utils import load_checkpoint


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.seq = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 2),
        )
        self.l2 = nn.Linear(2, 2)

class MyModule(L.LightningModule):

    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = BaseModel()
        print(self.model.state_dict())
        checkpoint = lazy_load(self.checkpoint)
        checkpoint["state_dict"] = {"model." + k: v for k, v in checkpoint["state_dict"].items()}
        self.load_state_dict(checkpoint["state_dict"], assign=False)
        print(self.model.state_dict())

    def forward(self, x):
        return self.model.l2(self.model.seq(self.model.l1(x)))
    
    def training_step(self, batch, batch_idx):
        return self(batch).sum()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def train_dataloader(self) -> torch.Any:
        return torch.utils.data.DataLoader(torch.rand(3, 2, device=torch.device("cuda")), batch_size=2)
    
    def on_load_checkpoint(self, checkpoint: torch.Dict[str, TRAIN_DATALOADERS]) -> None:
        print({k: v.device for k, v in checkpoint["state_dict"].items()})



def main():

    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=1)
    # trainer = L.Fabric(accelerator="gpu", devices=1)

    if not os.path.exists("./state_dict.pt"):
        base_model = BaseModel()
        torch.save({"state_dict": base_model.state_dict()}, "./state_dict.pt")
    
    with trainer.init_module():
        model = MyModule("state_dict.pt")

    # model = MyModule.load_from_checkpoint("./state_dict.pt")
    # with trainer.init_module():
        # model = MyModule.load_from_checkpoint("./state_dict.pt")
        # model = MyModule()
        # state_dict = torch.load("./state_dict.pt")
        # model.load_state_dict(state_dict["state_dict"])
        # model = AutoModelForSequenceClassification.from_pretrained("gpt2")
    # print(model)
    # model = MyModule()
    # trainer.load_raw("state_dict.pt", model, strict=False)
    # print(model.state_dict())
    # print({k: v.device for k, v in state_dict["state_dict"].items()})
    # print({k: v.device for k, v in model.state_dict().items()})





if __name__ == "__main__":
    main()