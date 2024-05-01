

import torch
import lightning as L
import torch.utils
import torch.utils.data


class TestModule(L.LightningModule):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        print(self.l1, x)
        return self.l1(x)
    
    def setup(self, stage):
        with self.trainer.init_module():
            self.l1 = torch.nn.Linear(self.in_features,3)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return torch.nn.functional.cross_entropy(y_hat, y)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(10,4),torch.randint(0, 2, (10,))), batch_size=2
    )

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.AdamW(self.parameters(), lr=1e-3)
        }


def main():
    model = TestModule(4)
    trainer = L.Trainer(accelerator="cpu",devices=2, max_epochs=10)    
    trainer.fit(model)


if __name__ == "__main__":
    main()