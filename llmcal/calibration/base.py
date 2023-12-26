
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightning as L


class BaseCalibrator(nn.Module):
    """
    Base class for every calibrator. A calibrator object is a nn.Module
    that obtains the calibrated logposteriors from a feature vector.
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Number of input features of the calibrator
        self.num_features = kwargs["num_features"]

        # Number of output classes of the calibrator
        self.num_classes = kwargs["num_classes"]

        # Random state generator
        generator = torch.Generator(device="cpu")
        if kwargs["random_state"] is not None:
            generator = generator.manual_seed(kwargs["random_state"])
        self.generator = generator

    def forward(self, features):
        raise NotImplementedError
    
    def calibrate(self, features, batch_size=None, accelerator="cpu", num_devices=1):
        if batch_size is None:
            batch_size = features.shape[0]

        fabric = L.Fabric(accelerator=accelerator, devices=num_devices)
        fabric.launch()

        fake_labels = torch.zeros(features.shape[0], dtype=torch.long)
        dataset = TensorDataset(features, fake_labels)
        dataloader = fabric.setup_dataloaders(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )
        model = fabric.setup(self)
    
        model.eval()
        with torch.no_grad():
            logits = []
            for batch, _ in dataloader:
                batch_logits = model(batch)
                logits.append(batch_logits)
            logits = torch.cat(logits, dim=0)
        return torch.log_softmax(logits, dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_features={self.num_features}, num_classes={self.num_classes})"
    
