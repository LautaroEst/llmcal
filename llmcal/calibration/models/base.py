from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L


class BaseCalibrator(nn.Module):
    """
    Base class for every calibrator. A calibrator object is a nn.Module
    that obtains the calibrated logposteriors from a feature vector.

    Parameters
    ----------
    num_features : int
        Number of input features of the calibrator.
    num_classes : int
        Number of output classes of the calibrator.
    random_state : int, optional
        Random state generator, by default None
    """
    def __init__(self, num_features: int, num_classes: int, random_state: int = None):
        super().__init__()
        
        # Number of input features of the calibrator
        self.num_features = num_features

        # Number of output classes of the calibrator
        self.num_classes = num_classes

        # Random state generator
        generator = torch.Generator(device="cpu")
        if random_state is not None:
            generator = generator.manual_seed(random_state)
        self.generator = generator

    def forward(self, features: torch.Tensor):
        """
        Run the calibrator on the input feature vector.

        Parameters
        ----------
        features : torch.Tensor(shape=(num_samples, num_features))
            Input feature vector.

        Returns
        -------
        torch.Tensor(shape=(num_samples, num_classes))
            Calibrated logits.
        """
        raise NotImplementedError
    
    def calibrate(self, features: torch.Tensor, batch_size: int = None, **kwargs):
        """
        Calibrate the logits of the input feature vector.

        Parameters
        ----------
        features : torch.Tensor(shape=(num_samples, num_features))
            Input feature vector.
        batch_size : int, optional
            Batch size to use, by default None
        **kwargs : dict
            Fabric initialization arguments. Check https://lightning.ai/docs/fabric/stable/api/fabric_args.html for more information.

        Returns
        -------
        torch.Tensor(shape=(num_samples, num_classes))
            Calibrated logposteriors.
        """
        if batch_size is None:
            batch_size = features.shape[0] # Calibrate all samples at once

        # Initialize the accelerator
        fabric = L.Fabric(**kwargs)
        fabric.launch()

        # Prepare the dataloader
        dataset = _FeaturesDataset(features)
        dataloader = fabric.setup_dataloaders(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )

        # Prepare the model
        model = fabric.setup(self)
        model.eval()

        # Calibrate the logits
        with torch.no_grad():
            logits = []
            for batch in dataloader:
                batch_logits = model(batch)
                logits.append(batch_logits)
            logits = torch.cat(logits, dim=0)

        return torch.log_softmax(logits, dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_features={self.num_features}, num_classes={self.num_classes}, random_state={self.generator.initial_seed()})"
    
    

class _FeaturesDataset(Dataset):
    """
    Dummy dataset to use in the calibrator calibration.
    """
    def __init__(self, features: torch.Tensor):
        self.features = features
    
    def __getitem__(self, index):
        return self.features[index]
    
    def __len__(self):
        return self.features.shape[0]