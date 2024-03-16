
from typing import Literal
from torch import nn
import torch
import lightning as L

class AffineCalibrator(nn.Module):
    """
    Affine calibration block. It is a linear block that performs an affine transformation
    of the input feature vector ino order to output the calibrated logits.

    Parameters
    ----------
    num_features : int
        Number of input features of the calibrator.
    num_classes : int
        Number of output classes of the calibrator.
    alpha : {"vector", "scalar", "matrix", "none"}, optional
        Type of affine transformation, by default "vector"
    beta : bool, optional
        Whether to use a beta term, by default True
    """    
    def __init__(
        self, 
        num_classes: int, 
        alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
        beta: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.additional_arguments = {
            "alpha": alpha,
            "beta": beta,
        }

        # Set the alpha parameter
        if alpha == "matrix":
            self.alpha = nn.Parameter(torch.zeros(num_classes, num_classes))
        elif alpha == "vector":
            self.alpha = nn.Parameter(torch.zeros(num_classes))
        elif alpha == "scalar":
            self.alpha = nn.Parameter(torch.tensor(0))
        elif alpha == "none":
            self.alpha = None
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        # Set the beta parameter
        self.beta = nn.Parameter(torch.zeros(num_classes)) if beta else None

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        num_classes = self.num_classes
        beta = self.beta

        if alpha is None:
            output = logits
        elif alpha.shape == (num_classes, num_classes):
            output = alpha @ logits
        elif alpha.shape in [(num_classes,), ()]:
            output = alpha * logits

        if beta is not None:
            output += beta

        return {"calibrated_logits": output}
    
    def init_params(self, fabric: L.Fabric):
        if self.alpha is not None:
            self.alpha.data.fill_(1.)
        if self.beta is not None:
            self.beta.data.fill_(0.)
        self = fabric.setup_module(self)
        return self