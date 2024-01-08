
from typing import Literal, Optional
import torch
import torch.nn as nn

from .base import BaseCalibrator
from ..feature_maps import init_feature_map
from ..losses import LogLoss, BrierLoss
from ..optim import LBFGSMixin


class AffineCalibrator(BaseCalibrator, LBFGSMixin):
    """
    Affine calibrator. It is a linear calibrator that performs an affine transformation
    of the input feature vector.

    Parameters
    ----------
    num_features : int
        Number of input features of the calibrator.
    num_classes : int
        Number of output classes of the calibrator.
    alpha : {"vector", "scalar", "matrix", "none"}, optional
        Type of affine transformation, by default "vector"
    bias : bool, optional
        Whether to use a bias term, by default True
    loss : {"log-loss", "brier"}, optional
        Loss function to use, by default "log-loss"
    random_state : int, optional
        Random state generator, by default None
    """
    def __init__(
        self, 
        num_features: int, 
        num_classes: int, 
        alpha: Literal["vector", "scalar", "matrix", "none"] = "vector",
        bias: bool = True, 
        loss: Literal["log-loss", "brier"] = "log-loss",
        random_state: Optional[int] = None
    ):
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.additional_arguments = {
            "alpha": alpha,
            "bias": bias,
            "loss": loss
        }

        # Set the alpha parameter
        if alpha == "vector":
            if num_features != num_classes:
                raise ValueError(f"Cannot perform vector scaling when num_features != num_classes")
            self.alpha = nn.Parameter(torch.randn(num_classes, generator=self.generator) * 0.01)
        elif alpha == "scalar":
            if num_features != num_classes:
                raise ValueError(f"Cannot perform temperature scaling when num_features != num_classes")
            self.alpha = nn.Parameter(torch.randn(1, generator=self.generator) * 0.01)
        elif alpha == "matrix":
            self.alpha = nn.Parameter(torch.randn(num_classes, num_features, generator=self.generator) * 0.01)
        elif alpha == "none":
            if num_features != num_classes:
                raise ValueError(f"Alpha cannot be None when num_features != num_classes")
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        # Set the bias parameter
        if bias:
            self.bias = nn.Parameter(torch.randn(num_classes, generator=self.generator) * 0.01)
        else:
            self.bias = torch.zeros(num_classes)

        # Set the loss function
        if loss == "log-loss":
            self.loss = LogLoss()
        elif loss == "brier":
            self.loss = BrierLoss()
        else:
            raise ValueError(f"Invalid loss: {loss}")
        
    def _get_alpha(self, device):
        if self.additional_arguments["alpha"] == "vector":
            return torch.diag(self.alpha)
        elif self.additional_arguments["alpha"] == "scalar":
            return torch.eye(self.num_classes, device=device) * self.alpha
        elif self.additional_arguments["alpha"] == "matrix":
            return self.alpha
        elif self.additional_arguments["alpha"] == "none":
            return torch.eye(self.num_classes, device=device)

    def forward(self, features):
        alpha = self._get_alpha(features.device)
        logits = features @ alpha.T + self.bias
        return logits
    
        
class AffineCalibratorWithFeatureMap(BaseCalibrator, LBFGSMixin):

    def __init__(
        self, 
        feature_map: str, 
        **kwargs
    ):
        super().__init__(num_features=kwargs["num_features"], num_classes=kwargs["num_classes"], random_state=kwargs["random_state"])
        self.feature_map = init_feature_map(kwargs["num_features"], feature_map)
        kwargs["num_features"] = self.feature_map.num_features
        self.calibrator = AffineCalibrator(**kwargs)

    def forward(self, features):
        return self.calibrator(self.feature_map(features))

    def loss(self, logits, labels):
        return self.calibrator.loss(logits, labels)
    