
import torch
import torch.nn as nn

from .base import BaseCalibrator
from .losses import LogLoss, BrierLoss


class AffineCalibrator(BaseCalibrator):

    def __init__(self, num_classes, alpha="vector", bias=True, loss="log-loss"):
        super().__init__()
        if alpha == "vector":
            self.alpha = nn.Parameter(torch.ones(num_classes))
        elif alpha == "scalar":
            self.alpha = nn.Parameter(torch.ones(1))
        elif alpha == "matrix":
            self.alpha = nn.Parameter(torch.eye(num_classes))
        elif alpha == "none":
            self.alpha = None
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.bias = None

        if loss == "log-loss":
            self.loss = LogLoss()
        elif loss == "brier":
            self.loss = BrierLoss()
        else:
            raise ValueError(f"Invalid loss: {loss}")

    def forward(self, logits):
        if self.alpha is None:
            if self.bias is None:
                return logits
            else:
                return logits + self.bias
        else:
            