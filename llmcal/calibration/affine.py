
import torch
import torch.nn as nn

from .base import LBFGSBCalibrator
from .losses import LogLoss, BrierLoss


class AffineCalibrator(LBFGSBCalibrator):

    def __init__(self, num_features, num_classes, alpha="vector", bias=True, loss="log-loss", random_state=None):
        generator = torch.Generator(device="cpu")
        if random_state is not None:
            generator = generator.manual_seed(random_state)
        super().__init__(num_features=num_features, num_classes=num_classes, generator=generator)
            
        if alpha == "vector":
            if num_features != num_classes:
                raise ValueError(f"Cannot perform vector scaling when num_features != num_classes")
            self.alpha = nn.Parameter(torch.randn(num_classes, generator=self.generator) * 0.01)
            self._alpha = torch.diag(self.alpha)
        elif alpha == "scalar":
            if num_features != num_classes:
                raise ValueError(f"Cannot perform temperature scaling when num_features != num_classes")
            self.alpha = nn.Parameter(torch.randn(1, generator=self.generator) * 0.01)
            self._alpha = torch.eye(num_classes) * self.alpha
        elif alpha == "matrix":
            self.alpha = nn.Parameter(torch.randn(num_classes, num_features, generator=self.generator) * 0.01)
            self._alpha = self.alpha
        elif alpha == "None":
            if num_features != num_classes:
                raise ValueError(f"Alpha cannot be None when num_features != num_classes")
            self.alpha = torch.eye(num_classes)
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        if bias:
            self.bias = nn.Parameter(torch.randn(num_classes, generator=self.generator) * 0.01)
        else:
            self.bias = torch.zeros(num_classes)

        if loss == "log-loss":
            self.loss = LogLoss()
        elif loss == "brier":
            self.loss = BrierLoss()
        else:
            raise ValueError(f"Invalid loss: {loss}")

    def forward(self, features):
        return torch.log_softmax(features @ self._alpha.T + self.bias, dim=1)



        
