
from collections import OrderedDict
import torch
import torch.nn as nn

from .base import LBFGSBCalibrator
from .losses import LogLoss, BrierLoss


class AffineCalibrator(LBFGSBCalibrator):

    def __init__(self, num_features, num_classes, alpha="vector", bias=True, loss="log-loss", random_state=None):
        self.arguments = OrderedDict([
            ("num_features", num_features), 
            ("num_classes", num_classes), 
            ("alpha", alpha), 
            ("bias", bias), 
            ("loss", loss), 
            ("random_state", random_state)
        ])
        super().__init__(**self.arguments)

        self._alpha_shape = alpha
        if self._alpha_shape == "vector":
            if num_features != num_classes:
                raise ValueError(f"Cannot perform vector scaling when num_features != num_classes")
            self.alpha = nn.Parameter(torch.randn(num_classes, generator=self.generator) * 0.01)
        elif self._alpha_shape == "scalar":
            if num_features != num_classes:
                raise ValueError(f"Cannot perform temperature scaling when num_features != num_classes")
            self.alpha = nn.Parameter(torch.randn(1, generator=self.generator) * 0.01)
        elif self._alpha_shape == "matrix":
            self.alpha = nn.Parameter(torch.randn(num_classes, num_features, generator=self.generator) * 0.01)
        elif self._alpha_shape == "None":
            if num_features != num_classes:
                raise ValueError(f"Alpha cannot be None when num_features != num_classes")
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        self._bias_shape = bias
        if bias:
            self.bias = nn.Parameter(torch.randn(num_classes, generator=self.generator) * 0.01)
        else:
            self.bias = torch.zeros(num_classes)

        self._loss_type = loss
        if loss == "log-loss":
            self.loss = LogLoss()
        elif loss == "brier":
            self.loss = BrierLoss()
        else:
            raise ValueError(f"Invalid loss: {loss}")
        
    def _get_alpha(self, device):
        if self._alpha_shape == "vector":
            return torch.diag(self.alpha)
        elif self._alpha_shape == "scalar":
            return torch.eye(self.num_classes, device=device) * self.alpha
        elif self._alpha_shape == "matrix":
            return self.alpha
        elif self._alpha_shape == "None":
            return torch.eye(self.num_classes, device=device)

    def forward(self, features):
        alpha = self._get_alpha(features.device)
        logits = features @ alpha.T + self.bias
        return logits



        
