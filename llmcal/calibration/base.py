
import torch
import torch.nn as nn


class BaseCalibrator(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits):
        raise NotImplementedError
    
    def fit(self, logits, labels=None):
        raise NotImplementedError

    def calibrate(self, logits):
        with torch.no_grad():
            return self(logits)
    
