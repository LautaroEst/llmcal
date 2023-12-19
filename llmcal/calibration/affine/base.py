
import torch
from torch import nn, optim
from .psrcal import AffineCalLogLoss, AffineCalBrier



class AffineCalibrator(nn.Module):

    MODELS = [
        "vector scaling", # Two parameters (scaling and shift) per class
        "bias only", # One parameter per class (shift)
    ]

    PSRS = [
        "log-loss",
        "brier"
    ]

    def __init__(
        self,
        num_features, 
        num_classes, 
        alpha="vector", 
        bias=True, 
        loss="log-loss", 
        random_state=None
    ):
        super().__init__()

        if alpha == "vector" and bias and loss == "log-loss":
            calmodel = AffineCalLogLoss(num_classes, bias=True, scale=True)
        elif alpha == "vector" and bias and loss == "brier":
            calmodel = AffineCalBrier(num_classes, bias=True, scale=True)
        elif alpha == "None" and bias and loss == "log-loss":
            calmodel = AffineCalLogLoss(num_classes, bias=True, scale=False)
        elif alpha == "None" and bias and loss == "brier":
            calmodel = AffineCalBrier(num_classes, bias=True, scale=False)
        else:
            raise ValueError(f"Calibration method not supported.")
        self.calmodel = calmodel
        
    def forward(self, features):
        scores = torch.log_softmax(features, dim=1)
        cal_scores = self.calmodel.calibrate(scores)
        cal_logprobs = torch.log_softmax(cal_scores, dim=1)
        return cal_logprobs

    def fit(self, features, labels, **kwargs):
        scores = torch.log_softmax(features, dim=1)
        self.calmodel.train(scores, labels)

    def calibrate(self, logits):
        with torch.no_grad():
            cal_logits = self(logits)
        return cal_logits