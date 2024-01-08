
import torch
from torch import nn
import torch.nn.functional as F

class LogLoss(nn.CrossEntropyLoss):
    pass

class BrierLoss(nn.Module):
    def forward(self, logits, labels):
        labels = F.one_hot(labels, num_classes=logits.shape[-1])
        return torch.mean((logits.softmax(dim=-1) - labels) ** 2)