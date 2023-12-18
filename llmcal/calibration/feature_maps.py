
import torch
import torch.nn as nn

class BaseFeatureMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        raise NotImplementedError


class QuadraticFeatureMap(BaseFeatureMap):

    def forward(self, logits):
        quad_logits = logits.unsqueeze(-1)
        quad_logits = torch.bmm(quad_logits, quad_logits.transpose(-1, -2)).reshape(logits.shape[0], -1)
        return torch.cat([quad_logits, logits], dim=-1)


def apply_feature_map(logits, feature_map):
    if feature_map is None:
        feature_map = lambda x: x
    elif feature_map == "quadratic":
        feature_map = QuadraticFeatureMap()
    else:
        raise ValueError(f"Invalid feature map: {feature_map}")
    return feature_map(logits)