
import torch
import torch.nn as nn

class BaseFeatureMap(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, logits):
        raise NotImplementedError


class IdentityFeatureMap(BaseFeatureMap):
    
        def __init__(self, num_features):
            super().__init__(num_features)
    
        def forward(self, logits):
            return logits


class QuadraticFeatureMap(BaseFeatureMap):

    def __init__(self, num_features):
        super().__init__(num_features * (num_features + 1))

    def forward(self, logits):
        quad_logits = logits.unsqueeze(-1)
        quad_logits = torch.bmm(quad_logits, quad_logits.transpose(-1, -2)).reshape(logits.shape[0], -1)
        return torch.cat([quad_logits, logits], dim=-1)


def init_feature_map(num_features, feature_map):
    if feature_map is None:
        feature_map = IdentityFeatureMap(num_features)
    elif feature_map == "quadratic":
        feature_map = QuadraticFeatureMap(num_features)
    else:
        raise ValueError(f"Invalid feature map: {feature_map}")
    return feature_map