from .base import BaseCalibrator
from ..optim import LBFGSBMixin, SGDMixin
from ..feature_maps import init_feature_map
from .mahalanobis import MahalanobisCalibrator, QDACalibrator, LDACalibrator
from .affine import AffineCalibrator


class AffineCalibratorWithFeatureMap(BaseCalibrator, LBFGSBMixin):

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



class MahalanobisCalibratorWithFeatureMap(BaseCalibrator, SGDMixin):

    def __init__(
        self, 
        feature_map: str,
        **kwargs
    ):
        super().__init__(num_features=kwargs["num_features"], num_classes=kwargs["num_classes"], random_state=kwargs["random_state"])
        self.feature_map = init_feature_map(kwargs["num_features"], feature_map)
        kwargs["num_features"] = self.feature_map.num_features
        self.calibrator = MahalanobisCalibrator(**kwargs)

    def forward(self, features):
        logits = self.calibrator(self.feature_map(features))
        return logits

    def loss(self, logits, labels):
        loss = self.calibrator.loss(logits, labels)
        return loss


class QDACalibratorWithFeatureMap(BaseCalibrator):
    
    def __init__(
        self, 
        feature_map: str,
        **kwargs
    ):
        super().__init__(num_features=kwargs["num_features"], num_classes=kwargs["num_classes"], random_state=kwargs["random_state"])
        self.feature_map = init_feature_map(kwargs["num_features"], feature_map)
        kwargs["num_features"] = self.feature_map.num_features
        self.calibrator = QDACalibrator(**kwargs)

    def forward(self, features):
        logits = self.calibrator(self.feature_map(features))
        return logits


class LDACalibratorWithFeatureMap(BaseCalibrator):
    
    def __init__(
        self, 
        feature_map: str,
        **kwargs
    ):
        super().__init__(num_features=kwargs["num_features"], num_classes=kwargs["num_classes"], random_state=kwargs["random_state"])
        self.feature_map = init_feature_map(kwargs["num_features"], feature_map)
        kwargs["num_features"] = self.feature_map.num_features
        self.calibrator = LDACalibrator(**kwargs)

    def forward(self, features):
        logits = self.calibrator(self.feature_map(features))
        return logits