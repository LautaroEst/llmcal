from .affine import AffineCalibrator
from .priors import PriorsAdaptator
from .mahalanobis import (
    QDACalibrator,
    LDACalibrator,
    DiscriminativeMahalanobisCalibrator
)
from .feature_maps import init_feature_map