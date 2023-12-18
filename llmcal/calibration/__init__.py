from .affine import AffineCalibrator
from .priors import PriorsAdaptator
from .mahalanobis import (
    QDACalibrator,
    LDACalibrator,
    DiscriminativeMahalanobisCalibrator
)
from .feature_maps import apply_feature_map