from .affine import AffineCalibrator, AffineCalibratorWithFeatureMap
from .mahalanobis import (
    QDACalibrator,
    LDACalibrator,
    DiscriminativeMahalanobisCalibrator,
    QDACalibratorWithFeatureMap,
    LDACalibratorWithFeatureMap,
    DiscriminativeMahalanobisCalibratorWithFeatureMap,
)
from .priors import PriorsAdaptator