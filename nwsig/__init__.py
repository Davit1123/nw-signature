"""
Nadaraya-Watson Signature Estimator
====================================

A non-parametric regression and classification method using path signatures.
"""

from .Robnwsignature import (
    NadarayaWatsonSignature,
    NWSignatureRegressor,
    NWSignatureClassifier
)

from .metrics import (
    MSE,
    supremum_metric,
    euclidean_metric,
    trunc_sig_metric
)

from .FBM_package import FBM, fbm, fgn, times

from .utils import (
    fBm_sampling,
    augment_with_time,
    suggest_truncation_level
)

__version__ = "1.0.0"

__all__ = [
    "NadarayaWatsonSignature",
    "NWSignatureRegressor",
    "NWSignatureClassifier",
    "MSE",
    "supremum_metric",
    "euclidean_metric",
    "trunc_sig_metric",
    "FBM",
    "fbm",
    "fgn",
    "times",
    "fBm_sampling",
    "augment_with_time",
    "suggest_truncation_level"
]
