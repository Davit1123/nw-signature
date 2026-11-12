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

from .learned_distance import (
    NWLearnedDistance,
    NWLearnedDistanceRegressor,
    NWLearnedDistanceClassifier
)

from .sliding_window import (
    window_edges,
    sliding_window_signatures,
    windowed_signature_features
)

from .window_tree import WindowTree

from .metrics import (
    MSE,
    supremum_metric,
    euclidean_metric,
    trunc_sig_metric
)

from .FBM_package import FBM, fbm, fgn, times

from .p_var_vectorized import p_variation

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
    "NWLearnedDistance",
    "NWLearnedDistanceRegressor",
    "NWLearnedDistanceClassifier",
    "window_edges",
    "sliding_window_signatures",
    "windowed_signature_features",
    "WindowTree",
    "MSE",
    "supremum_metric",
    "euclidean_metric",
    "trunc_sig_metric",
    "FBM",
    "fbm",
    "fgn",
    "times",
    "p_variation",
    "fBm_sampling",
    "augment_with_time",
    "suggest_truncation_level"
]
