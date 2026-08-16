"""Surrogate models: predictive mirror of the analysis layer for optimization.

    from doekit.assessment.surrogate import fit_surrogate
    sur = fit_surrogate(design, y)          # GP if sklearn else OLS
    mean, std = sur.predict(candidates)
    sur.calibration()                       # LOO interval coverage
"""

from .base import (Surrogate, fit_surrogate, loo_calibration,
                   encode_features, infer_factors)
from .ols import OLSSurrogate

__all__ = [
    "Surrogate",
    "OLSSurrogate",
    "GPSurrogate",
    "fit_surrogate",
    "loo_calibration",
    "encode_features",
    "infer_factors",
]


def __getattr__(name):
    # Lazy: GPSurrogate pulls scikit-learn only when actually referenced.
    if name == "GPSurrogate":
        from .gp import GPSurrogate  # noqa: PLC0415
        return GPSurrogate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
