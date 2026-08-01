"""Ports (interfaces) for statistical estimation backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EstimatorBackend(Protocol):
    """Minimal OLS backend seam (statsmodels today; mockable in tests)."""

    def fit_ols(
        self,
        y: np.ndarray,
        X: np.ndarray,
        cov_type: str = "nonrobust",
    ) -> Any:
        """Return a result object with params, bse, tvalues, pvalues, etc."""
        ...
