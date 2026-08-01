"""statsmodels adapter implementing :class:`EstimatorBackend`."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm


class StatsmodelsBackend:
    """Default OLS backend used by :func:`~doekit.assessment.analysis.ols.fit_linear_model`."""

    def fit_ols(self, y: np.ndarray, X: np.ndarray, cov_type: str = "nonrobust"):
        ols = sm.OLS(y, X)
        if cov_type == "nonrobust":
            return ols.fit()
        return ols.fit(cov_type=cov_type)
