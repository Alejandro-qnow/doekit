"""OLS surrogate: the polynomial model doekit already fits, with prediction sigma.

``std(x0) = sqrt(sigma2 * x0' (X'X)^-1 x0)`` is the standard error of the mean
prediction — it *grows away from the data* (high leverage), which is exactly the
epistemic signal a Bayesian optimizer needs. No new dependencies: this is the
default backend so ``optimize`` works without ``doekit[bo]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...domain.model import Model
from .base import as_factor_frame, loo_calibration, resolve_surrogate_model


class OLSSurrogate:
    """Linear (polynomial) surrogate with leverage-based predictive variance.

    Uses the same OLS surface as analysis; ``std(x)`` is the standard error of
    the mean prediction and grows with leverage away from the design points.

    Formulas
    --------
    - **Mean:** ``mu(x0) = x0' @ beta``.
    - **Variance:** ``Var(mu(x0)) = sigma2 * x0' (X'X)^-1 x0``,
      ``std(x0) = sqrt(Var)``.
    """

    def __init__(self, model: Model, factors: list, frame: pd.DataFrame,
                 y: np.ndarray, coef: np.ndarray, xtx_inv: np.ndarray,
                 sigma2: float, dof: int):
        self.model = model
        self.factors = factors
        self.factor_names = list(frame.columns)
        self._frame = frame
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self.coef = coef
        self._xtx_inv = xtx_inv
        self.sigma2 = float(sigma2)
        self.dof = int(dof)

    # -- construction --------------------------------------------------------
    @classmethod
    def fit(cls, design, y, model: Optional[Model] = None,
            factors: Optional[list] = None) -> "OLSSurrogate":
        """Fit an OLS surrogate to ``(design, y)``.

        Parameters
        ----------
        design : Design or DataFrame
            Executed design (factor levels).
        y : array-like
            Measured response.
        model : Model, optional
            Polynomial model matrix; defaults to full quadratic on the frame.
        factors : list, optional
            Factor definitions (inferred when omitted).

        Returns
        -------
        OLSSurrogate

        Raises
        ------
        ValueError
            If ``y`` length does not match the design rows.
        """
        model, factors, frame = resolve_surrogate_model(design, model, factors)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != len(frame):
            raise ValueError(
                f"response length ({y.shape[0]}) must match n_runs ({len(frame)})"
            )
        X = np.asarray(model.matrix(frame), dtype=float)
        n, p = X.shape
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        rank = int(np.linalg.matrix_rank(X))
        dof = n - rank
        rss = float(resid @ resid)
        # Fall back to a small positive variance on saturated designs so the
        # surrogate still yields a usable (if optimistic) sigma(x).
        if dof > 0:
            sigma2 = rss / dof
        else:
            spread = float(np.var(y)) if n > 1 else 1.0
            sigma2 = max(spread, 1e-8)
        xtx_inv = np.linalg.pinv(X.T @ X)
        return cls(model, factors, frame, y, coef, xtx_inv, sigma2, dof)

    # -- prediction ----------------------------------------------------------
    def predict(self, X_new) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` where ``std`` is the SE of the mean prediction.

        Formulas
        --------
        ``std_i = sqrt(sigma2 * x_i' (X'X)^-1 x_i)`` for each row ``x_i`` of the
        model matrix at ``X_new``.

        Parameters
        ----------
        X_new : Design, DataFrame or array-like
            Points to predict at (factor column order preserved).

        Returns
        -------
        mean, std : ndarray
            Length ``n_rows`` each; ``std >= 0``.
        """
        frame = as_factor_frame(X_new, self.factor_names)
        Xm = np.asarray(self.model.matrix(frame), dtype=float)
        mean = Xm @ self.coef
        # var(x0) = sigma2 * x0' (X'X)^-1 x0, per-row via einsum
        quad = np.einsum("ij,jk,ik->i", Xm, self._xtx_inv, Xm)
        var = self.sigma2 * np.clip(quad, 0.0, None)
        std = np.sqrt(var)
        return mean, std

    def calibration(self, levels: Tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
        """Leave-one-out interval coverage (auditing the linear box)."""
        model, factors = self.model, self.factors

        def _fit(frame, y):
            return OLSSurrogate.fit(frame, y, model=model, factors=factors)

        return loo_calibration(_fit, self._frame, self._y, levels=levels)

    def __repr__(self) -> str:
        return (f"OLSSurrogate(terms={len(self.coef)}, dof={self.dof}, "
                f"sigma={np.sqrt(self.sigma2):.4g})")
