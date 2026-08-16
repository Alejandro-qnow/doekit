"""Gaussian-process surrogate with the OLS surface as its prior mean.

The GP does **not** replace the interpretable polynomial — it *corrects* it: the
OLS trend carries the story (main effects, curvature), while the GP models the
non-parametric residual and, crucially, a calibrated ``sigma(x)`` that grows in
unexplored regions. So ``optimize`` is a strict extension of ``learn``.

Kernel: ``ConstantKernel * RBF + WhiteKernel`` (signal amplitude, smoothness,
and an explicit noise/nugget term). Requires ``pip install "doekit[bo]"``.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...domain.model import Model
from .base import (as_factor_frame, encode_features, loo_calibration,
                   resolve_surrogate_model)
from .ols import OLSSurrogate


def _require_sklearn_gp():
    """Import scikit-learn GP pieces or raise a helpful ImportError."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: PLC0415
        from sklearn.gaussian_process.kernels import (  # noqa: PLC0415
            ConstantKernel, RBF, WhiteKernel,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for GPSurrogate. Install with "
            "'pip install \"doekit[bo]\"'."
        ) from exc
    return GaussianProcessRegressor, ConstantKernel, RBF, WhiteKernel


class GPSurrogate:
    """GP surrogate over the OLS residual (interpretable trend + calibrated sigma).

    The prior mean is the OLS polynomial surface; a GP models the residual with
    ``ConstantKernel * RBF + WhiteKernel``. Total predictive uncertainty combines
    trend leverage, GP posterior variance and observation noise.

    Formulas
    --------
    - **Mean:** ``mu(x) = mu_OLS(x) + mu_GP(x)`` on encoded features ``z(x)``.
    - **Variance:** ``Var(y|x) = Var_OLS(mu(x)) + Var_GP(r|z) + noise``;
      ``std = sqrt(Var)``.
    """

    def __init__(self, ols: OLSSurrogate, gpr, feat_mean: np.ndarray,
                 feat_scale: np.ndarray, factors: list, factor_names: list,
                 frame: pd.DataFrame, y: np.ndarray):
        self._ols = ols
        self._gpr = gpr
        self._feat_mean = feat_mean
        self._feat_scale = feat_scale
        self.model = ols.model
        self.factors = factors
        self.factor_names = list(factor_names)
        self._frame = frame
        self._y = np.asarray(y, dtype=float).reshape(-1)
        # Observation-noise variance (WhiteKernel) added to the latent posterior
        # variance so predictive intervals cover *observations* (honest LOO
        # calibration) while still growing away from the data.
        try:
            self._noise = float(gpr.kernel_.get_params().get("k2__noise_level", 0.0))
        except (AttributeError, ValueError):
            self._noise = 0.0

    # -- construction --------------------------------------------------------
    @classmethod
    def fit(cls, design, y, model: Optional[Model] = None,
            factors: Optional[list] = None, prior: str = "ols",
            n_restarts: int = 5, alpha: float = 1e-10,
            seed: Optional[int] = None) -> "GPSurrogate":
        """Fit a GP surrogate with an OLS (or constant) prior mean.

        Fits :class:`OLSSurrogate` for the trend, then a GP on the residual
        ``y - trend`` in encoded feature space.

        Formulas
        --------
        Residual GP: ``r ~ GP(0, k)`` with
        ``k = sigma_f^2 * RBF(length_scale) + WhiteKernel(noise)``.
        Hyper-parameters are optimized by marginal likelihood (with restarts).

        Parameters
        ----------
        design, y
            Executed design and measured response.
        model : Model, optional
            Prior-mean polynomial (defaults to the design model / full quadratic).
        factors : list, optional
            Factor definitions (inferred when omitted).
        prior : {"ols", "mean"}, default "ols"
            Prior mean: the OLS surface, or a flat mean of ``y``.
        n_restarts : int, default 5
            GP hyper-parameter optimizer restarts.
        alpha : float, default 1e-10
            Jitter added to the kernel diagonal for numerical stability.
        seed : int, optional
            RNG seed for the optimizer restarts.

        Returns
        -------
        GPSurrogate

        Raises
        ------
        ValueError
            If ``y`` length does not match the design rows.
        ImportError
            If scikit-learn is not installed (``pip install "doekit[bo]"``).
        """
        GaussianProcessRegressor, ConstantKernel, RBF, WhiteKernel = _require_sklearn_gp()
        model, factors, frame = resolve_surrogate_model(design, model, factors)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != len(frame):
            raise ValueError(
                f"response length ({y.shape[0]}) must match n_runs ({len(frame)})"
            )

        ols = OLSSurrogate.fit(frame, y, model=model, factors=factors)
        if prior == "mean":
            trend = np.full_like(y, float(np.mean(y)))
        else:
            trend, _ = ols.predict(frame)
        resid = y - trend

        feats = encode_features(factors, frame)
        feat_mean = feats.mean(axis=0) if feats.shape[1] else np.zeros(0)
        feat_scale = feats.std(axis=0) if feats.shape[1] else np.zeros(0)
        feat_scale = np.where(feat_scale > 1e-12, feat_scale, 1.0)
        Z = (feats - feat_mean) / feat_scale if feats.shape[1] else feats

        resid_scale = float(np.std(resid)) or 1.0
        n_dim = max(1, Z.shape[1])
        kernel = (
            ConstantKernel(resid_scale ** 2, (1e-5, 1e5))
            * RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=max(ols.sigma2, 1e-6),
                          noise_level_bounds=(1e-8, 1e2))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=False,
            n_restarts_optimizer=int(n_restarts), random_state=seed,
        )
        # Refitting a GP many times (LOO, constant-liar batches) routinely nudges
        # a hyper-parameter to its bound; that is expected, not user-actionable.
        try:
            from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            ConvergenceWarning = Warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gpr.fit(Z if Z.shape[1] else np.zeros((len(frame), 1)), resid)
        if not Z.shape[1]:
            feat_mean = np.zeros(1)
            feat_scale = np.ones(1)
        return cls(ols, gpr, feat_mean, feat_scale, factors,
                   list(frame.columns), frame, y)

    # -- prediction ----------------------------------------------------------
    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        feats = encode_features(self.factors, frame)
        if not feats.shape[1]:
            return np.zeros((len(frame), 1))
        return (feats - self._feat_mean) / self._feat_scale

    def predict(self, X_new) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` = OLS trend + GP residual, with total sigma.

        Predictive variance combines three sources: OLS trend estimation SE
        (leverage), GP residual posterior variance, and observation noise.
        Treating the estimated trend as fixed would understate uncertainty.

        Formulas
        --------
        ``mu = mu_OLS + mu_GP``,
        ``std = sqrt(std_OLS^2 + std_GP^2 + noise)``.

        Parameters
        ----------
        X_new : Design, DataFrame or array-like
            Points to predict at.

        Returns
        -------
        mean, std : ndarray
            Length ``n_rows`` each.
        """
        frame = as_factor_frame(X_new, self.factor_names)
        trend, trend_std = self._ols.predict(frame)
        Z = self._encode(frame)
        resid_mean, resid_std = self._gpr.predict(Z, return_std=True)
        var = (np.asarray(trend_std, dtype=float) ** 2
               + np.asarray(resid_std, dtype=float) ** 2 + self._noise)
        return trend + resid_mean, np.sqrt(var)

    def calibration(self, levels: Tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
        """Leave-one-out interval coverage (auditing the black box)."""
        model, factors = self.model, self.factors

        def _fit(frame, y):
            return GPSurrogate.fit(frame, y, model=model, factors=factors,
                                   n_restarts=0)

        return loo_calibration(_fit, self._frame, self._y, levels=levels)

    def __repr__(self) -> str:
        return f"GPSurrogate(kernel={self._gpr.kernel_}, n={len(self._y)})"
