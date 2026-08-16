"""Surrogate protocol: a predictive mirror of :class:`FitResult`.

A :class:`Surrogate` answers ``predict(X) -> (mean, std)`` with a **calibrated**
``std`` — the epistemic uncertainty that drives Bayesian optimization. Unlike a
plain fit, its job is to say *how sure* it is at unseen points, so the
acquisition layer (:mod:`doekit.orchestration.optimize`) knows where to sample.

Two concrete backends live next to this module:

- :class:`~doekit.assessment.surrogate.ols.OLSSurrogate` — the polynomial OLS
  model doekit already fits, with the linear prediction-variance as ``std``.
  Default backend; **no new dependencies**.
- :class:`~doekit.assessment.surrogate.gp.GPSurrogate` — a Gaussian process whose
  *prior mean is the OLS surface*, so ``optimize`` is an extension of ``learn``
  (interpretable trend + non-parametric residual with calibrated ``sigma(x)``).
  Requires ``pip install "doekit[bo]"`` (scikit-learn).
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...domain.factors import ContinuousFactor, CategoricalFactor


@runtime_checkable
class Surrogate(Protocol):
    """Predictive model with calibrated uncertainty (larger ``std`` = less sure).

    Concrete surrogates additionally expose ``model`` (the polynomial trend),
    ``factors`` and ``factor_names`` so the optimize layer can build candidates.
    """

    def predict(self, X_new) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` for each row of ``X_new`` (Design/DataFrame/array)."""
        ...

    def calibration(self, levels: Tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
        """Leave-one-out coverage of prediction intervals (auditing the box)."""
        ...


# ---------------------------------------------------------------------------
# frame / feature helpers
# ---------------------------------------------------------------------------

def infer_factors(frame: pd.DataFrame) -> list:
    """Infer factors from a factor frame (continuous ranges / categoricals)."""
    facs: list = []
    for name in frame.columns:
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col):
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
            facs.append(ContinuousFactor(str(name), lo, hi))
        else:
            facs.append(CategoricalFactor(str(name), list(pd.unique(col))))
    return facs


def as_factor_frame(X, factor_names: list) -> pd.DataFrame:
    """Coerce Design / DataFrame / ndarray to a factor-column DataFrame."""
    if isinstance(X, Design):
        X = X.matrix
    if isinstance(X, pd.DataFrame):
        missing = [c for c in factor_names if c not in X.columns]
        if missing:
            raise ValueError(f"missing factor columns for prediction: {missing}")
        return X.loc[:, factor_names].reset_index(drop=True)
    arr = np.atleast_2d(np.asarray(X, dtype=float))
    if arr.shape[1] != len(factor_names):
        raise ValueError(
            f"array has {arr.shape[1]} columns; expected {len(factor_names)} "
            f"factors {factor_names}"
        )
    return pd.DataFrame(arr, columns=factor_names)


def encode_features(factors: list, frame: pd.DataFrame) -> np.ndarray:
    """Numeric feature matrix for a distance-based model (GP kernel input).

    Continuous / discrete factors are coded to ``[-1, 1]``; categoricals are
    one-hot expanded so the kernel treats levels as equidistant.
    """
    by_name = {getattr(f, "name", None): f for f in factors}
    cols: list[np.ndarray] = []
    for name in frame.columns:
        f = by_name.get(name)
        series = frame[name]
        if f is not None and getattr(f, "is_categorical", False):
            for lvl in list(f.levels):
                cols.append((series.to_numpy(dtype=object) == lvl).astype(float))
        elif f is not None and hasattr(f, "encode"):
            try:
                cols.append(np.asarray(f.encode(series.to_numpy()), dtype=float))
            except (ValueError, TypeError, KeyError):
                cols.append(series.to_numpy(dtype=float))
        elif pd.api.types.is_numeric_dtype(series):
            cols.append(series.to_numpy(dtype=float))
        else:  # unknown categorical without a factor: one-hot on observed levels
            for lvl in pd.unique(series):
                cols.append((series.to_numpy(dtype=object) == lvl).astype(float))
    if not cols:
        return np.empty((len(frame), 0), dtype=float)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def loo_calibration(fit_fn, frame: pd.DataFrame, y: np.ndarray,
                    levels: Tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
    """Leave-one-out interval coverage of a surrogate.

    Refits ``fit_fn(frame_train, y_train)`` for each held-out row and checks
    whether the observation falls inside each nominal interval. Well-calibrated
    ``sigma(x)`` gives coverage close to the nominal level.

    Parameters
    ----------
    fit_fn : callable
        ``fit_fn(frame, y) -> Surrogate`` (must expose ``predict``).
    frame : DataFrame
        Factor frame (n_runs x n_factors).
    y : ndarray
        Response vector.
    levels : tuple of float
        Nominal central-interval levels in ``(0, 1)``.

    Returns
    -------
    dict
        ``{"levels", "coverage" {level: frac}, "rmse_standardized", "n"}``.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    levels = tuple(float(v) for v in levels)
    result = {
        "levels": list(levels),
        "coverage": {lvl: float("nan") for lvl in levels},
        "rmse_standardized": float("nan"),
        "n": int(n),
    }
    if n < 4:
        return result  # LOO is meaningless on tiny samples

    z = {lvl: float(stats.norm.ppf(0.5 + lvl / 2.0)) for lvl in levels}
    covered = {lvl: 0 for lvl in levels}
    zscores: list[float] = []
    used = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            sur = fit_fn(frame.iloc[mask].reset_index(drop=True), y[mask])
            mu, sd = sur.predict(frame.iloc[[i]])
        except (ValueError, np.linalg.LinAlgError):
            continue
        mu_i, sd_i = float(np.asarray(mu).reshape(-1)[0]), float(np.asarray(sd).reshape(-1)[0])
        sd_eff = sd_i if sd_i > 1e-12 else 1e-12
        err = y[i] - mu_i
        zscores.append(err / sd_eff)
        for lvl in levels:
            if abs(err) <= z[lvl] * sd_eff:
                covered[lvl] += 1
        used += 1

    if used:
        result["coverage"] = {lvl: covered[lvl] / used for lvl in levels}
        result["rmse_standardized"] = float(np.sqrt(np.mean(np.square(zscores))))
        result["n"] = int(used)
    return result


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def resolve_surrogate_model(design_or_frame, model: Optional[Model],
                            factors: Optional[list]) -> Tuple[Model, list, pd.DataFrame]:
    """Resolve ``(model, factors, factor_frame)`` from flexible inputs."""
    if isinstance(design_or_frame, Design):
        frame = design_or_frame.matrix.reset_index(drop=True)
        factors = list(factors or design_or_frame.factors or [])
        model = model or design_or_frame.model
    else:
        frame = pd.DataFrame(design_or_frame).reset_index(drop=True)
        factors = list(factors or [])
    if not factors:
        factors = infer_factors(frame)
    names = [f.name for f in factors]
    frame = frame.loc[:, [c for c in names if c in frame.columns]] if names else frame
    if model is None:
        # Optimization implies curvature: default to a full quadratic surface.
        model = Model.full_quadratic(list(frame.columns))
    return model, factors, frame


def fit_surrogate(design, y, kind: str = "auto", model: Optional[Model] = None,
                  factors: Optional[list] = None, **kwargs) -> Surrogate:
    """Fit a surrogate to ``(design, y)``.

    Parameters
    ----------
    design : Design or DataFrame
        Executed design (factor levels).
    y : array-like
        Measured response.
    kind : {"auto", "ols", "gp"}, default "auto"
        ``"auto"`` picks ``"gp"`` when scikit-learn is installed, else ``"ols"``.
    model : Model, optional
        Polynomial trend / prior mean; defaults to the design model or a full
        quadratic surface.
    factors : list, optional
        Factor definitions (inferred from the frame when omitted).
    **kwargs
        Forwarded to the concrete surrogate (e.g. GP ``n_restarts``, ``seed``).

    Returns
    -------
    Surrogate
    """
    kind = kind.strip().lower()
    if kind == "auto":
        kind = "gp" if _sklearn_available() else "ols"
    if kind == "ols":
        from .ols import OLSSurrogate  # noqa: PLC0415
        return OLSSurrogate.fit(design, y, model=model, factors=factors)
    if kind == "gp":
        from .gp import GPSurrogate  # noqa: PLC0415
        return GPSurrogate.fit(design, y, model=model, factors=factors, **kwargs)
    raise ValueError(f"unknown surrogate kind {kind!r} (use 'auto', 'ols' or 'gp')")
