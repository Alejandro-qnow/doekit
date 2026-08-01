"""OLS fitting."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...shared.serialize import jsonify as _jsonify, as_float_list as _as_float_list
from ...domain.criteria.linalg import leverage as _leverage_rows

from .results import FitResult
from .helpers import (_resolve_blocks, _resolve_model, _factor_frame,
                      _augment_blocks)
from .adapters.statsmodels_backend import StatsmodelsBackend

_VALID_COV = frozenset({"nonrobust", "HC0", "HC1", "HC3"})
_DEFAULT_BACKEND = StatsmodelsBackend()

# ---------------------------------------------------------------------------

def fit_linear_model(design: Design, response, model: Optional[Model] = None,
                     blocks=None, cov_type: str = "nonrobust",
                     report=None) -> FitResult:
    """Fit an OLS linear model to a design's responses.

    Parameters
    ----------
    design : Design
        The executed design providing the factor levels.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model to fit; taken from ``model`` or ``design.model`` if omitted.
        Block columns are excluded automatically from the default main-effects
        model.
    blocks : None, False, str or array-like, optional
        Fixed block factor: a column name in ``design.matrix``, a per-run label
        array, or ``None``. If ``None`` but ``design.metadata['blocking']``
        names a column, that column is used. Pass ``False`` to force an
        unblocked fit (the block column is still excluded from the factor model).
    cov_type : {"nonrobust", "HC0", "HC1", "HC3"}, default "nonrobust"
        Covariance estimator for standard errors (via statsmodels).
    report : None, bool, str, Path or dict, optional
        If not ``None``, a full HTML report is generated and its path is stored
        in ``fit.report_path``.

    Returns
    -------
    FitResult
        The fitted model with coefficients, statistics and anomaly diagnostics.

    Raises
    ------
    ValueError
        If ``cov_type`` is unknown, blocks are invalid, or the design is
        saturated / rank-deficient after adding blocks.
    """
    if cov_type not in _VALID_COV:
        raise ValueError(
            f"cov_type must be one of {sorted(_VALID_COV)}, got {cov_type!r}"
        )

    block_labels, block_name, drop = _resolve_blocks(design, blocks)
    model = _resolve_model(design, model, drop=drop)
    frame = _factor_frame(design, drop=drop)
    X = np.asarray(model.matrix(frame), dtype=float)
    names = list(model.column_names(frame))
    y = np.asarray(response, dtype=float).reshape(-1)

    if y.shape[0] != design.n_runs:
        raise ValueError(
            f"response length ({y.shape[0]}) must match n_runs ({design.n_runs})"
        )

    if block_labels is not None:
        X, names = _augment_blocks(X, names, block_labels)

    n, p = X.shape
    dof = n - p
    if dof < 0:
        raise ValueError(
            f"saturated/over-parameterized design after blocks: n={n}, p={p} "
            f"(need n >= p). Reduce the model or add runs."
        )
    if np.linalg.matrix_rank(X) < p:
        raise ValueError(
            "model matrix is rank deficient; check collinear terms or blocks."
        )

    res = _DEFAULT_BACKEND.fit_ols(y, X, cov_type=cov_type)

    beta = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    tvals = np.asarray(res.tvalues, dtype=float)
    pvals = np.asarray(res.pvalues, dtype=float)
    fitted = np.asarray(res.fittedvalues, dtype=float)
    resid = np.asarray(res.resid, dtype=float)
    sigma2 = float(res.mse_resid) if dof > 0 else float("nan")
    r2 = float(res.rsquared)
    r2_adj = float(res.rsquared_adj) if dof > 0 else float("nan")

    # Influence diagnostics from the classical (nonrobust) hat matrix
    XtX_inv = np.linalg.pinv(X.T @ X)
    leverage = _leverage_rows(X, XtX_inv)
    stud = np.full(n, np.nan)
    cooks = np.full(n, np.nan)
    if dof > 1 and np.isfinite(sigma2) and sigma2 > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            internal = resid / np.sqrt(sigma2 * np.clip(1 - leverage, 1e-12, None))
            denom = np.clip(dof - internal ** 2, 1e-12, None)
            stud = internal * np.sqrt((dof - 1) / denom)
            cooks = ((resid ** 2 / (p * sigma2))
                     * (leverage / np.clip((1 - leverage) ** 2, 1e-12, None)))

    fit = FitResult(
        names, beta, se, tvals, pvals, resid, sigma2, r2, dof,
        fitted=fitted, r_squared_adj=r2_adj, leverage=leverage,
        studentized_resid=stud, cooks_distance=cooks,
        cov_type=cov_type, blocks=block_name, _sm_result=res,
    )
    if report is not None:
        from ..._compat import maybe_report  # noqa: PLC0415
        fit.report_path = maybe_report(
            design, report=report, response=response, model=model,
        )
    return fit


