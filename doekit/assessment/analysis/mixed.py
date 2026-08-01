"""Linear mixed models (MixedLM)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...shared.serialize import jsonify as _jsonify, as_float_list as _as_float_list
from ...domain.criteria.linalg import leverage as _leverage_rows

from statsmodels.regression.mixed_linear_model import MixedLM

from .results import MixedFitResult
from .helpers import _blocking_column, _resolve_model, _factor_frame

# ---------------------------------------------------------------------------

def fit_mixed_model(design: Design, response, groups,
                    model: Optional[Model] = None,
                    re_formula: str = "1",
                    method: str = "reml",
                    reml: Optional[bool] = None) -> MixedFitResult:
    """Fit a linear mixed model (random intercept / slopes via MixedLM).

    Parameters
    ----------
    design : Design
        Executed design.
    response : array-like
        Measured response per run.
    groups : str or array-like
        Grouping factor for random effects (column name or per-run labels).
        Typical use: batch, whole-plot, or hard-to-change factor.
    model : Model, optional
        Fixed-effects model; defaults as in :func:`fit_linear_model`.
    re_formula : str, default "1"
        Random-effects formula in statsmodels style (``"1"`` = random intercept).
    method : {"reml", "ml"}, default "reml"
        Estimation method (ignored if ``reml`` is given explicitly).
    reml : bool, optional
        If set, overrides ``method`` (``True`` -> REML, ``False`` -> ML).

    Returns
    -------
    MixedFitResult
        Fixed effects, variance components and fit diagnostics.

    Raises
    ------
    ValueError
        If groups are invalid (fewer than 2 levels, length mismatch, etc.).
    """
    if isinstance(groups, str):
        if groups not in design.matrix.columns:
            raise ValueError(
                f"groups column {groups!r} not found in design.matrix columns "
                f"{list(design.matrix.columns)}"
            )
        group_labels = design.matrix[groups].to_numpy()
        group_name = groups
        drop = [groups]
    else:
        group_labels = np.asarray(groups)
        if group_labels.shape[0] != design.n_runs:
            raise ValueError(
                f"groups array length ({group_labels.shape[0]}) must match "
                f"n_runs ({design.n_runs})"
            )
        group_name = "array"
        drop = []

    # also drop block column from fixed effects if present as metadata
    blk = _blocking_column(design)
    if blk and blk not in drop:
        drop = drop + [blk]

    n_groups = int(len(pd.unique(pd.Series(group_labels))))
    if n_groups < 2:
        raise ValueError(
            f"groups must have at least 2 distinct levels (got {n_groups})"
        )

    model = _resolve_model(design, model, drop=drop)
    frame = _factor_frame(design, drop=drop)
    X = np.asarray(model.matrix(frame), dtype=float)
    names = list(model.column_names(frame))
    y = np.asarray(response, dtype=float).reshape(-1)

    if y.shape[0] != design.n_runs:
        raise ValueError(
            f"response length ({y.shape[0]}) must match n_runs ({design.n_runs})"
        )
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"not enough runs for mixed model fixed effects: n={X.shape[0]}, "
            f"p={X.shape[1]}"
        )

    use_reml = reml if reml is not None else (method.lower() == "reml")
    method_label = "reml" if use_reml else "ml"

    exog_re = None
    if re_formula.strip() not in ("1", "1.0"):
        from patsy import dmatrix  # noqa: PLC0415
        exog_re = np.asarray(dmatrix(re_formula, frame), dtype=float)

    md = MixedLM(endog=y, exog=X, groups=group_labels, exog_re=exog_re)
    try:
        res = md.fit(reml=use_reml, method="lbfgs", maxiter=200, disp=False)
    except Exception:
        res = md.fit(reml=use_reml, disp=False)

    fe = np.asarray(res.fe_params, dtype=float)
    # bse_fe may be unavailable if singular; fall back to nan
    try:
        se = np.asarray(res.bse_fe, dtype=float)
    except Exception:
        se = np.full(len(fe), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        zvals = fe / se
        pvals = 2 * stats.norm.sf(np.abs(zvals))

    fitted = np.asarray(res.fittedvalues, dtype=float)
    resid = y - fitted
    sigma2 = float(res.scale)

    re_var = {}
    cov_re = getattr(res, "cov_re", None)
    if cov_re is not None:
        cov_re = np.atleast_2d(np.asarray(cov_re, dtype=float))
        if cov_re.shape == (1, 1):
            re_var["Intercept"] = float(cov_re[0, 0])
        else:
            for i in range(cov_re.shape[0]):
                re_var[f"re_{i}"] = float(cov_re[i, i])

    converged = bool(getattr(res, "converged", True))

    return MixedFitResult(
        names=names if len(names) == len(fe) else [f"x{i}" for i in range(len(fe))],
        coef=fe, se=se, zvalues=zvals, pvalues=pvals, resid=resid,
        sigma2=sigma2, groups=group_name, n_groups=n_groups, re_var=re_var,
        method=method_label, converged=converged,
        llf=float(res.llf), aic=float(res.aic), bic=float(res.bic),
        fitted=fitted, _sm_result=res,
    )


