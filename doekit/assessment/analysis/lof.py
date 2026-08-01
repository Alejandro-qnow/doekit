"""Lack-of-fit vs pure-error decomposition."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...shared.serialize import jsonify as _jsonify, as_float_list as _as_float_list
from ...domain.criteria.linalg import leverage as _leverage_rows

from .ols import fit_linear_model
from .helpers import _resolve_blocks, _factor_frame

def lack_of_fit(design: Design, response, model: Optional[Model] = None,
                blocks=None) -> pd.DataFrame:
    """Lack-of-fit vs pure-error decomposition when replicate runs exist.

    Identical rows of the (non-block) factor matrix are treated as replicates.
    Pure-error SS is the within-replicate sum of squares; lack-of-fit SS is
    ``RSS - SS_PE``. An F-test compares LOF against pure error.

    Parameters
    ----------
    design : Design
        Executed design.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model used for the fitted RSS; defaults as in :func:`fit_linear_model`.
    blocks : str or array-like, optional
        Optional fixed blocks (same semantics as :func:`fit_linear_model`).

    Returns
    -------
    DataFrame
        One-row-per-source table with columns ``source``, ``df``, ``ss``,
        ``ms``, ``F``, ``p_value``.

    Raises
    ------
    ValueError
        If there are no replicate groups (no pure-error degrees of freedom).
    """
    fit = fit_linear_model(design, response, model=model, blocks=blocks)
    _, _, drop = _resolve_blocks(design, blocks)
    frame = _factor_frame(design, drop=drop)
    y = np.asarray(response, dtype=float).reshape(-1)

    keys = pd.Series(
        [tuple(row) for row in frame.to_numpy()],
        name="run_key",
    )
    # pure error from within-replicate variance
    ss_pe = 0.0
    df_pe = 0
    for _, idx in keys.groupby(keys).groups.items():
        yi = y[list(idx)]
        if len(yi) < 2:
            continue
        ss_pe += float(((yi - yi.mean()) ** 2).sum())
        df_pe += len(yi) - 1

    if df_pe <= 0:
        raise ValueError(
            "lack_of_fit requires replicate runs (identical factor-level rows); "
            "none were found in the design matrix."
        )

    rss = float(fit.resid @ fit.resid)
    ss_lof = max(0.0, rss - ss_pe)
    df_lof = fit.dof - df_pe
    if df_lof < 0:
        # e.g. blocks absorbed df; clamp and warn via nan F
        df_lof = 0

    ms_pe = ss_pe / df_pe if df_pe > 0 else float("nan")
    ms_lof = ss_lof / df_lof if df_lof > 0 else float("nan")
    if df_lof > 0 and ms_pe > 0:
        f_stat = ms_lof / ms_pe
        p_val = float(stats.f.sf(f_stat, df_lof, df_pe))
    else:
        f_stat = float("nan")
        p_val = float("nan")

    return pd.DataFrame([
        {"source": "lack_of_fit", "df": df_lof, "ss": ss_lof, "ms": ms_lof,
         "F": f_stat, "p_value": p_val},
        {"source": "pure_error", "df": df_pe, "ss": ss_pe, "ms": ms_pe,
         "F": float("nan"), "p_value": float("nan")},
        {"source": "residual", "df": fit.dof, "ss": rss,
         "ms": rss / fit.dof if fit.dof > 0 else float("nan"),
         "F": float("nan"), "p_value": float("nan")},
    ])


# ---------------------------------------------------------------------------
# Mixed models
# ---------------------------------------------------------------------------

