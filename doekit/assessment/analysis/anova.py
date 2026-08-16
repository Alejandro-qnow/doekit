"""ANOVA tables for OLS fits."""

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

def anova_table(fit: FitResult, typ: Union[int, str] = 2) -> pd.DataFrame:
    """Partial-F (Wald) ANOVA table for an OLS :class:`FitResult`.

    For single-degree-of-freedom terms (the usual coded DoE case) each row is a
    partial F-test of that term given all others. Multi-df categorical blocks
    appear as separate dummy rows (``block[...]``).

    Formulas
    --------
    - **Single-df term:** ``F_j = t_j^2`` where ``t_j`` is the OLS t-statistic
      (equivalent to a Type III partial F when terms are orthogonal).
    - **p-value:** ``P(F_{1, dof} > F_j)`` from the residual dof of the fit.

    Parameters
    ----------
    fit : FitResult
        Result of :func:`fit_linear_model`.
    typ : {2, 3, "II", "III"}, default 2
        Accepted for API compatibility; the table is a partial-F / Wald table
        (Type III for single-df terms). A formula-based statsmodels Type II/III
        table is used when the underlying result was fit with a formula.

    Returns
    -------
    DataFrame
        Columns ``term``, ``df``, ``F``, ``p_value`` (plus residual row).

    Examples
    --------
    >>> import doekit as ed
    >>> pb = ed.plackett_burman(5)
    >>> y = 3 * pb.matrix["factor1"]
    >>> anova = ed.anova_table(ed.fit_linear_model(pb, y))
    >>> "factor1" in anova["term"].values
    True
    """
    # Prefer statsmodels formula ANOVA when available
    res = fit._sm_result
    if res is not None and getattr(res.model, "formula", None):
        from statsmodels.stats.anova import anova_lm  # noqa: PLC0415
        return anova_lm(res, typ=typ)

    rows = []
    for name, t, p in zip(fit.names, fit.tvalues, fit.pvalues):
        if name in ("(Intercept)", "Intercept"):
            continue
        f_stat = float(t ** 2) if np.isfinite(t) else float("nan")
        rows.append({
            "term": name,
            "df": 1,
            "F": f_stat,
            "p_value": float(p) if np.isfinite(p) else float("nan"),
        })
    rows.append({
        "term": "Residual",
        "df": int(fit.dof),
        "F": float("nan"),
        "p_value": float("nan"),
    })
    return pd.DataFrame(rows)


