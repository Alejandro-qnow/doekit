"""Main effects and half-normal plot data."""

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
from .helpers import _factor_frame

def main_effects(design: Design, response, model: Optional[Model] = None,
                 scale: str = "coefficient") -> pd.Series:
    """Estimated main effects of a screening design.

    Parameters
    ----------
    design : Design
        The executed design.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model to fit; a main-effects model (no intercept) is used if omitted.
    scale : {"coefficient", "effect"}, default "coefficient"
        Magnitude convention (the relative *ordering* is the same either way, so
        significance detection is unchanged):

        - ``"coefficient"``: the regression coefficients ``beta``.
        - ``"effect"``: the **classical DoE effect** (Montgomery), defined as
          ``mean(+1) - mean(-1)``. For an orthogonal factor in ``+/-1`` this
          equals ``2*beta``. This is the magnitude the screening literature plots
          in the half-normal plot.

    Returns
    -------
    pandas.Series
        Effects indexed by term name.

    Raises
    ------
    ValueError
        If ``scale`` is neither ``"coefficient"`` nor ``"effect"``.
    """
    if model is None:
        frame = _factor_frame(design)
        model = Model.main_effects(list(frame.columns), intercept=False)
    fit = fit_linear_model(design, response, model=model)
    coef = fit.coef
    if scale == "effect":
        coef = 2.0 * coef
    elif scale != "coefficient":
        raise ValueError("scale must be 'coefficient' or 'effect'")
    return pd.Series(coef, index=fit.names, name=scale)



def half_normal_data(effects, labels: Optional[Sequence[str]] = None):
    """Half-normal quantiles vs. effect magnitude, sorted.

    Parameters
    ----------
    effects : array-like
        Estimated effects (or coefficients).
    labels : sequence of str, optional
        Labels for each effect; defaults to ``e1..em``.

    Returns
    -------
    DataFrame
        Columns ``label``, ``abs_effect`` and ``half_normal_quantile``, ready to
        plot (see :mod:`doekit.presentation.render.figures_mpl`).
    """
    eff = np.asarray(effects, dtype=float)
    if labels is None:
        labels = [f"e{i + 1}" for i in range(len(eff))]
    abs_eff = np.abs(eff)
    order = np.argsort(abs_eff)
    m = len(eff)
    ranks = np.arange(1, m + 1)
    quantiles = stats.norm.ppf(0.5 + 0.5 * (ranks - 0.5) / m)
    return pd.DataFrame({
        "label": np.asarray(labels)[order],
        "abs_effect": abs_eff[order],
        "half_normal_quantile": quantiles,
    })
