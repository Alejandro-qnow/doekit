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

    Fits a main-effects model (no intercept by default) and returns one magnitude
    per term. The relative ordering is identical under both scales, so
    half-normal screening is unchanged.

    Formulas
    --------
    - **Coefficient scale:** ``beta_j`` from ``y ~ sum_j beta_j x_j`` (no intercept).
    - **Effect scale (Montgomery):** ``E_j = mean(y | x_j = +1) - mean(y | x_j = -1)``.
      For an orthogonal factor coded ``+/-1``, ``E_j = 2 * beta_j``.

    Parameters
    ----------
    design : Design
        The executed design.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model to fit; a main-effects model (no intercept) is used if omitted.
    scale : {"coefficient", "effect"}, default "coefficient"
        Magnitude convention:

        - ``"coefficient"``: the regression coefficients ``beta``.
        - ``"effect"``: the classical DoE effect (see Formulas).

    Returns
    -------
    pandas.Series
        Effects indexed by term name.

    Raises
    ------
    ValueError
        If ``scale`` is neither ``"coefficient"`` nor ``"effect"``.

    Examples
    --------
    >>> import doekit as ed
    >>> pb = ed.plackett_burman(5)
    >>> y = 2 * pb.matrix["factor1"]
    >>> eff = ed.main_effects(pb, y, scale="effect")
    >>> eff["factor1"] > eff.drop("factor1").abs().max()
    True
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

    Builds the data for a Daniel half-normal plot: inactive effects fall on a
    straight line through the origin; active effects depart upward.

    Formulas
    --------
    For rank ``k = 1..m`` (sorted by ``|effect|``):

    ``q_k = Phi^-1(0.5 + 0.5 * (k - 0.5) / m)``

    where ``Phi`` is the standard normal CDF.

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

    Examples
    --------
    >>> import doekit as ed
    >>> hnd = ed.half_normal_data([0.1, -5.0, 0.3], ["a", "b", "c"])
    >>> hnd.iloc[-1]["label"]
    'b'
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
