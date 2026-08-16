"""Minimal split-plot design generation (whole-plot / subplot structure)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...domain.design import Design
from ...domain.factors import (
    ContinuousFactor, DiscreteFactor, CategoricalFactor, as_factors, Factor,
)
from ...domain.model import Model
from ...shared.errors import InapplicableDesign
from .factorial import full_factorial


def _levels_for(f: Factor, nlev: int = 2) -> list:
    if isinstance(f, ContinuousFactor):
        if nlev == 2:
            return [f.low, f.high]
        return [f.low, (f.low + f.high) / 2.0, f.high]
    if isinstance(f, DiscreteFactor):
        return list(f.levels)
    if isinstance(f, CategoricalFactor):
        return list(f.levels)
    raise InapplicableDesign(
        f"split_plot_design cannot level factor type {type(f).__name__}"
    )


def split_plot_design(
    whole_plot,
    subplot,
    *,
    n_whole_plots: Optional[int] = None,
    whole_plot_reps: int = 1,
    model: Optional[Model] = None,
    seed: Optional[int] = None,
) -> Design:
    """Build a simple split-plot design (whole-plot / subplot structure).

    Whole-plot (hard-to-change) factor combinations define plots; within each
    plot, all subplot factor combinations are run. Plot identity is stored in
    column ``whole_plot_id`` for mixed-model analysis
    (``fit_mixed_model(..., groups="whole_plot_id")``).

    Parameters
    ----------
    whole_plot : factor spec
        Hard-to-change factors (int / dict / sequence of Factor).
    subplot : factor spec
        Easy-to-change factors within a plot.
    n_whole_plots : int, optional
        If set, sample this many whole-plot level combinations (with
        replacement across the WP factorial); default = full WP factorial
        times ``whole_plot_reps``.
    whole_plot_reps : int, default 1
        Replicates of each whole-plot combination when ``n_whole_plots`` is None.
    model : Model, optional
        Defaults to main effects of all treatment factors (no plot id).
    seed : int, optional
        RNG seed when sampling whole plots.

    Returns
    -------
    Design
        Matrix includes treatment columns plus ``whole_plot_id``.

    Raises
    ------
    InapplicableDesign
        When whole_plot or subplot is empty or factor types are unsupported.
    ValueError
        When factor names overlap or ``n_whole_plots < 1``.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.split_plot_design(whole_plot=1, subplot=2)
    >>> "whole_plot_id" in d.matrix.columns
    True
    """
    wp = as_factors(whole_plot)
    sp = as_factors(subplot)
    if not wp or not sp:
        raise InapplicableDesign("split_plot_design needs whole_plot and subplot factors")

    wp_names = [f.name for f in wp]
    sp_names = [f.name for f in sp]
    if set(wp_names) & set(sp_names):
        raise ValueError("whole_plot and subplot factor names must be disjoint")

    wp_levels = {f.name: _levels_for(f, 2) for f in wp}
    sp_levels = {f.name: _levels_for(f, 2) for f in sp}
    wp_grid = full_factorial(wp_levels).matrix.reset_index(drop=True)
    sp_grid = full_factorial(sp_levels).matrix.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    if n_whole_plots is not None:
        if n_whole_plots < 1:
            raise ValueError("n_whole_plots must be >= 1")
        idx = rng.integers(0, len(wp_grid), size=n_whole_plots)
        wp_rows = wp_grid.iloc[idx].reset_index(drop=True)
    else:
        blocks = [wp_grid] * max(1, int(whole_plot_reps))
        wp_rows = pd.concat(blocks, ignore_index=True)

    pieces = []
    for plot_id, (_, wrow) in enumerate(wp_rows.iterrows()):
        block = sp_grid.copy()
        for c in wp_names:
            block[c] = wrow[c]
        block["whole_plot_id"] = plot_id
        pieces.append(block)

    mat = pd.concat(pieces, ignore_index=True)
    # column order: WP, SP, id
    mat = mat.loc[:, wp_names + sp_names + ["whole_plot_id"]]

    treat_names = wp_names + sp_names
    if model is None:
        model = Model.main_effects(treat_names)

    return Design(
        matrix=mat,
        factors=list(wp) + list(sp),
        model=model,
        metadata={
            "kind": "SplitPlot",
            "whole_plot": wp_names,
            "subplot": sp_names,
            "hard_to_change": wp_names,
            "n_whole_plots": int(wp_rows.shape[0]),
            "blocking": {"column": "whole_plot_id",
                         "n_blocks": int(wp_rows.shape[0])},
        },
    )
