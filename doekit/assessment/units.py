"""Explicit coded / natural unit policy for assessment."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..domain.design import Design
from ..domain.factors.coding import encode_frame
from ..domain.model import Model
from ..shared.typing import Space


def coded_frame(design: Design) -> pd.DataFrame:
    """Design matrix with continuous/discrete factors in ``+/-1`` (Space.CODED).

    Assessment metrics and model matrices should use coded units so efficiencies
    and prediction variances are scale-invariant.

    Parameters
    ----------
    design : Design
        Source design.

    Returns
    -------
    pandas.DataFrame
        Coded factor columns; categoricals are unchanged.

    Examples
    --------
    >>> import doekit as ed
    >>> from doekit.assessment.units import coded_frame
    >>> cf = coded_frame(ed.full_factorial(3))
    >>> (cf.abs() <= 1).all().all()
    True
    >>> cf["x"].between(-1, 1).all()
    True
    """
    return encode_frame(design.matrix, design.factors)


def natural_frame(design: Design) -> pd.DataFrame:
    """Design matrix as stored (Space.NATURAL — typically constructor output).

    Returns a copy of ``design.matrix`` without recoding factors.

    Parameters
    ----------
    design : Design
        Source design.

    Returns
    -------
    pandas.DataFrame
        Factor columns in natural units as originally stored.
    """
    return design.matrix.copy()


def frame_for(design: Design, space: Space = Space.CODED) -> pd.DataFrame:
    """Return the design matrix in the requested coordinate space.

    Dispatches to :func:`coded_frame` or :func:`natural_frame` according to
    ``space``.

    Parameters
    ----------
    design : Design
        Source design.
    space : Space, default Space.CODED
        ``Space.CODED`` for ``+/-1`` factors; ``Space.NATURAL`` for stored units.

    Returns
    -------
    pandas.DataFrame
        Factor frame in the requested space.

    Raises
    ------
    ValueError
        If ``space`` is not recognized.
    """
    if space is Space.CODED:
        return coded_frame(design)
    if space is Space.NATURAL:
        return natural_frame(design)
    raise ValueError(f"unknown space: {space!r}")


def resolve_model(design: Design, model: Optional[Model] = None,
                  drop: Optional[Sequence[str]] = None,
                  intercept: bool = True) -> Model:
    """Return ``model`` or fall back to ``design.model`` / main effects.

    Shared resolver for evaluation and analysis: uses an explicit ``model`` when
    given, otherwise ``design.model``, otherwise a main-effects model on all
    non-dropped columns.

    Parameters
    ----------
    design : Design
        Source design.
    model : Model, optional
        Explicit model; overrides defaults when provided.
    drop : sequence of str, optional
        Columns to exclude when building the default main-effects model
        (e.g. block / grouping columns). Ignored when an explicit model is given.
    intercept : bool, default True
        Whether the default main-effects model includes an intercept.

    Returns
    -------
    Model
        Resolved model ready for ``model.matrix(frame)``.
    """
    model = model or design.model
    if model is None:
        drop_set = set(drop or [])
        cols = [c for c in design.matrix.columns if c not in drop_set]
        model = Model.main_effects(cols, intercept=intercept)
    return model
