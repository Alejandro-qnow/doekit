"""Explicit coded / natural unit policy for assessment."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..domain.design import Design
from ..domain.factors.coding import encode_frame
from ..domain.model import Model
from ..shared.typing import Space


def coded_frame(design: Design) -> pd.DataFrame:
    """Design matrix with continuous/discrete factors in ``+/-1`` (Space.CODED)."""
    return encode_frame(design.matrix, design.factors)


def natural_frame(design: Design) -> pd.DataFrame:
    """Design matrix as stored (Space.NATURAL — typically constructor output)."""
    return design.matrix.copy()


def frame_for(design: Design, space: Space = Space.CODED) -> pd.DataFrame:
    """Return the design matrix in the requested coordinate space."""
    if space is Space.CODED:
        return coded_frame(design)
    if space is Space.NATURAL:
        return natural_frame(design)
    raise ValueError(f"unknown space: {space!r}")


def resolve_model(design: Design, model: Optional[Model] = None,
                  drop: Optional[Sequence[str]] = None,
                  intercept: bool = True) -> Model:
    """Return ``model`` or fall back to ``design.model`` / main effects.

    Parameters
    ----------
    drop : sequence of str, optional
        Columns to exclude when building the default main-effects model
        (e.g. block / grouping columns). Ignored when an explicit model is given.
    intercept : bool, default True
        Whether the default main-effects model includes an intercept.
    """
    model = model or design.model
    if model is None:
        drop_set = set(drop or [])
        cols = [c for c in design.matrix.columns if c not in drop_set]
        model = Model.main_effects(cols, intercept=intercept)
    return model
