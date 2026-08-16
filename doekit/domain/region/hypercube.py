"""Hypercube region in coded units (default DoE region)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class HypercubeRegion:
    """Axis-aligned hypercube region for standard factorial/RSM designs.

    Continuous factors are sampled uniformly on ``[low, high]`` (default coded
    ``[-1, 1]``); categorical factors draw uniformly from declared levels.

    Parameters
    ----------
    factor_names : sequence of str
        Column names to sample.
    categorical_levels : dict, optional
        Mapping ``name -> levels`` for categorical factors.
    low, high : float
        Bounds for continuous factors (default coded ``[-1, 1]``).

    Examples
    --------
    >>> import doekit as ed
    >>> r = ed.HypercubeRegion(["x1", "x2"])
    >>> s = r.sample(5, rng=__import__("numpy").random.default_rng(0))
    >>> s.shape
    (5, 2)
    """

    factor_names: Sequence[str]
    categorical_levels: dict = field(default_factory=dict)
    low: float = -1.0
    high: float = 1.0

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Draw ``n`` uniform points from the hypercube.

        Parameters
        ----------
        n : int
            Number of points.
        rng : numpy.random.Generator, optional
            Random generator.

        Returns
        -------
        DataFrame
            Sampled factor columns.
        """
        rng = rng or np.random.default_rng()
        cols = {}
        for name in self.factor_names:
            if name in self.categorical_levels:
                levels = np.array(self.categorical_levels[name], dtype=object)
                cols[name] = rng.choice(levels, size=n)
            else:
                cols[name] = rng.uniform(self.low, self.high, size=n)
        return pd.DataFrame(cols)

    def contains(self, points: pd.DataFrame) -> np.ndarray:
        """Test membership in the hypercube (and categorical level sets).

        Parameters
        ----------
        points : DataFrame
            Candidate points.

        Returns
        -------
        ndarray of bool
            ``True`` for rows inside the region.
        """
        mask = np.ones(len(points), dtype=bool)
        for name in self.factor_names:
            if name not in points.columns:
                mask[:] = False
                break
            if name in self.categorical_levels:
                allowed = set(self.categorical_levels[name])
                mask &= points[name].isin(allowed).to_numpy()
            else:
                v = points[name].to_numpy(dtype=float)
                mask &= (v >= self.low) & (v <= self.high)
        return mask


def region_from_design(design, model=None):
    """Build the default experimental region for a design or model.

    Mixture factors (or ``metadata['region'] == 'simplex'``) yield a
    :class:`~doekit.domain.region.simplex.SimplexRegion`; otherwise a
    :class:`HypercubeRegion` with categorical level maps from attached factors.

    Parameters
    ----------
    design : Design
        Design whose factors and metadata define the region kind.
    model : Model, optional
        Model whose :attr:`~doekit.domain.model.Model.factor_names` override
        column selection when provided.

    Returns
    -------
    HypercubeRegion or SimplexRegion
        Region appropriate for I/G-efficiency and FDS sampling.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(2)
    >>> r = ed.region_from_design(d)
    >>> hasattr(r, "sample")
    True
    """
    from .simplex import SimplexRegion  # noqa: PLC0415

    names = (model.factor_names if model is not None
             else list(design.matrix.columns))
    facs = list(design.factors or [])
    mix = [f for f in facs if getattr(f, "is_mixture", False)]
    kind = (design.metadata or {}).get("region") or (design.metadata or {}).get("kind", "")
    if mix or str(kind).startswith("Simplex") or kind == "simplex":
        mix_names = [f.name for f in mix] if mix else list(names)
        lowers = {f.name: float(getattr(f, "lower", 0.0)) for f in mix}
        return SimplexRegion(factor_names=mix_names, lower=lowers)
    cat = {f.name: list(f.levels) for f in facs
           if getattr(f, "is_categorical", False)}
    return HypercubeRegion(factor_names=names, categorical_levels=cat)
