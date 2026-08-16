"""Simplex region for mixture experiments (Σx_i = 1, x_i ≥ 0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SimplexRegion:
    """Unit simplex region for mixture experiments in proportion space.

    Samples via Dirichlet draws on the residual simplex after per-component
    lower bounds; membership checks non-negativity, upper bound ``1``, sum
    constraint, and lower bounds.

    Formulas
    --------
    ``sum_i x_i = 1``, ``lower_i <= x_i <= 1``, ``x_i >= 0``.

    Parameters
    ----------
    factor_names : sequence of str
        Mixture component names.
    lower : dict, optional
        Per-component lower bounds (default 0). Upper bounds are implied by the
        remaining mass after other lowers.

    Examples
    --------
    >>> import doekit as ed
    >>> r = ed.SimplexRegion(["A", "B", "C"])
    >>> s = r.sample(3, rng=__import__("numpy").random.default_rng(0))
    >>> abs(s.sum(axis=1) - 1.0).max() < 1e-6
    True
    """

    factor_names: Sequence[str]
    lower: dict = field(default_factory=dict)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Draw ``n`` uniform points on the simplex (Dirichlet / stick-breaking).

        Parameters
        ----------
        n : int
            Number of points.
        rng : numpy.random.Generator, optional
            Random generator.

        Returns
        -------
        DataFrame
            Proportion columns summing to 1 per row.

        Raises
        ------
        ValueError
            When fewer than two components or lower bounds are infeasible.
        """
        rng = rng or np.random.default_rng()
        names = list(self.factor_names)
        q = len(names)
        if q < 2:
            raise ValueError("simplex region needs at least 2 components")

        lowers = np.array([float(self.lower.get(nm, 0.0)) for nm in names], dtype=float)
        if lowers.min() < 0 or lowers.sum() > 1.0 + 1e-12:
            raise ValueError("mixture lower bounds must be >= 0 and sum to <= 1")

        # Sample free mass on the residual simplex, then shift by lowers
        free = 1.0 - float(lowers.sum())
        # Dirichlet(1,...,1) = uniform on simplex
        raw = rng.dirichlet(np.ones(q), size=n)
        pts = lowers + free * raw
        # numerical cleanup
        pts = np.clip(pts, 0.0, 1.0)
        pts = pts / pts.sum(axis=1, keepdims=True)
        return pd.DataFrame(pts, columns=names)

    def contains(self, points: pd.DataFrame) -> np.ndarray:
        """Test simplex membership including lower-bound constraints.

        Parameters
        ----------
        points : DataFrame
            Candidate proportion rows.

        Returns
        -------
        ndarray of bool
            ``True`` for valid mixture points.
        """
        names = list(self.factor_names)
        if any(n not in points.columns for n in names):
            return np.zeros(len(points), dtype=bool)
        X = points.loc[:, names].to_numpy(dtype=float)
        ok = np.all(X >= -1e-9, axis=1) & np.all(X <= 1.0 + 1e-9, axis=1)
        ok &= np.abs(X.sum(axis=1) - 1.0) < 1e-6
        for i, nm in enumerate(names):
            lo = float(self.lower.get(nm, 0.0))
            ok &= X[:, i] >= lo - 1e-9
        return ok
