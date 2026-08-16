"""Experimental region protocol (hypercube, simplex, constrained)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Region(Protocol):
    """Protocol for sampling and membership over an experimental region.

    Implementations draw points in the units used for evaluation (coded
    ``[-1, 1]`` for standard factors, proportions on the simplex for mixtures).

    Methods
    -------
    sample(n, rng)
        Draw ``n`` points from the region.
    contains(points)
        Boolean mask of points inside the region.
    """

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Draw ``n`` points from the region.

        Parameters
        ----------
        n : int
            Number of points to sample.
        rng : numpy.random.Generator, optional
            Random generator; a default is created when omitted.

        Returns
        -------
        DataFrame
            Sampled points with one column per factor in the region.
        """
        ...

    def contains(self, points: pd.DataFrame) -> np.ndarray:
        """Test which rows lie inside the region.

        Parameters
        ----------
        points : DataFrame
            Candidate points (must include region factor columns).

        Returns
        -------
        ndarray of bool, shape (n_points,)
            ``True`` where the point is inside the region.
        """
        ...
