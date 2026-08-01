"""Experimental region protocol (hypercube, simplex, constrained)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Region(Protocol):
    """Sampling / membership over the experimental region in coded units."""

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Draw ``n`` points from the region."""
        ...

    def contains(self, points: pd.DataFrame) -> np.ndarray:
        """Boolean mask of points inside the region."""
        ...
