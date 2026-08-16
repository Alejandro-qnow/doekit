"""Mixture component: proportion in ``[lower, upper]`` with Σx_i = 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import Factor


@dataclass
class MixtureFactor(Factor):
    """Mixture component for Scheffé / simplex designs.

    Values are **proportions** on ``[lower, upper]`` subject to ``sum x_i = 1``
    across components. Encoding is the identity (proportions are not mapped to
    ``±1``); the experimental region is a simplex, not a hypercube.

    Formulas
    --------
    Constraint: ``sum_i x_i = 1`` with ``lower_i <= x_i <= upper_i``.

    Parameters
    ----------
    name : str
        Component name.
    lower : float, default 0.0
        Lower bound on the proportion.
    upper : float, default 1.0
        Upper bound on the proportion.

    Raises
    ------
    ValueError
        When bounds violate ``0 <= lower < upper <= 1``.

    Examples
    --------
    >>> import doekit as ed
    >>> f = ed.MixtureFactor("A")
    >>> float(f.encode(0.5))
    0.5
    """

    name: str
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.lower < self.upper <= 1.0):
            raise ValueError(
                f"mixture factor '{self.name}': need 0 <= lower < upper <= 1 "
                f"(got lower={self.lower}, upper={self.upper})"
            )

    @property
    def is_mixture(self) -> bool:
        return True

    def encode(self, values):
        """Return proportions unchanged (identity coding for Scheffé models)."""
        return np.asarray(values, dtype=float)

    def decode(self, coded):
        return np.asarray(coded, dtype=float)

    def to_dict(self) -> dict:
        return {
            "type": "mixture",
            "name": self.name,
            "lower": float(self.lower),
            "upper": float(self.upper),
        }
