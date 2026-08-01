"""Mixture component: proportion in ``[lower, upper]`` with Σx_i = 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import Factor


@dataclass
class MixtureFactor(Factor):
    """Mixture component (Scheffé / simplex designs).

    Values are **proportions** already on a meaningful scale. Encoding is the
    identity (assessment does not map them to ``±1``); the experimental region
    is a simplex, not a hypercube.

    Parameters
    ----------
    name : str
        Component name.
    lower : float, default 0.0
        Lower bound on the proportion.
    upper : float, default 1.0
        Upper bound on the proportion.
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
        """Proportions stay as-is (identity coding for Scheffé models)."""
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
