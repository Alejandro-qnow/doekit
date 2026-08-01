"""Numeric factor with a finite set of levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .protocols import Factor


@dataclass
class DiscreteFactor(Factor):
    """Numeric factor snapped to nearest level on decode."""

    name: str
    levels: Sequence[float]

    def __post_init__(self):
        self.levels = sorted(float(v) for v in self.levels)
        if len(self.levels) < 2:
            raise ValueError(f"factor '{self.name}': >= 2 levels are required")
        self._low = self.levels[0]
        self._high = self.levels[-1]

    def encode(self, values):
        x = np.asarray(values, dtype=float)
        return 2.0 * (x - self._low) / (self._high - self._low) - 1.0

    def decode(self, coded):
        c = np.asarray(coded, dtype=float)
        natural = self._low + (c + 1.0) / 2.0 * (self._high - self._low)
        levels = np.asarray(self.levels)
        idx = np.abs(natural.reshape(-1, 1) - levels.reshape(1, -1)).argmin(axis=1)
        snapped = levels[idx]
        return snapped.reshape(np.shape(coded)) if np.ndim(coded) else float(snapped[0])

    def to_dict(self) -> dict:
        return {"type": "discrete", "name": self.name,
                "levels": [float(v) for v in self.levels]}
