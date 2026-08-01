"""Continuous factor over a natural interval ``[low, high]``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import Factor


@dataclass
class ContinuousFactor(Factor):
    """Continuous factor: ``low -> -1``, ``high -> +1``."""

    name: str
    low: float
    high: float

    def __post_init__(self):
        if self.high == self.low:
            raise ValueError(f"factor '{self.name}': low and high must differ")

    def encode(self, values):
        x = np.asarray(values, dtype=float)
        return 2.0 * (x - self.low) / (self.high - self.low) - 1.0

    def decode(self, coded):
        c = np.asarray(coded, dtype=float)
        return self.low + (c + 1.0) / 2.0 * (self.high - self.low)

    def to_dict(self) -> dict:
        return {"type": "continuous", "name": self.name,
                "low": float(self.low), "high": float(self.high)}
