"""Categorical factor (non-numeric levels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .protocols import Factor


@dataclass
class CategoricalFactor(Factor):
    """Categorical factor: level <-> integer index ``0..k-1``."""

    name: str
    levels: Sequence[object]

    def __post_init__(self):
        self.levels = list(self.levels)
        if len(self.levels) < 2:
            raise ValueError(f"factor '{self.name}': >= 2 levels are required")

    @property
    def is_categorical(self) -> bool:
        return True

    def encode(self, values):
        lut = {lvl: i for i, lvl in enumerate(self.levels)}
        arr = np.atleast_1d(np.asarray(values, dtype=object))
        out = np.array([lut[v] for v in arr], dtype=int)
        return out if np.ndim(values) else int(out[0])

    def decode(self, coded):
        arr = np.atleast_1d(np.asarray(coded)).astype(int)
        out = np.array([self.levels[i] for i in arr], dtype=object)
        return out if np.ndim(coded) else self.levels[int(arr[0])]

    def to_dict(self) -> dict:
        return {"type": "categorical", "name": self.name, "levels": list(self.levels)}
