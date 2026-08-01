"""Factor abstractions with natural <-> coded conversion.

A factor knows how to translate between its natural units (degrees, mol/L, a
category) and the coded units used by DoE designs:

- continuous/discrete factors: natural interval ``[low, high]`` <-> ``[-1, 1]``
- categorical factors: level <-> integer index ``0..k-1``

Every design constructor accepts a list of factors and returns the matrix in
natural units, keeping the coded version as metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class Factor:
    """Common interface of a factor. Do not instantiate directly."""

    name: str

    def encode(self, values):  # pragma: no cover - interface
        """Map natural-unit ``values`` to coded units. Implemented by subclasses."""
        raise NotImplementedError

    def decode(self, coded):  # pragma: no cover - interface
        """Map ``coded`` values back to natural units. Implemented by subclasses."""
        raise NotImplementedError

    def to_dict(self) -> dict:  # pragma: no cover - interface
        """Serialize the factor to a plain ``dict``. Implemented by subclasses."""
        raise NotImplementedError

    @property
    def is_categorical(self) -> bool:
        """Whether the factor is categorical (dummy-coded in the model matrix)."""
        return False


@dataclass
class ContinuousFactor(Factor):
    """Continuous factor over the natural interval ``[low, high]``.

    The coding is the standard linear DoE map::

        coded   = 2 * (x - low) / (high - low) - 1
        natural = low + (coded + 1) / 2 * (high - low)

    so that ``low -> -1`` and ``high -> +1``.

    Parameters
    ----------
    name : str
        Factor name (used as the design-matrix column label).
    low, high : float
        Endpoints of the natural interval; must differ.
    """

    name: str
    low: float
    high: float

    def __post_init__(self):
        if self.high == self.low:
            raise ValueError(f"factor '{self.name}': low and high must differ")

    def encode(self, values):
        """Map natural ``values`` to coded units in ``[-1, 1]``."""
        x = np.asarray(values, dtype=float)
        return 2.0 * (x - self.low) / (self.high - self.low) - 1.0

    def decode(self, coded):
        """Map ``coded`` values in ``[-1, 1]`` back to natural units."""
        c = np.asarray(coded, dtype=float)
        return self.low + (c + 1.0) / 2.0 * (self.high - self.low)

    def to_dict(self) -> dict:
        """Serialize to ``{"type": "continuous", "name", "low", "high"}``."""
        return {"type": "continuous", "name": self.name,
                "low": float(self.low), "high": float(self.high)}


@dataclass
class DiscreteFactor(Factor):
    """Numeric factor with a finite set of levels.

    Coded as a continuous factor between ``min(levels)`` and ``max(levels)``, but
    on decoding the value is snapped to the nearest valid level.

    Parameters
    ----------
    name : str
        Factor name.
    levels : sequence of float
        Allowed numeric levels (at least two); sorted on construction.
    """

    name: str
    levels: Sequence[float]

    def __post_init__(self):
        self.levels = sorted(float(v) for v in self.levels)
        if len(self.levels) < 2:
            raise ValueError(f"factor '{self.name}': >= 2 levels are required")
        self._low = self.levels[0]
        self._high = self.levels[-1]

    def encode(self, values):
        """Map natural ``values`` to coded units in ``[-1, 1]``."""
        x = np.asarray(values, dtype=float)
        return 2.0 * (x - self._low) / (self._high - self._low) - 1.0

    def decode(self, coded):
        """Map ``coded`` values back to natural units, snapped to the nearest level."""
        c = np.asarray(coded, dtype=float)
        natural = self._low + (c + 1.0) / 2.0 * (self._high - self._low)
        levels = np.asarray(self.levels)
        idx = np.abs(natural.reshape(-1, 1) - levels.reshape(1, -1)).argmin(axis=1)
        snapped = levels[idx]
        return snapped.reshape(np.shape(coded)) if np.ndim(coded) else float(snapped[0])

    def to_dict(self) -> dict:
        """Serialize to ``{"type": "discrete", "name", "levels"}``."""
        return {"type": "discrete", "name": self.name,
                "levels": [float(v) for v in self.levels]}


@dataclass
class CategoricalFactor(Factor):
    """Categorical factor (non-numeric levels, e.g. materials or suppliers).

    Encodes a level to its integer index ``0..k-1`` and decodes back to the
    level. For the model matrix a dummy encoding is used (see :mod:`doekit.model`).

    Parameters
    ----------
    name : str
        Factor name.
    levels : sequence
        The distinct levels (at least two); the first is the reference level.
    """

    name: str
    levels: Sequence[object]

    def __post_init__(self):
        self.levels = list(self.levels)
        if len(self.levels) < 2:
            raise ValueError(f"factor '{self.name}': >= 2 levels are required")

    @property
    def is_categorical(self) -> bool:
        """Always ``True`` for categorical factors."""
        return True

    def encode(self, values):
        """Map each level in ``values`` to its integer index ``0..k-1``."""
        lut = {lvl: i for i, lvl in enumerate(self.levels)}
        arr = np.atleast_1d(np.asarray(values, dtype=object))
        out = np.array([lut[v] for v in arr], dtype=int)
        return out if np.ndim(values) else int(out[0])

    def decode(self, coded):
        """Map integer indices in ``coded`` back to their levels."""
        arr = np.atleast_1d(np.asarray(coded)).astype(int)
        out = np.array([self.levels[i] for i in arr], dtype=object)
        return out if np.ndim(coded) else self.levels[int(arr[0])]

    def to_dict(self) -> dict:
        """Serialize to ``{"type": "categorical", "name", "levels"}``."""
        return {"type": "categorical", "name": self.name, "levels": list(self.levels)}


def factor_from_dict(d: dict) -> Factor:
    """Rebuild a :class:`Factor` from its ``to_dict`` output (serialization/MCP).

    Parameters
    ----------
    d : dict
        Mapping produced by ``Factor.to_dict`` (must carry a ``"type"`` key of
        ``"continuous"``, ``"discrete"`` or ``"categorical"``).

    Returns
    -------
    Factor
        The reconstructed factor instance.

    Raises
    ------
    ValueError
        If ``d["type"]`` is not a known factor type.
    """
    t = d.get("type")
    if t == "continuous":
        return ContinuousFactor(d["name"], d["low"], d["high"])
    if t == "discrete":
        return DiscreteFactor(d["name"], d["levels"])
    if t == "categorical":
        return CategoricalFactor(d["name"], d["levels"])
    raise ValueError(f"unknown factor type: {t!r}")


def as_factors(spec) -> list[Factor]:
    """Normalize a flexible factor specification to ``list[Factor]``.

    Parameters
    ----------
    spec : int or dict or sequence of Factor
        Accepts:

        - an integer ``n`` -> ``n`` continuous factors ``factor1..factorn`` in ``[-1, 1]``;
        - a ``dict`` ``{name: (low, high)}`` (continuous) or ``{name: [levels...]}``
          (discrete if all-numeric, otherwise categorical);
        - a list of :class:`Factor` (returned as-is).

    Returns
    -------
    list of Factor
        The normalized factors.
    """
    if isinstance(spec, int):
        return [ContinuousFactor(f"factor{i + 1}", -1.0, 1.0) for i in range(spec)]
    if isinstance(spec, dict):
        factors: list[Factor] = []
        for name, rng in spec.items():
            if isinstance(rng, tuple) and len(rng) == 2:
                factors.append(ContinuousFactor(name, float(rng[0]), float(rng[1])))
            else:
                vals = list(rng)
                if all(isinstance(v, (int, float)) for v in vals):
                    factors.append(DiscreteFactor(name, vals))
                else:
                    factors.append(CategoricalFactor(name, vals))
        return factors
    return list(spec)
