"""Factor deserialization and flexible-spec normalization."""

from __future__ import annotations

from ...shared.errors import UnknownFactorTypeError
from .protocols import Factor
from .continuous import ContinuousFactor
from .discrete import DiscreteFactor
from .categorical import CategoricalFactor
from .mixture import MixtureFactor

_FACTOR_TYPES = {
    "continuous": lambda d: ContinuousFactor(d["name"], d["low"], d["high"]),
    "discrete": lambda d: DiscreteFactor(d["name"], d["levels"]),
    "categorical": lambda d: CategoricalFactor(d["name"], d["levels"]),
    "mixture": lambda d: MixtureFactor(
        d["name"], d.get("lower", 0.0), d.get("upper", 1.0)
    ),
}


def register_factor_type(type_name: str, factory) -> None:
    """Register a factor ``type`` key for :func:`factor_from_dict` (Open/Closed)."""
    _FACTOR_TYPES[type_name] = factory


def factor_from_dict(d: dict) -> Factor:
    """Rebuild a :class:`Factor` from its ``to_dict`` output."""
    t = d.get("type")
    factory = _FACTOR_TYPES.get(t)
    if factory is None:
        raise UnknownFactorTypeError(f"unknown factor type: {t!r}")
    return factory(d)


def as_factors(spec) -> list[Factor]:
    """Normalize a flexible factor specification to ``list[Factor]``."""
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
