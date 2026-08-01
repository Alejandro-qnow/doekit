"""Shared helpers for design catalog constructors."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...domain.factors import Factor, ContinuousFactor


def resolve_factors(spec, n_hint: Optional[int] = None):
    """Normalize factor specification for RSM / DSD designs.

    Returns ``(names, factors_or_None)``. If ``spec`` is an integer, coded factors
    are generated (no decoding, columns in ``[-1, 1]``).
    """
    if isinstance(spec, int):
        names = [f"factor{i + 1}" for i in range(spec)]
        return names, None
    if isinstance(spec, dict):
        factors = [ContinuousFactor(k, float(v[0]), float(v[1])) for k, v in spec.items()]
    else:
        factors = list(spec)
    return [f.name for f in factors], factors


def decode_coded(coded: np.ndarray, factors: Optional[Sequence[Factor]]) -> np.ndarray:
    """Decode a coded matrix column-by-column to natural units."""
    if factors is None:
        return coded
    out = np.empty_like(coded, dtype=float)
    for j, f in enumerate(factors):
        out[:, j] = f.decode(coded[:, j])
    return out


def is_prime(n: int) -> bool:
    """Return True if ``n`` is prime."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def legendre(a: int, p: int) -> int:
    """Legendre symbol ``(a/p)`` for odd prime ``p``: 0, +1 or -1."""
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else 1
