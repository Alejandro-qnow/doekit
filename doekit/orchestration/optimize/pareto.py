"""Pareto utilities for multi-objective optimization (dominance, front, hypervolume).

Everything is computed in a common **maximization space**: objectives flagged
``"min"`` are negated on the way in, so dominance / hypervolume never special-case
direction. The public helpers accept raw objective values plus a ``goals`` map.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


def to_max_space(Y: np.ndarray, goals: Optional[Mapping[str, str]] = None,
                 columns: Optional[Sequence[str]] = None) -> np.ndarray:
    """Flip ``min`` objectives so every column is *larger-is-better*."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if not goals or columns is None:
        return Y
    signs = np.array([-1.0 if str(goals.get(c, "max")).lower() == "min" else 1.0
                      for c in columns], dtype=float)
    return Y * signs


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if ``a`` dominates ``b`` in maximization space (>= all, > one)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``Y`` (already in max space)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i != j and mask[j] and dominates(Y[j], Y[i]):
                mask[i] = False
                break
    return mask


def pareto_front(Y: np.ndarray, goals: Optional[Mapping[str, str]] = None,
                 columns: Optional[Sequence[str]] = None) -> np.ndarray:
    """Return the non-dominated rows of ``Y`` (in the original objective space)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    Z = to_max_space(Y, goals, columns)
    return Y[pareto_mask(Z)]


def hypervolume(Y: np.ndarray, ref: Sequence[float]) -> float:
    """Dominated hypervolume above ``ref`` (maximization space).

    Exact for 1-2 objectives; Monte-Carlo for higher dimensions.

    Parameters
    ----------
    Y : ndarray
        Objective points in maximization space (n x m).
    ref : sequence of float
        Reference (nadir) point; only the region ``> ref`` is counted.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    ref = np.asarray(ref, dtype=float)
    m = Y.shape[1]
    # keep only the part of each point that dominates the reference
    Y = Y[np.all(Y > ref, axis=1)] if len(Y) else Y
    if len(Y) == 0:
        return 0.0
    Y = Y[pareto_mask(Y)]
    if m == 1:
        return float(np.max(Y[:, 0]) - ref[0])
    if m == 2:
        order = np.argsort(-Y[:, 0])  # descending by first objective
        hv = 0.0
        prev_y = ref[1]
        for x, y in Y[order]:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
        return float(hv)
    return _hypervolume_mc(Y, ref)


def _hypervolume_mc(Y: np.ndarray, ref: np.ndarray, n: int = 20000,
                    seed: Optional[int] = None) -> float:
    """Monte-Carlo dominated volume for >2 objectives."""
    ideal = Y.max(axis=0)
    box = ideal - ref
    if np.any(box <= 0):
        return 0.0
    rng = np.random.default_rng(seed)
    pts = ref + rng.random((n, len(ref))) * box
    # a sample is dominated if some front point is >= it on all objectives
    dominated = np.zeros(n, dtype=bool)
    for p in Y:
        dominated |= np.all(pts <= p, axis=1)
    return float(dominated.mean() * np.prod(box))


def default_reference(Y: np.ndarray, goals: Optional[Mapping[str, str]] = None,
                      columns: Optional[Sequence[str]] = None,
                      margin: float = 0.1) -> np.ndarray:
    """A nadir reference (in max space) just below the observed worst per objective."""
    Z = to_max_space(Y, goals, columns)
    lo = Z.min(axis=0)
    hi = Z.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo - margin * span
