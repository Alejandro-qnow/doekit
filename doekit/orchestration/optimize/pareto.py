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
    """Map objectives into a common maximization space.

    Columns flagged ``"min"`` in ``goals`` are negated so every dimension is
    *larger-is-better*. When ``goals`` or ``columns`` is omitted, ``Y`` is
    returned unchanged (assumed already in max space).

    Parameters
    ----------
    Y : array-like
        Objective matrix (n x m).
    goals : mapping, optional
        ``{column: "max"|"min"}`` per objective.
    columns : sequence of str, optional
        Column names aligned with the last axis of ``Y``.

    Returns
    -------
    ndarray
        Same shape as ``Y`` in maximization space.

    Examples
    --------
    >>> from doekit.orchestration.optimize.pareto import to_max_space
    >>> import numpy as np
    >>> Y = np.array([[1.0, 5.0], [2.0, 3.0]])
    >>> Z = to_max_space(Y, goals={"y1": "max", "y2": "min"}, columns=["y1", "y2"])
    >>> Z[0, 1] < 0
    True
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if not goals or columns is None:
        return Y
    signs = np.array([-1.0 if str(goals.get(c, "max")).lower() == "min" else 1.0
                      for c in columns], dtype=float)
    return Y * signs


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return whether ``a`` Pareto-dominates ``b`` in maximization space.

    Point ``a`` dominates ``b`` when ``a >= b`` on every objective and
    ``a > b`` on at least one.

    Parameters
    ----------
    a, b : array-like
        Objective vectors (already in maximization space).

    Returns
    -------
    bool

    Examples
    --------
    >>> import doekit as ed
    >>> import numpy as np
    >>> ed.dominates(np.array([2.0, 3.0]), np.array([1.0, 2.0]))
    True
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``Y``.

    Assumes ``Y`` is already in maximization space (see :func:`to_max_space`).

    Parameters
    ----------
    Y : array-like
        Objective matrix (n x m).

    Returns
    -------
    ndarray of bool
        ``True`` for non-dominated rows.

    Examples
    --------
    >>> import doekit as ed
    >>> import numpy as np
    >>> Y = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])
    >>> ed.pareto_mask(Y).sum() >= 2
    True
    """
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
    """Return the non-dominated rows of ``Y``.

    Converts to maximization space internally, then returns the corresponding
    rows in the **original** objective units.

    Parameters
    ----------
    Y : array-like
        Objective matrix (n x m).
    goals : mapping, optional
        ``{column: "max"|"min"}`` per objective.
    columns : sequence of str, optional
        Column names aligned with the last axis of ``Y``.

    Returns
    -------
    ndarray
        Subset of ``Y`` containing only Pareto-optimal points.

    Examples
    --------
    >>> import doekit as ed
    >>> import numpy as np
    >>> Y = np.array([[1.0, 5.0], [2.0, 3.0], [0.5, 4.0]])
    >>> front = ed.pareto_front(Y, goals={"y1": "max", "y2": "min"}, columns=["y1", "y2"])
    >>> len(front) >= 1
    True
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    Z = to_max_space(Y, goals, columns)
    return Y[pareto_mask(Z)]


def hypervolume(Y: np.ndarray, ref: Sequence[float]) -> float:
    """Dominated hypervolume above a reference point (maximization space).

    Measures the volume of the region dominated by the Pareto front and bounded
    below by ``ref``. Exact for 1–2 objectives; Monte Carlo for higher dimensions.

    Formulas
    --------
    For 2 objectives with Pareto points sorted by decreasing first objective:

    - ``HV = sum_i (x_i - ref_0) * (y_i - y_{i-1})`` where ``y_{i-1}`` starts at
      ``ref_1``.

    For ``m > 2``, volume is estimated by Monte Carlo over the box
    ``[ref, ideal]``.

    Parameters
    ----------
    Y : array-like
        Objective points in maximization space (n x m).
    ref : sequence of float
        Reference (nadir) point; only the region ``> ref`` per axis is counted.

    Returns
    -------
    float
        Non-negative hypervolume; ``0.0`` when no point dominates ``ref``.

    Examples
    --------
    >>> import doekit as ed
    >>> import numpy as np
    >>> Y = np.array([[2.0, 3.0], [3.0, 2.0]])
    >>> ed.hypervolume(Y, ref=[0.0, 0.0]) > 0
    True
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
    """Build a nadir reference point just below the observed worst per objective.

    Useful as the hypervolume reference when none is supplied. Computed in
    maximization space after applying ``goals``.

    Parameters
    ----------
    Y : array-like
        Observed objective matrix (n x m).
    goals : mapping, optional
        ``{column: "max"|"min"}`` per objective.
    columns : sequence of str, optional
        Column names aligned with the last axis of ``Y``.
    margin : float, default 0.1
        Fraction of each objective span placed below the minimum.

    Returns
    -------
    ndarray
        Reference vector in maximization space.

    Examples
    --------
    >>> from doekit.orchestration.optimize.pareto import default_reference
    >>> import numpy as np
    >>> Y = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> ref = default_reference(Y)
    >>> len(ref) == 2
    True
    """
    Z = to_max_space(Y, goals, columns)
    lo = Z.min(axis=0)
    hi = Z.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo - margin * span
