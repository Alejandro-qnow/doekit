"""Mixture designs on the simplex (Scheffé lattice / centroid)."""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...domain.design import Design
from ...domain.factors import MixtureFactor, as_factors
from ...domain.model import Model
from ...shared.errors import InapplicableDesign


def _as_mixture_factors(factors) -> list[MixtureFactor]:
    """Normalize to a list of MixtureFactor (q >= 2)."""
    if isinstance(factors, int):
        facs = [MixtureFactor(f"x{i + 1}") for i in range(factors)]
    elif isinstance(factors, dict):
        facs = []
        for name, spec in factors.items():
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                facs.append(MixtureFactor(name, float(spec[0]), float(spec[1])))
            else:
                facs.append(MixtureFactor(name))
    else:
        raw = as_factors(factors)
        facs = []
        for f in raw:
            if isinstance(f, MixtureFactor):
                facs.append(f)
            else:
                # coerce continuous-like to mixture on [0, 1]
                facs.append(MixtureFactor(f.name, 0.0, 1.0))
    if len(facs) < 2:
        raise InapplicableDesign("mixture designs need at least 2 components")
    return facs


def _lattice_points(q: int, degree: int) -> np.ndarray:
    """All non-negative integer compositions of ``degree`` into ``q`` parts / degree."""
    # recursive compositions
    pts = []

    def rec(remaining, left, acc):
        if left == 1:
            pts.append(acc + [remaining])
            return
        for k in range(remaining + 1):
            rec(remaining - k, left - 1, acc + [k])

    rec(degree, q, [])
    arr = np.asarray(pts, dtype=float) / float(degree)
    # unique rows
    return np.unique(np.round(arr, decimals=12), axis=0)


def simplex_lattice(factors, degree: int = 2,
                    model: Optional[Model] = None) -> Design:
    """Build a ``{q, m}``-simplex lattice design (Scheffé).

    Places runs at lattice points where each proportion is a multiple of
    ``1/m`` on the ``(q-1)``-simplex. Lower bounds on components shift the
    lattice within the feasible simplex.

    Formulas
    --------
    Lattice points: ``x_i = k_i / m`` with ``k_i >= 0`` and ``sum_i k_i = m``.

    Parameters
    ----------
    factors : int, dict, or sequence of Factor
        Mixture components (``q``). An ``int`` names them ``x1..xq``.
    degree : int, default 2
        Lattice order ``m`` (proportions are multiples of ``1/m``).
    model : Model, optional
        Defaults to Scheffé linear (``m=1``) or quadratic (``m>=2``).

    Returns
    -------
    Design
        Rows are proportions summing to 1; metadata ``kind='SimplexLattice'``.

    Raises
    ------
    ValueError
        When ``degree < 1`` or lower bounds sum to more than 1.
    InapplicableDesign
        When fewer than two mixture components.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.simplex_lattice(3, degree=2)
    >>> d.n_runs > 0 and abs(d.matrix.sum(axis=1) - 1.0).max() < 1e-9
    True
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    facs = _as_mixture_factors(factors)
    names = [f.name for f in facs]
    q = len(names)
    pts = _lattice_points(q, degree)
    # apply lower bounds by shifting free mass (simple support check)
    lowers = np.array([f.lower for f in facs], dtype=float)
    if lowers.sum() > 1.0 + 1e-12:
        raise ValueError("mixture lower bounds sum to more than 1")
    if lowers.sum() > 0:
        free = 1.0 - lowers.sum()
        pts = lowers + free * pts
        pts = pts / pts.sum(axis=1, keepdims=True)

    mat = pd.DataFrame(pts, columns=names)
    if model is None:
        model = (Model.scheffe_linear(names) if degree == 1
                 else Model.scheffe_quadratic(names))
    return Design(
        matrix=mat, factors=list(facs), model=model,
        metadata={
            "kind": "SimplexLattice",
            "region": "simplex",
            "degree": int(degree),
            "n_components": q,
        },
    )


def simplex_centroid(factors, model: Optional[Model] = None) -> Design:
    """Build a simplex-centroid design (pure blends through overall centroid).

    For ``q`` components, includes every point with equal proportions among
    any non-empty subset of components (``2^q - 1`` runs). Covers vertices,
    edge midpoints, face centroids, and the overall centroid.

    Formulas
    --------
    For subset ``S`` of size ``r``: ``x_i = 1/r`` if ``i in S``, else ``0``.

    Parameters
    ----------
    factors : int, dict, or sequence of Factor
        Mixture components (``q >= 2``).
    model : Model, optional
        Defaults to Scheffé quadratic.

    Returns
    -------
    Design
        Centroid design with proportions summing to 1 per row.

    Raises
    ------
    InapplicableDesign
        When fewer than two mixture components.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.simplex_centroid(3)
    >>> d.n_runs == 2**3 - 1
    True
    """
    facs = _as_mixture_factors(factors)
    names = [f.name for f in facs]
    q = len(names)
    rows = []
    for r in range(1, q + 1):
        for comb in combinations(range(q), r):
            pt = np.zeros(q, dtype=float)
            pt[list(comb)] = 1.0 / r
            rows.append(pt)
    mat = pd.DataFrame(np.asarray(rows), columns=names)
    if model is None:
        model = Model.scheffe_quadratic(names)
    return Design(
        matrix=mat, factors=list(facs), model=model,
        metadata={
            "kind": "SimplexCentroid",
            "region": "simplex",
            "n_components": q,
        },
    )
