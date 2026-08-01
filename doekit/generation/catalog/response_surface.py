"""Response-surface designs: Box-Behnken and Central Composite.

Classic constructions (Box & Behnken; Central Composite) with:

- factors that accept a natural range and **decode** the matrix to real units;
- a configurable number of center points.
"""

from __future__ import annotations

from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ...domain.factors import Factor, ContinuousFactor
from ...domain.model import Model
from ...domain.design import Design
from ._shared import resolve_factors as _resolve_factors
from ._shared import decode_coded as _decode


# --- Box-Behnken -----------------------------------------------------------

# Suggested default center-point counts by number of factors.
_DEFAULT_BB_CENTER = [0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]


def _boxbehnken_coded(n: int, center: int) -> np.ndarray:
    """Build the coded Box-Behnken matrix for ``n`` factors and ``center`` points."""
    A_fact = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
    rows = int(0.5 * n * (n - 1) * A_fact.shape[0])
    A = np.zeros((rows, n))
    l = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            block = slice(l * 4, (l + 1) * 4)
            A[block, i] = A_fact[:, 0]
            A[block, j] = A_fact[:, 1]
            l += 1
    if center > 0:
        A = np.vstack([A, np.zeros((center, n))])
    return A


def box_behnken(factors, center: Optional[int] = None,
                model: Optional[Model] = None) -> Design:
    """Box-Behnken design.

    Parameters
    ----------
    factors : int or dict or sequence of Factor
        An integer (coded columns) or factors with a natural range (the matrix is
        decoded to real units).
    center : int, optional
        Number of center points; defaults to the classical suggestion for the
        given number of factors.
    model : Model, optional
        Model to attach; defaults to a full quadratic model.

    Returns
    -------
    Design
        The Box-Behnken design (requires >= 3 factors).

    Raises
    ------
    ValueError
        If fewer than 3 factors are given.
    """
    names, facs = _resolve_factors(factors)
    n = len(names)
    if n < 3:
        raise ValueError("Box-Behnken requires >= 3 factors")
    if center is None:
        center = _DEFAULT_BB_CENTER[n - 1] if n <= 16 else 3

    coded = _boxbehnken_coded(n, center)
    natural = _decode(coded, facs)
    df = pd.DataFrame(natural, columns=names)
    if model is None:
        model = Model.full_quadratic(names)
    meta = {"kind": "BoxBehnken", "center": center}
    return Design(matrix=df, factors=facs or [], model=model, metadata=meta)


# --- Central Composite -----------------------------------------------------

def _star(n: int, alpha: str, center: Sequence[int]) -> tuple[np.ndarray, float]:
    """Build the star (axial) block and its ``alpha`` distance for a CCD."""
    if alpha == "faced":
        a = 1.0
    elif alpha == "orthogonal":
        nc = 2.0 ** n
        nco = center[0]
        na = 2.0 * n
        nao = center[1]
        a = (n * (1 + nao / na) / (1 + nco / nc)) ** 0.5
    elif alpha == "rotatable":
        a = (2.0 ** n) ** 0.25
    else:
        raise ValueError(f"invalid alpha: {alpha}")
    H = np.zeros((2 * n, n))
    for i in range(n):
        H[2 * i:2 * i + 2, i] = [-1, 1]
    H = H * a
    return H, a


def _ccdesign_coded(n: int, center: Sequence[int], alpha: str, face: str) -> np.ndarray:
    """Build the coded Central Composite matrix (factorial + star + center points)."""
    H2, a = _star(n, alpha, center)
    factorial = np.array(list(product([-1, 1], repeat=n)), dtype=float)
    if face == "inscribed":
        H1 = factorial / a
        H2, _ = _star(n, "faced", [1, 1])
    elif face == "faced":
        H2, _ = _star(n, "faced", [1, 1])
        H1 = factorial
    elif face == "circumscribed":
        H1 = factorial
    else:
        raise ValueError(f"invalid face: {face}")

    C1 = np.zeros((center[0], n))
    C2 = np.zeros((center[1], n))
    H1 = np.vstack([H1, C1])
    H2 = np.vstack([H2, C2])
    return np.vstack([H1, H2])


def central_composite(factors, center: Sequence[int] = (4, 4),
                      alpha: str = "orthogonal", face: str = "circumscribed",
                      model: Optional[Model] = None) -> Design:
    """Central Composite design (CCD).

    Parameters
    ----------
    factors : int or dict or sequence of Factor
        An integer (coded columns) or factors with a natural range (decoded).
    center : sequence of int, default (4, 4)
        Center points ``(nc_factorial, nc_star)``.
    alpha : {"orthogonal", "rotatable", "faced"}, default "orthogonal"
        Star-distance rule.
    face : {"circumscribed", "inscribed", "faced"}, default "circumscribed"
        Placement of the factorial/star blocks.
    model : Model, optional
        Model to attach; defaults to a full quadratic model.

    Returns
    -------
    Design
        The CCD, with ``alpha_value``, ``alpha``, ``face`` and ``center`` in
        ``metadata`` (requires >= 2 factors).

    Raises
    ------
    ValueError
        If fewer than 2 factors are given.
    """
    names, facs = _resolve_factors(factors)
    n = len(names)
    if n < 2:
        raise ValueError("Central Composite requires >= 2 factors")
    center = list(center)

    coded = _ccdesign_coded(n, center, alpha, face)
    natural = _decode(coded, facs)
    df = pd.DataFrame(natural, columns=names)
    if model is None:
        model = Model.full_quadratic(names)
    _, a = _star(n, alpha, center)
    meta = {"kind": "CentralComposite", "alpha": alpha, "face": face,
            "alpha_value": float(a), "center": center}
    return Design(matrix=df, factors=facs or [], model=model, metadata=meta)
