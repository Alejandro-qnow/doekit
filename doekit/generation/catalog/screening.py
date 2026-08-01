"""Screening designs: Plackett-Burman, fold and a PB check.

Plackett-Burman designs from Hadamard matrices via Sylvester and Paley I/II,
covering multiples-of-4 orders at the minimum number of runs. Every constructed
matrix is validated as Hadamard before use; if an order is not constructible by
these methods the next one is used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from ...domain.model import Model
from ...domain.design import Design
from ._shared import is_prime as _is_prime
from ._shared import legendre as _legendre


def _is_hadamard(H: np.ndarray) -> bool:
    """Return whether ``H`` is a Hadamard matrix (square, ``+/-1``, ``H H' = nI``)."""
    n = H.shape[0]
    if H.shape[0] != H.shape[1]:
        return False
    if not np.all(np.isin(H, (-1, 1))):
        return False
    return np.allclose(H @ H.T, n * np.eye(n))


# --- Construcciones de Hadamard --------------------------------------------

def _sylvester(order: int) -> Optional[np.ndarray]:
    """Sylvester Hadamard if ``order`` is a power of 2, else ``None``."""
    if order == 1:
        return np.array([[1]])
    if order % 2 != 0:
        return None
    half = _hadamard(order // 2)
    if half is None:
        return None
    return np.block([[half, half], [half, -half]])


def _paley1(order: int) -> Optional[np.ndarray]:
    """Paley I: order q+1 with q prime, q == 3 (mod 4)."""
    q = order - 1
    if not (_is_prime(q) and q % 4 == 3):
        return None
    Q = np.array([[_legendre(a - b, q) for b in range(q)] for a in range(q)])
    H = np.ones((order, order), dtype=int)
    H[1:, 0] = -1
    H[1:, 1:] = np.eye(q, dtype=int) + Q
    return H if _is_hadamard(H) else None


def _paley2(order: int) -> Optional[np.ndarray]:
    """Paley II: order 2(q+1) with q prime, q == 1 (mod 4)."""
    if order % 2 != 0:
        return None
    q = order // 2 - 1
    if not (_is_prime(q) and q % 4 == 1):
        return None
    Q = np.array([[_legendre(a - b, q) for b in range(q)] for a in range(q)])
    # base (q+1) matrix with zero diagonal (symmetric for q == 1 mod 4)
    S = np.zeros((q + 1, q + 1), dtype=int)
    S[0, 1:] = 1
    S[1:, 0] = 1
    S[1:, 1:] = Q
    # 2x2 expansion: 0 -> [[1,-1],[-1,-1]], +1 -> [[1,1],[1,-1]], -1 -> [[-1,-1],[-1,1]]
    block0 = np.array([[1, -1], [-1, -1]])
    block_p = np.array([[1, 1], [1, -1]])
    block_n = np.array([[-1, -1], [-1, 1]])
    m = q + 1
    H = np.zeros((2 * m, 2 * m), dtype=int)
    for i in range(m):
        for j in range(m):
            b = block0 if S[i, j] == 0 else (block_p if S[i, j] == 1 else block_n)
            H[2 * i:2 * i + 2, 2 * j:2 * j + 2] = b
    return H if _is_hadamard(H) else None


@lru_cache(maxsize=None)
def _hadamard(order: int) -> Optional[np.ndarray]:
    """Return a Hadamard matrix of ``order`` or ``None`` if not constructible here."""
    if order == 1:
        return np.array([[1]])
    if order == 2:
        return np.array([[1, 1], [1, -1]])
    for builder in (_sylvester, _paley1, _paley2):
        H = builder(order)
        if H is not None:
            return H
    return None


def _next_hadamard_order(min_order: int) -> tuple[int, np.ndarray]:
    """Smallest multiple-of-4 order (>= min_order) with a constructible Hadamard."""
    n = max(4, ((min_order + 3) // 4) * 4)
    while n <= 4 * min_order + 8:
        H = _hadamard(n)
        if H is not None:
            return n, H
        n += 4
    raise ValueError(f"no constructible Hadamard found near {min_order}")


# --- Screening API ----------------------------------------------------------

def plackett_burman(n_factors: int, names: Optional[list[str]] = None,
                    model: Optional[Model] = None) -> Design:
    """Plackett-Burman design for ``n_factors`` factors.

    Uses the smallest Hadamard matrix (Sylvester/Paley) with ``N-1 >= n_factors``.
    Returns ``N`` runs and ``N-1`` columns: the first ``n_factors`` are the real
    factors and the rest are *dummy* columns (possible interaction aliases).

    Parameters
    ----------
    n_factors : int
        Number of real factors to screen.
    names : list of str, optional
        Names of the real factors; defaults to ``factor1..factorn``.
    model : Model, optional
        Model to attach; defaults to a no-intercept main-effects model.

    Returns
    -------
    Design
        The Plackett-Burman design, with ``order``, ``factors`` and
        ``dummy_factors`` in ``metadata``.
    """
    order, H = _next_hadamard_order(n_factors + 1)
    # normalize: first column all +1 (intercept); use the rest as factors
    H = H * H[0, :]  # ensures first row +1
    H = (H.T * H[:, 0]).T  # ensures first column +1
    design_cols = H[:, 1:]  # N-1 factor columns (zero-sum)

    n_dummy = design_cols.shape[1] - n_factors
    if names is None:
        names = [f"factor{i + 1}" for i in range(n_factors)]
    dummy_names = [f"dummy{i + 1}" for i in range(n_dummy)]
    all_names = list(names) + dummy_names

    df = pd.DataFrame(design_cols, columns=all_names).astype(int)
    meta = {
        "kind": "PlackettBurman",
        "order": order,
        "factors": list(names),
        "dummy_factors": dummy_names,
    }
    if model is None:
        model = Model.main_effects(all_names, intercept=False)
    return Design(matrix=df, model=model, metadata=meta)


def is_plackett_burman(design, tol: float = 1e-8) -> bool:
    """Check the Plackett-Burman properties of a design.

    Requires the three defining PB properties (entrywise, not aggregate):

    1. entries are ``+/-1`` (balanced two-level design);
    2. **each** column sums to zero (``D.sum(axis=0) == 0``);
    3. columns are entrywise orthogonal: ``D^T D = N I``.

    Parameters
    ----------
    design : Design or array-like
        The design (or its matrix) to check.
    tol : float, default 1e-8
        Absolute tolerance for the zero-sum and orthogonality checks.

    Returns
    -------
    bool
        Whether all three properties hold.
    """
    D = design.matrix.to_numpy() if isinstance(design, Design) else np.asarray(design)
    D = D.astype(float)
    n, ncols = D.shape
    two_level = bool(np.all(np.isin(D, (-1.0, 1.0))))
    zero_sum = bool(np.allclose(D.sum(axis=0), 0.0, atol=tol))
    orthogonal = bool(np.allclose(D.T @ D, n * np.eye(ncols), atol=tol))
    return two_level and zero_sum and orthogonal


def fold(design: Design) -> Design:
    """Duplicate the design by appending its sign-reflected mirror (foldover).

    Every interaction aliased with a main effect is de-confounded, raising the
    resolution.

    Parameters
    ----------
    design : Design
        The design to fold.

    Returns
    -------
    Design
        The folded design (``2N`` runs).
    """
    mirror = design.matrix * -1
    folded = pd.concat([design.matrix, mirror], ignore_index=True)
    meta = dict(design.metadata)
    meta["kind"] = meta.get("kind", "Design") + " (folded)"
    return Design(matrix=folded, factors=design.factors, model=design.model, metadata=meta)
