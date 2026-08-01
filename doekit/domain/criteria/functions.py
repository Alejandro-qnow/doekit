"""Concrete D/A/T/G/E/I criterion functions and registry."""

from __future__ import annotations

import numpy as np

from ...shared.errors import UnknownCriterionError
from .base import CriterionContext, score_criterion
from .linalg import info_matrix

TOLERANCE = 1e-9


def d_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """D-optimality score ``det((X'X)/N)^(1/p)`` (larger is better)."""
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = info_matrix(X, tolerance) / n
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p))


def a_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """A-optimality score ``p / tr((X'X)^-1) / N`` (larger is better)."""
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = info_matrix(X, tolerance)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    tr = np.trace(Minv)
    if tr <= 0:
        return 0.0
    return float(p / tr / n)


def t_criterion(model_matrix: np.ndarray, tolerance: float = 0.0) -> float:
    """T-optimality score ``tr(X'X) / N / p`` (larger is better)."""
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = info_matrix(X, tolerance)
    return float(np.trace(M) / n / p)


def g_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """G-optimality score ``p / max(diag(H))`` (larger is better)."""
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = info_matrix(X, tolerance)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    H = X @ Minv @ X.T
    m = np.max(np.diag(H))
    if m <= 0:
        return 0.0
    return float(p / m)


def e_criterion(model_matrix: np.ndarray, tolerance: float = 0.0) -> float:
    """E-optimality score ``min(eig(X'X)) / N`` (larger is better)."""
    X = np.asarray(model_matrix, dtype=float)
    n = X.shape[0]
    M = info_matrix(X, tolerance)
    lam = np.linalg.eigvalsh(M)
    return float(np.min(lam) / n)


def i_criterion(model_matrix: np.ndarray, moment_matrix: np.ndarray | None = None,
                tolerance: float = TOLERANCE) -> float:
    """I-optimality score ``1 / mean prediction variance`` (larger is better)."""
    X = np.asarray(model_matrix, dtype=float)
    M = info_matrix(X, tolerance)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    R = X if moment_matrix is None else np.asarray(moment_matrix, dtype=float)
    W = (R.T @ R) / R.shape[0]
    mean_var = float(np.trace(Minv @ W))
    if mean_var <= 0:
        return 0.0
    return 1.0 / mean_var


i_criterion._needs_moments = True  # type: ignore[attr-defined]


#: Homogeneous registry: every entry is scored via :func:`score_criterion`.
CRITERIA = {
    "D": d_criterion,
    "A": a_criterion,
    "T": t_criterion,
    "G": g_criterion,
    "E": e_criterion,
    "I": i_criterion,
}


def get_criterion(name: str):
    """Look up a criterion function by name (``"D"``, ``"A"``, ..., ``"I"``)."""
    key = name.strip().upper()
    if key in CRITERIA:
        return CRITERIA[key]
    raise UnknownCriterionError(
        f"unknown criterion: {name!r}. Use one of {list(CRITERIA)}"
    )


def all_criteria(model_matrix: np.ndarray,
                 context: CriterionContext | None = None) -> dict:
    """Evaluate every registered criterion on a design (for reporting)."""
    ctx = context or CriterionContext()
    return {name: score_criterion(fn, model_matrix, ctx)
            for name, fn in CRITERIA.items()}
