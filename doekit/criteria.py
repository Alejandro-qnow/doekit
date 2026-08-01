"""Optimality criteria over the model matrix ``X`` (``N x p``).

Default numerical tolerance ``1e-9``. Convention: **larger is better** for every
function, so that ``optimal_design`` always maximizes the chosen criterion.

- D-optimality: maximizes ``det((X'X)/N)^(1/p)`` (Shannon information of estimates).
- A-optimality: minimizes ``tr((X'X)^-1)`` -> metric ``p / tr(M^-1) / N``.
- T-optimality: maximizes ``tr(X'X)`` -> metric ``tr(M) / N / p``.
- G-optimality: minimizes the maximum of ``diag(H)`` -> metric ``p / max diag(H)``.
- E-optimality: maximizes the smallest eigenvalue -> metric ``min lambda(M) / N``.
- I-optimality: minimizes the mean prediction variance over a region -> metric
  ``1 / (mean prediction variance)``.
"""

from __future__ import annotations

import numpy as np

TOLERANCE = 1e-9


def _info_matrix(X: np.ndarray, tolerance: float) -> np.ndarray:
    """Return the (Tikhonov-regularized) information matrix ``X'X + tol*I``."""
    return X.T @ X + np.eye(X.shape[1]) * tolerance


def d_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """D-optimality score ``det((X'X)/N)^(1/p)`` (larger is better).

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X``.
    tolerance : float, default ``1e-9``
        Ridge added to the diagonal of ``X'X`` for numerical stability.

    Returns
    -------
    float
        The score, or ``0.0`` if the information matrix is singular.
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = _info_matrix(X, tolerance) / n
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p))


def a_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """A-optimality score ``p / tr((X'X)^-1) / N`` (larger is better).

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X``.
    tolerance : float, default ``1e-9``
        Ridge added to the diagonal of ``X'X`` for numerical stability.

    Returns
    -------
    float
        The score, or ``0.0`` if the information matrix is singular.
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = _info_matrix(X, tolerance)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    tr = np.trace(Minv)
    if tr <= 0:
        return 0.0
    return float(p / tr / n)


def t_criterion(model_matrix: np.ndarray, tolerance: float = 0.0) -> float:
    """T-optimality score ``tr(X'X) / N / p`` (larger is better).

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X``.
    tolerance : float, default ``0.0``
        Ridge added to the diagonal of ``X'X``.

    Returns
    -------
    float
        The score.
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = _info_matrix(X, tolerance)
    return float(np.trace(M) / n / p)


def g_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """G-optimality score ``p / max(diag(H))`` (larger is better).

    ``H = X (X'X)^-1 X'`` is the hat matrix, so ``max(diag(H))`` is the worst-case
    scaled prediction variance among the design points.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X``.
    tolerance : float, default ``1e-9``
        Ridge added to the diagonal of ``X'X`` for numerical stability.

    Returns
    -------
    float
        The score, or ``0.0`` if the information matrix is singular.
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = _info_matrix(X, tolerance)
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
    """E-optimality score ``min(eig(X'X)) / N`` (larger is better).

    Maximizes the smallest eigenvalue of the information matrix, i.e. improves the
    worst-conditioned direction of the estimate covariance.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X``.
    tolerance : float, default ``0.0``
        Ridge added to the diagonal of ``X'X``.

    Returns
    -------
    float
        The score.
    """
    X = np.asarray(model_matrix, dtype=float)
    n = X.shape[0]
    M = _info_matrix(X, tolerance)
    lam = np.linalg.eigvalsh(M)
    return float(np.min(lam) / n)


def i_criterion(model_matrix: np.ndarray, moment_matrix: np.ndarray | None = None,
                tolerance: float = TOLERANCE) -> float:
    """I-optimality score ``1 / mean prediction variance`` (larger is better).

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X`` of the candidate design.
    moment_matrix : ndarray, optional
        Model matrix that defines the prediction region (e.g. the candidate set).
        If ``None``, the design itself is used as the region.
    tolerance : float, default ``1e-9``
        Ridge added to the diagonal of ``X'X`` for numerical stability.

    Returns
    -------
    float
        The score, or ``0.0`` if the information matrix is singular.

    Notes
    -----
    Scale note: unlike :func:`d_criterion`/:func:`a_criterion`, here the
    information matrix is **not** divided by ``N``, so its magnitude is not
    comparable to that of the other criteria. It is correct for *ranking* designs
    of the same size ``N`` (its only use in ``optimal_design``).
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = _info_matrix(X, tolerance)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    R = X if moment_matrix is None else np.asarray(moment_matrix, dtype=float)
    W = (R.T @ R) / R.shape[0]  # region moments
    mean_var = float(np.trace(Minv @ W))
    if mean_var <= 0:
        return 0.0
    return 1.0 / mean_var


#: Registry of candidate-independent criteria (name -> callable(X) -> score).
CRITERIA = {
    "D": d_criterion,
    "A": a_criterion,
    "T": t_criterion,
    "G": g_criterion,
    "E": e_criterion,
}


def get_criterion(name: str):
    """Look up a criterion function by name (``"D"``, ``"A"``, ..., ``"I"``).

    Parameters
    ----------
    name : str
        Criterion letter, case-insensitive.

    Returns
    -------
    callable
        The matching criterion function.

    Raises
    ------
    ValueError
        If ``name`` is not a known criterion.
    """
    key = name.strip().upper()
    if key in CRITERIA:
        return CRITERIA[key]
    if key == "I":
        return i_criterion
    raise ValueError(f"unknown criterion: {name!r}. Use one of {list(CRITERIA) + ['I']}")


def all_criteria(model_matrix: np.ndarray) -> dict:
    """Evaluate every candidate-independent criterion on a design (for reporting).

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        The model matrix ``X``.

    Returns
    -------
    dict
        Mapping ``{criterion name: score}`` for D/A/T/G/E.
    """
    return {name: fn(model_matrix) for name, fn in CRITERIA.items()}
