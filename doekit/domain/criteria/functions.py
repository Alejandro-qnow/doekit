"""Concrete D/A/T/G/E/I criterion functions and registry."""

from __future__ import annotations

import numpy as np

from ...shared.errors import UnknownCriterionError
from .base import CriterionContext, score_criterion
from .linalg import info_matrix

TOLERANCE = 1e-9


def d_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """D-optimality score for a model matrix (larger is better).

    Normalizes the information matrix by ``N`` before taking the geometric mean
    of eigenvalues. Used by exchange algorithms and design comparison.

    Formulas
    --------
    ``score = det((X'X) / N)^(1/p)``

    where ``N`` is the number of runs and ``p`` is the number of parameters.
    Equivalent to maximizing ``det(X'X)^(1/p) / N``.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix ``X`` in coded units.
    tolerance : float, default 1e-9
        Ridge added to ``X'X`` before the determinant (via :func:`info_matrix`).

    Returns
    -------
    float
        D-optimality score; ``0.0`` when ``X'X`` is singular or non-positive.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> ed.d_criterion(X) > 0.9
    True
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = info_matrix(X, tolerance) / n
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p))


def a_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """A-optimality score for a model matrix (larger is better).

    Penalizes the average variance of coefficient estimates. Inverse of the
    trace of the variance-covariance matrix, normalized by run count.

    Formulas
    --------
    ``score = p / tr((X'X)^-1) / N``

    where ``p`` is the number of parameters and ``N`` the number of runs.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix ``X`` in coded units.
    tolerance : float, default 1e-9
        Ridge added to ``X'X`` before inversion.

    Returns
    -------
    float
        A-optimality score; ``0.0`` when ``X'X`` is singular.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> ed.a_criterion(X) > 0.9
    True
    """
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
    """T-optimality score for a model matrix (larger is better).

    Rewards designs with large total information (trace of ``X'X``), normalized
    per parameter and run.

    Formulas
    --------
    ``score = tr(X'X) / N / p``

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix ``X`` in coded units.
    tolerance : float, default 0.0
        Ridge added to ``X'X`` (usually zero for T).

    Returns
    -------
    float
        T-optimality score.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> ed.t_criterion(X) > 0
    True
    """
    X = np.asarray(model_matrix, dtype=float)
    n, p = X.shape
    M = info_matrix(X, tolerance)
    return float(np.trace(M) / n / p)


def g_criterion(model_matrix: np.ndarray, tolerance: float = TOLERANCE) -> float:
    """G-optimality score for a model matrix (larger is better).

    Minimizes the maximum prediction variance over design points (equivalence
    theorem links G- and D-optimality under regularity).

    Formulas
    --------
    ``score = p / max(diag(H))``

    where ``H = X (X'X)^-1 X'`` is the hat matrix and ``p`` the number of
    parameters. ``max(diag(H))`` is the maximum leverage among design rows.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix ``X`` in coded units.
    tolerance : float, default 1e-9
        Ridge added to ``X'X`` before inversion.

    Returns
    -------
    float
        G-optimality score; ``0.0`` when ``X'X`` is singular.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> ed.g_criterion(X) > 0.9
    True
    """
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
    """E-optimality score for a model matrix (larger is better).

    Maximizes the smallest eigenvalue of the normalized information matrix,
    guarding against near-singular directions in parameter space.

    Formulas
    --------
    ``score = min(eig(X'X)) / N``

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix ``X`` in coded units.
    tolerance : float, default 0.0
        Ridge added to ``X'X`` before eigenvalues are computed.

    Returns
    -------
    float
        E-optimality score (smallest eigenvalue of ``X'X / N``).

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> ed.e_criterion(X) > 0
    True
    """
    X = np.asarray(model_matrix, dtype=float)
    n = X.shape[0]
    M = info_matrix(X, tolerance)
    lam = np.linalg.eigvalsh(M)
    return float(np.min(lam) / n)


def i_criterion(model_matrix: np.ndarray, moment_matrix: np.ndarray | None = None,
                tolerance: float = TOLERANCE) -> float:
    """I-optimality score for a model matrix (larger is better).

    Minimizes average prediction variance over a region; the score is the
    reciprocal of that mean variance.

    Formulas
    --------
    ``mean_var = tr((X'X)^-1 W)`` with ``W = (R'R) / n_R``

    where ``R`` is the region moment matrix (defaults to ``X`` when omitted).
    ``score = 1 / mean_var``.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix ``X`` of the selected design.
    moment_matrix : ndarray, optional
        Model matrix ``R`` evaluated on a region sample (required for true
        I-optimality; defaults to ``X`` when omitted).
    tolerance : float, default 1e-9
        Ridge added to ``X'X`` before inversion.

    Returns
    -------
    float
        I-optimality score; ``0.0`` when ``X'X`` is singular.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> ed.i_criterion(X) > 0
    True
    """
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
    """Look up a criterion function by name.

    Parameters
    ----------
    name : str
        Criterion key: ``"D"``, ``"A"``, ``"T"``, ``"G"``, ``"E"``, or ``"I"``
        (case-insensitive).

    Returns
    -------
    callable
        The registered criterion function.

    Raises
    ------
    UnknownCriterionError
        When ``name`` is not a registered criterion.

    Examples
    --------
    >>> from doekit.domain.criteria import get_criterion
    >>> import doekit as ed
    >>> fn = get_criterion("D")
    >>> fn is ed.d_criterion
    True
    """
    key = name.strip().upper()
    if key in CRITERIA:
        return CRITERIA[key]
    raise UnknownCriterionError(
        f"unknown criterion: {name!r}. Use one of {list(CRITERIA)}"
    )


def all_criteria(model_matrix: np.ndarray,
                 context: CriterionContext | None = None) -> dict:
    """Evaluate every registered criterion on a design matrix.

    Convenience wrapper for reporting; each entry is scored via
    :func:`score_criterion` so I-optimality receives region moments from
    ``context`` when provided.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix of the design.
    context : CriterionContext, optional
        Shared context (tolerance, moment matrix for I).

    Returns
    -------
    dict
        Mapping ``{"D": score, "A": score, ...}`` with one float per criterion.

    Examples
    --------
    >>> import doekit as ed
    >>> from doekit.domain.criteria import all_criteria
    >>> d = ed.full_factorial(3)
    >>> X = ed.Model.main_effects(d.factor_names).matrix(d.matrix)
    >>> sorted(all_criteria(X)) == ["A", "D", "E", "G", "I", "T"]
    True
    """
    ctx = context or CriterionContext()
    return {name: score_criterion(fn, model_matrix, ctx)
            for name, fn in CRITERIA.items()}
