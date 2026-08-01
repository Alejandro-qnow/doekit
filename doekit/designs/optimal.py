"""Optimal design: KL-exchange (D-optimal) and Fedorov (generic criterion).

- choice of criterion (D/A/I/...);
- KL-exchange for D-optimality and Fedorov for generic criteria;
- multi-start (``n_starts``) to escape local optima;
- all criteria of the final design are reported.

The seed-growth phase adds candidates greedily by largest prediction variance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .. import criteria as _crit
from ..model import Model
from .base import Design


def _leverage(X: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    """Prediction variance (leverage) of each row: ``x' M^-1 x``."""
    return np.einsum("ij,jk,ik->i", X, Minv, X)


def _inv(X_sel: np.ndarray, tolerance: float) -> np.ndarray:
    """Inverse of the (ridge-regularized) information matrix of the selected rows."""
    p = X_sel.shape[1]
    return np.linalg.inv(X_sel.T @ X_sel + np.eye(p) * tolerance)


def kl_exchange(model_matrix: np.ndarray, experiments: int, design_k: Optional[int] = None,
                candidates_l: Optional[int] = None, seed_design_size: int = 2,
                max_iterations: int = 1000, tolerance: float = 1e-9,
                rng: Optional[np.random.Generator] = None) -> list[int]:
    """KL-exchange for D-optimality. Returns the selected row indices.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix of the candidate set; the returned indices refer to its rows.
    experiments : int
        Number of runs to select.
    design_k, candidates_l : int, optional
        Sizes of the exchange pools (worst design rows / best candidates).
    seed_design_size : int, default 2
        Size of the initial random seed before greedy growth.
    max_iterations : int, default 1000
        Maximum number of exchange iterations.
    tolerance : float, default 1e-9
        Ridge and stopping tolerance.
    rng : numpy.random.Generator, optional
        Random generator for the seed selection.

    Returns
    -------
    list of int
        Indices of the selected rows.
    """
    X = np.asarray(model_matrix, dtype=float)
    N, p = X.shape
    rng = rng or np.random.default_rng()
    if design_k is None:
        design_k = experiments
    if candidates_l is None:
        candidates_l = N - experiments

    design_k = min(design_k, experiments)

    # --- seed phase: greedy growth by maximum variance ---
    seed = list(rng.choice(N, size=seed_design_size, replace=False))
    while len(seed) < experiments:
        Minv = _inv(X[seed], tolerance)
        cand_idx = np.setdiff1d(np.arange(N), seed, assume_unique=False)
        lev = _leverage(X[cand_idx], Minv)
        seed.append(int(cand_idx[np.argmax(lev)]))

    design = list(seed)

    # --- exchange phase (KL) ---
    for _ in range(max_iterations):
        Minv = _inv(X[design], tolerance)
        cand_idx = np.setdiff1d(np.arange(N), design, assume_unique=False)
        if cand_idx.size == 0:
            break

        var_design = _leverage(X[design], Minv)
        var_cand = _leverage(X[cand_idx], Minv)

        k_sel = np.argsort(var_design)[:design_k]                  # lowest variance -> remove
        l_sel = np.argsort(var_cand)[::-1][:min(candidates_l, cand_idx.size)]  # highest -> add

        Xk = X[[design[k] for k in k_sel]]
        Xl = X[cand_idx[l_sel]]
        cross = Xk @ Minv @ Xl.T  # (design_k x candidates_l)

        best = (-1, -1)
        best_delta = 1.0
        for ki, k in enumerate(k_sel):
            dk = var_design[k]
            for li, l in enumerate(l_sel):
                delta = (1.0 - dk) * (1.0 + var_cand[l]) + cross[ki, li] ** 2
                if delta > best_delta:
                    best_delta = delta
                    best = (k, l)

        if best_delta - 1.0 <= tolerance:
            break

        k, l = best
        design[k] = int(cand_idx[l])

    return design


def fedorov_exchange(model_matrix: np.ndarray, experiments: int, criterion_fn,
                     n_restarts_rows: Optional[int] = None, max_iterations: int = 1000,
                     tolerance: float = 1e-9, rng: Optional[np.random.Generator] = None,
                     moment_matrix: Optional[np.ndarray] = None) -> list[int]:
    """Fedorov exchange for a generic criterion (larger is better).

    Tries swapping each design row for each candidate and applies the best swap
    until there is no improvement. Works with any criterion.

    Parameters
    ----------
    model_matrix : ndarray, shape (N, p)
        Model matrix of the candidate set.
    experiments : int
        Number of runs to select.
    criterion_fn : callable
        Criterion function ``fn(X)`` (or ``fn(X, moment_matrix)`` for I).
    n_restarts_rows : int, optional
        Unused placeholder kept for signature compatibility.
    max_iterations : int, default 1000
        Maximum number of exchange iterations.
    tolerance : float, default 1e-9
        Minimum gain required to accept a swap.
    rng : numpy.random.Generator, optional
        Random generator for the initial selection.
    moment_matrix : ndarray, optional
        Region moment matrix passed to the criterion (used by I-optimality).

    Returns
    -------
    list of int
        Indices of the selected rows.
    """
    X = np.asarray(model_matrix, dtype=float)
    N = X.shape[0]
    rng = rng or np.random.default_rng()

    def score(idx):
        if moment_matrix is not None:
            return criterion_fn(X[idx], moment_matrix)
        return criterion_fn(X[idx])

    design = list(rng.choice(N, size=experiments, replace=False))
    current = score(design)

    for _ in range(max_iterations):
        cand_idx = np.setdiff1d(np.arange(N), design, assume_unique=False)
        best_gain = tolerance
        best_swap = None
        for ki in range(experiments):
            for c in cand_idx:
                trial = design.copy()
                trial[ki] = int(c)
                s = score(trial)
                if s - current > best_gain:
                    best_gain = s - current
                    best_swap = (ki, int(c))
        if best_swap is None:
            break
        ki, c = best_swap
        design[ki] = c
        current = score(design)

    return design


def optimal_design(candidates: Design, n_runs: int, model: Optional[Model] = None,
                   criterion: str = "D", algorithm: Optional[str] = None,
                   n_starts: int = 1, seed: Optional[int] = None,
                   tolerance: float = 1e-9, report=None, **kl_kwargs) -> Design:
    """Select ``n_runs`` D/A/I-optimal runs from a candidate set.

    Parameters
    ----------
    candidates : Design
        The candidate set (e.g. from ``random_design`` or ``full_factorial``).
    n_runs : int
        Number of runs to select.
    model : Model, optional
        Model to optimize; taken from ``candidates.model`` if omitted.
    criterion : {"D", "A", "T", "G", "E", "I"}, default "D"
        Optimality criterion.
    algorithm : {"kl", "fedorov"}, optional
        ``"kl"`` (D only) or ``"fedorov"`` (any criterion); defaults to KL for D
        and Fedorov otherwise.
    n_starts : int, default 1
        Number of independent restarts; the best design is returned.
    seed : int, optional
        Seed controlling all restarts.
    tolerance : float, default 1e-9
        Ridge and stopping tolerance.
    report : None, bool, str, Path or dict, optional
        If not ``None``, a design-quality HTML report is generated and its path is
        stored in ``result.metadata["report_path"]``.
    **kl_kwargs
        Extra keyword arguments forwarded to :func:`kl_exchange`.

    Returns
    -------
    Design
        The optimal subset, with ``criterion``, ``algorithm``, ``selected_rows``
        and all final ``criteria`` in ``metadata``.

    Raises
    ------
    ValueError
        If no model is available, or ``algorithm`` is unknown.
    """
    model = model or candidates.model
    if model is None:
        raise ValueError("a model is required (candidates.model or the model argument)")

    X_full = model.matrix(candidates.matrix)
    crit_fn = _crit.get_criterion(criterion)
    use_i = criterion.strip().upper() == "I"

    if algorithm is None:
        algorithm = "kl" if criterion.strip().upper() == "D" else "fedorov"

    def eval_score(idx):
        return _crit.i_criterion(X_full[idx], X_full) if use_i else crit_fn(X_full[idx])

    best_idx, best_score = None, -np.inf
    base = np.random.default_rng(seed)
    for _ in range(max(1, n_starts)):
        rng = np.random.default_rng(base.integers(0, 2**32 - 1))
        if algorithm == "kl":
            idx = kl_exchange(X_full, experiments=n_runs, tolerance=tolerance,
                              rng=rng, **kl_kwargs)
        elif algorithm == "fedorov":
            moments = X_full if use_i else None
            idx = fedorov_exchange(X_full, n_runs, crit_fn, tolerance=tolerance,
                                   rng=rng, moment_matrix=moments)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        s = eval_score(idx)
        if s > best_score:
            best_score, best_idx = s, idx

    sub = candidates.matrix.iloc[best_idx].reset_index(drop=True)
    X_best = X_full[best_idx]
    crits = _crit.all_criteria(X_best)
    crits["I"] = _crit.i_criterion(X_best, X_full)
    meta = {
        "kind": "OptimalDesign",
        "criterion": criterion.strip().upper(),
        "algorithm": algorithm,
        "n_starts": n_starts,
        "selected_rows": list(best_idx),
        "criteria": {k: round(v, 6) for k, v in crits.items()},
    }
    result = Design(matrix=sub, factors=candidates.factors, model=model, metadata=meta)
    if report is not None:
        from ..report import run_report_arg  # noqa: PLC0415
        path = run_report_arg(result, model=model, report=report, seed=seed)
        if path is not None:
            result.metadata["report_path"] = str(path)
    return result
