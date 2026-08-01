"""Shared linear-algebra primitives for criteria, evaluation and search."""

from __future__ import annotations

import numpy as np


def info_matrix(X: np.ndarray, tolerance: float) -> np.ndarray:
    """Return the (Tikhonov-regularized) information matrix ``X'X + tol*I``."""
    return X.T @ X + np.eye(X.shape[1]) * tolerance


def inv_info(X_sel: np.ndarray, tolerance: float) -> np.ndarray:
    """Inverse of the ridge-regularized information matrix of selected rows."""
    p = X_sel.shape[1]
    return np.linalg.inv(X_sel.T @ X_sel + np.eye(p) * tolerance)


def leverage(Xr: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    """Return ``x' M^-1 x`` per row (unscaled prediction variance)."""
    return np.einsum("ij,jk,ik->i", Xr, Minv, Xr)
