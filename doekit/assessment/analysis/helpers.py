"""Blocking / model-resolution helpers for analysis."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...shared.serialize import jsonify as _jsonify, as_float_list as _as_float_list
from ...domain.criteria.linalg import leverage as _leverage_rows

from ..units import resolve_model as _units_resolve_model

# ---------------------------------------------------------------------------

def _blocking_column(design: Design) -> Optional[str]:
    """Return the block column name declared in ``design.metadata['blocking']``."""
    blocking = design.metadata.get("blocking")
    if blocking is None:
        return None
    if isinstance(blocking, str):
        return blocking
    if isinstance(blocking, dict):
        return blocking.get("column")
    return None


def _factor_frame(design: Design, drop: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Design matrix without non-factor columns (blocks / grouping)."""
    drop_set = set(drop or [])
    blk = _blocking_column(design)
    if blk:
        drop_set.add(blk)
    cols = [c for c in design.matrix.columns if c not in drop_set]
    return design.matrix.loc[:, cols].copy()



def _resolve_model(design: Design, model: Optional[Model] = None,
                   drop: Optional[Sequence[str]] = None) -> Model:
    """Return ``model`` or fall back to main effects on non-block columns."""
    return _units_resolve_model(design, model, drop=drop, intercept=True)

def _resolve_blocks(design: Design, blocks) -> tuple[Optional[np.ndarray], Optional[str], list[str]]:
    """Resolve block labels.

    Parameters
    ----------
    blocks : None, False, str, or array-like
        Column name in ``design.matrix``, or a per-run label array.
        ``None`` falls back to ``design.metadata['blocking']``; ``False``
        forces an unblocked fit even if metadata declares a block column.

    Returns
    -------
    labels, name, drop_cols
        ``labels`` is ``None`` when unblocked; ``name`` is the column name or
        ``"array"``; ``drop_cols`` are matrix columns to exclude from the model.
        When metadata declares a block column but ``blocks is False``, that
        column is still dropped from the factor model (it is not a factor).
    """
    meta_col = _blocking_column(design)

    if blocks is False:
        # Explicit opt-out of fixed-block dummies; still exclude the block column
        # from the default factor frame so it is not treated as a regressor.
        return None, None, [meta_col] if meta_col else []

    if blocks is None:
        if meta_col is None:
            return None, None, []
        blocks = meta_col

    if isinstance(blocks, str):
        if blocks not in design.matrix.columns:
            raise ValueError(
                f"blocks column {blocks!r} not found in design.matrix columns "
                f"{list(design.matrix.columns)}"
            )
        labels = design.matrix[blocks].to_numpy()
        return labels, blocks, [blocks]

    labels = np.asarray(blocks)
    if labels.shape[0] != design.n_runs:
        raise ValueError(
            f"blocks array length ({labels.shape[0]}) must match "
            f"n_runs ({design.n_runs})"
        )
    return labels, "array", []


def _augment_blocks(X: np.ndarray, names: list[str], labels: np.ndarray
                    ) -> tuple[np.ndarray, list[str]]:
    """Append drop-first block dummies to ``X``."""
    levels = pd.unique(pd.Series(labels))
    if len(levels) < 2:
        raise ValueError(
            f"blocks must have at least 2 distinct levels (got {len(levels)})"
        )
    dummies = pd.get_dummies(pd.Series(labels), drop_first=True, dtype=float)
    dummies.columns = [f"block[{c}]" for c in dummies.columns]
    X2 = np.hstack([X, dummies.to_numpy(dtype=float)])
    names2 = list(names) + list(dummies.columns)
    return X2, names2


def attach_blocks(design: Design, blocks, name: str = "block") -> Design:
    """Return a copy of ``design`` with a block column and ``metadata['blocking']``.

    Writes per-run block labels into the design matrix and records the blocking
    metadata so :func:`fit_linear_model` picks up the column automatically.

    Parameters
    ----------
    design : Design
        Source design.
    blocks : array-like
        Per-run block labels (length ``n_runs``).
    name : str, default "block"
        Column name for the block factor.

    Returns
    -------
    Design
        New design with the block column appended (or overwritten).

    Raises
    ------
    ValueError
        If ``blocks`` length does not match ``design.n_runs``.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.attach_blocks(ed.plackett_burman(4), [0, 0, 0, 0, 1, 1, 1, 1])
    >>> d.metadata["blocking"]["n_blocks"]
    2
    """
    labels = np.asarray(blocks)
    if labels.shape[0] != design.n_runs:
        raise ValueError(
            f"blocks length ({labels.shape[0]}) must match n_runs ({design.n_runs})"
        )
    mat = design.matrix.copy()
    mat[name] = labels
    meta = dict(design.metadata)
    meta["blocking"] = {"column": name, "n_blocks": int(len(pd.unique(pd.Series(labels))))}
    return Design(matrix=mat, factors=list(design.factors or []),
                  model=design.model, metadata=meta)


# ---------------------------------------------------------------------------
# OLS / ANOVA / LOF
# ---------------------------------------------------------------------------

