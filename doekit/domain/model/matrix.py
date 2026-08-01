"""Build numeric model-matrix columns from terms + a level DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .terms import Intercept, Main, Interaction, Power, Term


def column_block(df: pd.DataFrame, term: Term) -> tuple[list[str], np.ndarray]:
    """Build the column block (names, values) contributed by a single term."""
    n = len(df)
    if isinstance(term, Intercept):
        return ["(Intercept)"], np.ones((n, 1))
    if isinstance(term, Main):
        col = df[term.name]
        dtype_name = str(col.dtype)
        if (col.dtype == object or dtype_name.startswith("category")
                or dtype_name in ("string", "str")
                or (hasattr(col.dtype, "kind") and col.dtype.kind in ("O", "U", "S"))):
            dummies = pd.get_dummies(col, prefix=term.name, drop_first=True)
            return list(dummies.columns), dummies.to_numpy(dtype=float)
        return [term.name], col.to_numpy(dtype=float).reshape(-1, 1)
    if isinstance(term, Power):
        v = df[term.name].to_numpy(dtype=float) ** term.degree
        return [term.label()], v.reshape(-1, 1)
    if isinstance(term, Interaction):
        v = np.ones(n)
        for name in term.names:
            v = v * df[name].to_numpy(dtype=float)
        return [term.label()], v.reshape(-1, 1)
    raise TypeError(f"unsupported term: {term!r}")


def build_matrix(terms: list[Term], df: pd.DataFrame) -> np.ndarray:
    """Stack term blocks into the model matrix ``X``."""
    blocks = [column_block(df, term)[1] for term in terms]
    if not blocks:
        return np.empty((len(df), 0))
    return np.hstack(blocks)


def column_names(terms: list[Term], df: pd.DataFrame) -> list[str]:
    """Return the column names of ``X`` for the given level ``DataFrame``."""
    names: list[str] = []
    for term in terms:
        names.extend(column_block(df, term)[0])
    return names
