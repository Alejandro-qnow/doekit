"""Natural <-> coded frame transforms (single coding policy)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def encode_frame(matrix: pd.DataFrame, factors: Sequence | None) -> pd.DataFrame:
    """Return a copy of ``matrix`` with continuous/discrete columns coded to ±1.

    Categorical factors are left raw (the model dummy-codes them). Mixture
    proportions stay as-is (identity coding for Scheffé). Columns with no
    associated factor are assumed already coded.
    """
    df = matrix.copy()
    for f in (factors or []):
        if f.name not in df.columns:
            continue
        if getattr(f, "is_categorical", False) or getattr(f, "is_mixture", False):
            # mixture: encode is identity; still apply for consistency
            if getattr(f, "is_mixture", False):
                df[f.name] = np.asarray(f.encode(df[f.name].to_numpy()), dtype=float)
            continue
        df[f.name] = np.asarray(f.encode(df[f.name].to_numpy()), dtype=float)
    return df


def decode_frame(matrix: pd.DataFrame, factors: Sequence | None) -> pd.DataFrame:
    """Return a copy of ``matrix`` with coded columns decoded to natural units."""
    df = matrix.copy()
    for f in (factors or []):
        if f.name in df.columns:
            df[f.name] = f.decode(df[f.name].to_numpy())
    return df
