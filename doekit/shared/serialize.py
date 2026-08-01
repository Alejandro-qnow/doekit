"""Versioned JSON helpers shared by Design / FitResult / evaluation DTOs."""

from __future__ import annotations

import numpy as np


def jsonify(obj):
    """Convert nested numpy/pandas scalars to native JSON-safe types.

    NaN/Inf float scalars become ``None`` so MCP and file codecs stay valid.
    """
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def as_float_list(a) -> list:
    """JSON-safe list of floats (NaN/Inf -> None)."""
    out = []
    for v in np.asarray(a, dtype=float).reshape(-1):
        fv = float(v)
        out.append(None if (np.isnan(fv) or np.isinf(fv)) else fv)
    return out
