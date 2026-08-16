"""Versioned JSON helpers shared by Design, FitResult, and evaluation DTOs.

Recursively converts numpy and pandas types to native Python values suitable
for ``json.dumps``, wave artifacts, and MCP tool responses.
"""

from __future__ import annotations

import numpy as np


def jsonify(obj):
    """Convert nested numpy/pandas values to JSON-safe native types.

    DataFrames become lists of row dicts; Series become str-keyed dicts.
    NaN and Inf float scalars become ``None`` so file and MCP codecs stay valid.

    Parameters
    ----------
    obj : any
        Object to convert (dict, list, DataFrame, ndarray, scalar, etc.).

    Returns
    -------
    any
        JSON-serializable structure with the same nesting shape as ``obj``.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        pd = None

    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return jsonify(obj.to_dict("records"))
        if isinstance(obj, pd.Series):
            return jsonify({str(k): v for k, v in obj.items()})

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
    """Convert an array-like to a JSON-safe list of floats.

    NaN and Inf entries become ``None``.

    Parameters
    ----------
    a : array-like
        Values to flatten to a 1-D float list.

    Returns
    -------
    list
        Floats and ``None`` placeholders (one per element after ravel).
    """
    out = []
    for v in np.asarray(a, dtype=float).reshape(-1):
        fv = float(v)
        out.append(None if (np.isnan(fv) or np.isinf(fv)) else fv)
    return out
