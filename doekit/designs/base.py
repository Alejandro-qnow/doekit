"""Common container of an experimental design.

A lightweight ``Design`` holds the run matrix in natural units, the factors, the
prior model and metadata for any constructor in this package.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..model import Model


def _jsonify(obj):
    """Convert nested numpy/pandas types to native JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    return obj


@dataclass
class Design:
    """A generated experimental design.

    Attributes
    ----------
    matrix : pandas.DataFrame
        The run matrix in natural units (one row per run, one column per factor).
    factors : list of Factor
        The factors, carrying the natural<->coded conversion.
    model : Model, optional
        The prior/analysis model associated with the design.
    metadata : dict
        Construction metadata (``kind``, ``resolution``, ``alpha``, ...).
    """

    matrix: pd.DataFrame
    factors: list = field(default_factory=list)
    model: Optional[Model] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        """Number of runs (rows) in the design matrix."""
        return len(self.matrix)

    @property
    def n_factors(self) -> int:
        """Number of factors (columns) in the design matrix."""
        return self.matrix.shape[1]

    @property
    def factor_names(self) -> list[str]:
        """Column names of the design matrix."""
        return list(self.matrix.columns)

    def model_matrix(self):
        """Return the model matrix ``X`` built from ``self.model``.

        Raises
        ------
        ValueError
            If the design has no associated model.
        """
        if self.model is None:
            raise ValueError("the design has no associated model")
        return self.model.matrix(self.matrix)

    # -- serialization (for MCP / persistence) --

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (matrix, factors, model, metadata)."""
        from ..factors import Factor  # noqa: PLC0415
        return {
            "schema": "doekit.Design/1",
            "matrix": _json.loads(self.matrix.to_json(orient="split", index=False)),
            "factors": [f.to_dict() for f in (self.factors or [])
                        if isinstance(f, Factor)],
            "model": self.model.to_dict() if self.model is not None else None,
            "metadata": _jsonify(dict(self.metadata)),
        }

    def to_json(self, **kwargs) -> str:
        """Serialize the design to a JSON string (kwargs go to ``json.dumps``)."""
        return _json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        """Rebuild a :class:`Design` from its :meth:`to_dict` output."""
        from ..factors import factor_from_dict  # noqa: PLC0415
        mat = d["matrix"]
        df = pd.DataFrame(mat["data"], columns=mat["columns"])
        factors = [factor_from_dict(fd) for fd in d.get("factors", [])]
        model = Model.from_dict(d["model"]) if d.get("model") else None
        return cls(matrix=df, factors=factors, model=model,
                   metadata=d.get("metadata", {}))

    @classmethod
    def from_json(cls, s: str) -> "Design":
        """Rebuild a :class:`Design` from a JSON string."""
        return cls.from_dict(_json.loads(s))

    def __repr__(self) -> str:
        kind = self.metadata.get("kind", "Design")
        lines = [
            kind,
            f"Dimension: ({self.n_runs}, {self.n_factors})",
            f"Factors: {self.factor_names}",
        ]
        if self.model is not None:
            lines.append(f"Model: {self.model!r}")
        for key in ("criteria", "generators", "resolution", "alpha", "face"):
            if key in self.metadata:
                lines.append(f"{key.capitalize()}: {self.metadata[key]}")
        lines.append("Design Matrix:")
        lines.append(repr(self.matrix))
        return "\n".join(lines)
