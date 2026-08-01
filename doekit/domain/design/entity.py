"""Design entity: run matrix, factors, model and typed metadata."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd

from ..model import Model
from ...shared.serialize import jsonify


@dataclass
class Design:
    """A generated experimental design.

    Prefer :meth:`replace` / constructing a new instance over mutating fields
    in place so callers keep stable references.
    """

    matrix: pd.DataFrame
    factors: list = field(default_factory=list)
    model: Optional[Model] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.matrix)

    @property
    def n_factors(self) -> int:
        return self.matrix.shape[1]

    @property
    def factor_names(self) -> list[str]:
        return list(self.matrix.columns)

    def model_matrix(self):
        if self.model is None:
            raise ValueError("the design has no associated model")
        return self.model.matrix(self.matrix)

    def replace(self, **kwargs) -> "Design":
        """Return a new Design with selected fields replaced (copy-on-write)."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        from ..factors import Factor  # noqa: PLC0415
        return {
            "schema": "doekit.Design/1",
            "matrix": _json.loads(self.matrix.to_json(orient="split", index=False)),
            "factors": [f.to_dict() for f in (self.factors or [])
                        if isinstance(f, Factor)],
            "model": self.model.to_dict() if self.model is not None else None,
            "metadata": jsonify(dict(self.metadata)),
        }

    def to_json(self, **kwargs) -> str:
        return _json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        from ..factors import factor_from_dict  # noqa: PLC0415
        mat = d["matrix"]
        df = pd.DataFrame(mat["data"], columns=mat["columns"])
        factors = [factor_from_dict(fd) for fd in d.get("factors", [])]
        model = Model.from_dict(d["model"]) if d.get("model") else None
        return cls(matrix=df, factors=factors, model=model,
                   metadata=d.get("metadata", {}))

    @classmethod
    def from_json(cls, s: str) -> "Design":
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
