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
    """A generated experimental design with run matrix, factors, and model.

    The run matrix holds factor levels (natural or coded units depending on
    generation). Optional :class:`~doekit.domain.factors.Factor` objects and a
    :class:`~doekit.domain.model.Model` enable coding, model-matrix construction,
    and evaluation. Metadata carries design-kind tags (resolution, generators,
    etc.).

    Prefer :meth:`replace` or constructing a new instance over mutating fields
    in place so callers keep stable references.

    Parameters
    ----------
    matrix : DataFrame
        Run-by-factor table; column names are factor names.
    factors : list, optional
        Typed factor objects for encode/decode (may be empty).
    model : Model, optional
        Model attached for matrix construction and evaluation.
    metadata : dict, optional
        Free-form tags (``kind``, ``resolution``, ``generators``, etc.).

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.full_factorial(2)
    >>> d.n_runs, d.n_factors
    (4, 2)
    """

    matrix: pd.DataFrame
    factors: list = field(default_factory=list)
    model: Optional[Model] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        """Number of experimental runs (rows in ``matrix``)."""
        return len(self.matrix)

    @property
    def n_factors(self) -> int:
        """Number of factor columns in ``matrix``."""
        return self.matrix.shape[1]

    @property
    def factor_names(self) -> list[str]:
        """Column names of the run matrix."""
        return list(self.matrix.columns)

    def model_matrix(self):
        """Build the model matrix ``X`` from the run matrix and attached model.

        Returns
        -------
        ndarray, shape (n_runs, n_params)
            Model matrix in the units implied by ``matrix`` and ``model``.

        Raises
        ------
        ValueError
            When no model is attached.
        """
        if self.model is None:
            raise ValueError("the design has no associated model")
        return self.model.matrix(self.matrix)

    def replace(self, **kwargs) -> "Design":
        """Return a new Design with selected fields replaced (copy-on-write).

        Parameters
        ----------
        **kwargs
            Fields to override (``matrix``, ``factors``, ``model``, ``metadata``).

        Returns
        -------
        Design
            A new instance; the original is unchanged.

        Examples
        --------
        >>> import doekit as ed
        >>> d = ed.full_factorial(2)
        >>> d2 = d.replace(metadata={"note": "copy"})
        >>> d2.metadata["note"]
        'copy'
        """
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Serialize the design to a JSON-compatible dict.

        Returns
        -------
        dict
            Schema ``doekit.Design/1`` with ``matrix``, ``factors``, ``model``,
            and ``metadata`` keys.
        """
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
        """Serialize the design to a JSON string.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`json.dumps` (e.g. ``indent=2``).

        Returns
        -------
        str
            JSON representation of :meth:`to_dict`.
        """
        return _json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        """Rebuild a :class:`Design` from a :meth:`to_dict` payload.

        Parameters
        ----------
        d : dict
            Serialized design (schema ``doekit.Design/1``).

        Returns
        -------
        Design
            Restored design instance.
        """
        from ..factors import factor_from_dict  # noqa: PLC0415
        mat = d["matrix"]
        df = pd.DataFrame(mat["data"], columns=mat["columns"])
        factors = [factor_from_dict(fd) for fd in d.get("factors", [])]
        model = Model.from_dict(d["model"]) if d.get("model") else None
        return cls(matrix=df, factors=factors, model=model,
                   metadata=d.get("metadata", {}))

    @classmethod
    def from_json(cls, s: str) -> "Design":
        """Rebuild a :class:`Design` from a JSON string.

        Parameters
        ----------
        s : str
            JSON from :meth:`to_json`.

        Returns
        -------
        Design
            Restored design instance.
        """
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
