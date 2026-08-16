"""Frozen term dataclasses for the model DSL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intercept:
    """Constant (intercept) column of ones in the model matrix.

    Appears as ``(Intercept)`` in column labels. Omitted in Scheffé mixture
    models where the constraint ``sum x_i = 1`` replaces the intercept.

    Examples
    --------
    >>> import doekit as ed
    >>> ed.Intercept().label()
    '(Intercept)'
    """

    def label(self) -> str:
        return "(Intercept)"

    def to_dict(self) -> dict:
        return {"kind": "intercept"}


@dataclass(frozen=True)
class Main:
    """Main-effect term for a single factor.

    The model column is the factor column from the run matrix (after any
    coding applied upstream).

    Parameters
    ----------
    name : str
        Factor name (must match a run-matrix column).

    Examples
    --------
    >>> import doekit as ed
    >>> ed.Main("temperature").label()
    'temperature'
    """

    name: str

    def label(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return {"kind": "main", "name": self.name}


@dataclass(frozen=True)
class Interaction:
    """Interaction term: element-wise product of factor columns.

    Parameters
    ----------
    names : tuple of str
        Factor names whose columns are multiplied (order preserved in label).

    Examples
    --------
    >>> import doekit as ed
    >>> ed.Interaction(("A", "B")).label()
    'A:B'
    """

    names: tuple

    def label(self) -> str:
        return ":".join(self.names)

    def to_dict(self) -> dict:
        return {"kind": "interaction", "names": list(self.names)}


@dataclass(frozen=True)
class Power:
    """Polynomial term: a factor column raised to an integer degree.

    Parameters
    ----------
    name : str
        Factor name.
    degree : int
        Exponent (typically ``2`` for pure quadratic terms).

    Examples
    --------
    >>> import doekit as ed
    >>> ed.Power("x", 2).label()
    'x^2'
    """

    name: str
    degree: int

    def label(self) -> str:
        return f"{self.name}^{self.degree}"

    def to_dict(self) -> dict:
        return {"kind": "power", "name": self.name, "degree": self.degree}


Term = object  # Intercept | Main | Interaction | Power


def term_from_dict(d: dict) -> Term:
    """Rebuild a term from its :meth:`to_dict` output."""
    k = d.get("kind")
    if k == "intercept":
        return Intercept()
    if k == "main":
        return Main(d["name"])
    if k == "interaction":
        return Interaction(tuple(d["names"]))
    if k == "power":
        return Power(d["name"], int(d["degree"]))
    raise ValueError(f"unknown term: {k!r}")
