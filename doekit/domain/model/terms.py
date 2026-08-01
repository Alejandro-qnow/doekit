"""Frozen term dataclasses for the model DSL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intercept:
    """The constant (intercept) term of a model."""

    def label(self) -> str:
        return "(Intercept)"

    def to_dict(self) -> dict:
        return {"kind": "intercept"}


@dataclass(frozen=True)
class Main:
    """A main-effect term for a single factor ``name``."""

    name: str

    def label(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return {"kind": "main", "name": self.name}


@dataclass(frozen=True)
class Interaction:
    """An interaction term: the product of the columns in ``names``."""

    names: tuple

    def label(self) -> str:
        return ":".join(self.names)

    def to_dict(self) -> dict:
        return {"kind": "interaction", "names": list(self.names)}


@dataclass(frozen=True)
class Power:
    """A polynomial term ``name`` raised to integer ``degree``."""

    name: str
    degree: int

    def label(self) -> str:
        return f"{self.name}^{self.degree}"

    def to_dict(self) -> dict:
        return {"kind": "power", "name": self.name, "degree": self.degree}


Term = object  # Intercept | Main | Interaction | Power


def term_from_dict(d: dict) -> Term:
    """Rebuild a term from its ``to_dict`` output."""
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
