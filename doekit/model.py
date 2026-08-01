"""Mini-DSL of model terms replacing ``StatsModels.@formula``.

This is the cross-cutting enabler of optimal design and analysis: it turns a
model specification (formula-like string or programmatic terms) plus a
``DataFrame`` of levels into a **model matrix** ``X`` (numpy) on which the
optimality criteria are evaluated and linear models are fit.

Supported string syntax (RHS)::

    "y ~ x1 + x2 + x1:x2 + x3^2"     # response y; implicit intercept
    "0 ~ x1 + x2 + x1:x2"            # no response; implicit intercept
    "y ~ -1 + x1 + x2"               # no intercept (as in Plackett-Burman)

Rules (StatsModels-compatible): the intercept is added by default unless the
RHS contains ``-1`` or ``0``. ``a:b`` is an interaction (product of columns)
and ``a^k`` is a power.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# --- Terms -----------------------------------------------------------------

@dataclass(frozen=True)
class Intercept:
    """The constant (intercept) term of a model."""

    def label(self) -> str:
        """Return the column label ``"(Intercept)"``."""
        return "(Intercept)"

    def to_dict(self) -> dict:
        """Serialize to ``{"kind": "intercept"}``."""
        return {"kind": "intercept"}


@dataclass(frozen=True)
class Main:
    """A main-effect term for a single factor ``name``."""

    name: str

    def label(self) -> str:
        """Return the factor name as column label."""
        return self.name

    def to_dict(self) -> dict:
        """Serialize to ``{"kind": "main", "name"}``."""
        return {"kind": "main", "name": self.name}


@dataclass(frozen=True)
class Interaction:
    """An interaction term: the product of the columns in ``names``."""

    names: tuple

    def label(self) -> str:
        """Return the ``a:b[:c...]`` column label."""
        return ":".join(self.names)

    def to_dict(self) -> dict:
        """Serialize to ``{"kind": "interaction", "names"}``."""
        return {"kind": "interaction", "names": list(self.names)}


@dataclass(frozen=True)
class Power:
    """A polynomial term ``name`` raised to integer ``degree`` (e.g. ``x^2``)."""

    name: str
    degree: int

    def label(self) -> str:
        """Return the ``name^degree`` column label."""
        return f"{self.name}^{self.degree}"

    def to_dict(self) -> dict:
        """Serialize to ``{"kind": "power", "name", "degree"}``."""
        return {"kind": "power", "name": self.name, "degree": self.degree}


Term = object  # Intercept | Main | Interaction | Power


def _term_from_dict(d: dict) -> Term:
    """Rebuild a term from its ``to_dict`` output; raise on an unknown kind."""
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


# --- Model -----------------------------------------------------------------

class Model:
    """An ordered set of terms able to build a model matrix ``X``.

    Parameters
    ----------
    terms : sequence of term
        The model terms (``Intercept``, ``Main``, ``Interaction``, ``Power``).
    response : str, optional
        Name of the response column, if any.
    """

    def __init__(self, terms: Sequence[Term], response: Optional[str] = None):
        self.terms = list(terms)
        self.response = response

    # -- construction --

    @classmethod
    def parse(cls, formula: str) -> "Model":
        """Parse a formula-like string such as ``"y ~ x1 + x2 + x1:x2 + x3^2"``.

        Parameters
        ----------
        formula : str
            Formula with an optional ``response ~`` left-hand side. ``a:b`` is an
            interaction, ``a^k`` a power. The intercept is implicit unless the
            right-hand side contains ``-1`` or ``0``.

        Returns
        -------
        Model
            The parsed model.
        """
        if "~" in formula:
            lhs, rhs = formula.split("~", 1)
            lhs = lhs.strip()
            response = None if lhs in ("", "0", "1") else lhs
        else:
            rhs, response = formula, None

        tokens = [t.strip() for t in rhs.split("+") if t.strip()]
        add_intercept = True
        terms: list[Term] = []
        for tok in tokens:
            if tok in ("-1", "0"):
                add_intercept = False
                continue
            if tok == "1":
                continue  # explicit intercept, already handled
            if ":" in tok:
                names = tuple(p.strip() for p in tok.split(":"))
                terms.append(Interaction(names))
            elif "^" in tok:
                base, deg = tok.split("^", 1)
                terms.append(Power(base.strip(), int(deg)))
            else:
                terms.append(Main(tok))

        if add_intercept:
            terms.insert(0, Intercept())
        return cls(terms, response=response)

    @classmethod
    def from_terms(cls, terms: Sequence[Term], response: Optional[str] = None,
                   intercept: bool = True) -> "Model":
        """Build a model from explicit ``terms``, prepending an intercept if asked.

        Parameters
        ----------
        terms : sequence of term
            The model terms.
        response : str, optional
            Name of the response column.
        intercept : bool, default True
            Prepend an :class:`Intercept` unless one is already present.

        Returns
        -------
        Model
        """
        terms = list(terms)
        if intercept and not any(isinstance(t, Intercept) for t in terms):
            terms.insert(0, Intercept())
        return cls(terms, response=response)

    @classmethod
    def full_quadratic(cls, factor_names: Sequence[str],
                       intercept: bool = True) -> "Model":
        """Full RSM model: intercept + linear + 2-way interactions + quadratics.

        Standard full second-order RSM model used with Box-Behnken / CCD designs.

        Parameters
        ----------
        factor_names : sequence of str
            Base factor names.
        intercept : bool, default True
            Whether to include an intercept term.

        Returns
        -------
        Model
        """
        terms: list[Term] = []
        terms.extend(Main(n) for n in factor_names)
        terms.extend(Interaction((a, b)) for a, b in combinations(factor_names, 2))
        terms.extend(Power(n, 2) for n in factor_names)
        return cls.from_terms(terms, intercept=intercept)

    @classmethod
    def main_effects(cls, factor_names: Sequence[str],
                     intercept: bool = True) -> "Model":
        """Main-effects-only model (intercept + one term per factor).

        Parameters
        ----------
        factor_names : sequence of str
            Base factor names.
        intercept : bool, default True
            Whether to include an intercept term.

        Returns
        -------
        Model
        """
        return cls.from_terms([Main(n) for n in factor_names], intercept=intercept)

    # -- evaluation --

    def _column(self, df: pd.DataFrame, term: Term) -> tuple[list[str], np.ndarray]:
        """Build the column block (names, values) contributed by a single term."""
        n = len(df)
        if isinstance(term, Intercept):
            return ["(Intercept)"], np.ones((n, 1))
        if isinstance(term, Main):
            col = df[term.name]
            if col.dtype == object or str(col.dtype).startswith("category"):
                # dummy coding (first level as reference)
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

    def column_names(self, df: pd.DataFrame) -> list[str]:
        """Return the column names of ``X`` for the given level ``DataFrame``."""
        names: list[str] = []
        for term in self.terms:
            names.extend(self._column(df, term)[0])
        return names

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build the model matrix ``X`` from a ``DataFrame`` of factor levels."""
        blocks = [self._column(df, term)[1] for term in self.terms]
        if not blocks:
            return np.empty((len(df), 0))
        return np.hstack(blocks)

    @property
    def factor_names(self) -> list[str]:
        """Base factor names referenced by the model (intercept excluded)."""
        seen: list[str] = []
        for t in self.terms:
            names = []
            if isinstance(t, Main):
                names = [t.name]
            elif isinstance(t, Power):
                names = [t.name]
            elif isinstance(t, Interaction):
                names = list(t.names)
            for nm in names:
                if nm not in seen:
                    seen.append(nm)
        return seen

    def __repr__(self) -> str:
        rhs = " + ".join(t.label() for t in self.terms)
        lhs = self.response or "0"
        return f"Model({lhs} ~ {rhs})"

    # -- serialization (for MCP / persistence) --

    def to_dict(self) -> dict:
        """Serialize to ``{"response", "terms": [...]}``."""
        return {"response": self.response, "terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        """Rebuild a model from its :meth:`to_dict` output."""
        terms = [_term_from_dict(t) for t in d.get("terms", [])]
        return cls(terms, response=d.get("response"))
