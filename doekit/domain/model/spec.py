"""Model specification: parsing, constructors, serialization."""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import pandas as pd

from .terms import Intercept, Main, Interaction, Power, Term, term_from_dict
from . import matrix as _matrix


class Model:
    """An ordered set of terms able to build a model matrix ``X``."""

    def __init__(self, terms: Sequence[Term], response: Optional[str] = None):
        self.terms = list(terms)
        self.response = response

    @classmethod
    def parse(cls, formula: str) -> "Model":
        """Parse a formula-like string such as ``"y ~ x1 + x2 + x1:x2 + x3^2"``."""
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
                continue
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
        terms = list(terms)
        if intercept and not any(isinstance(t, Intercept) for t in terms):
            terms.insert(0, Intercept())
        return cls(terms, response=response)

    @classmethod
    def full_quadratic(cls, factor_names: Sequence[str],
                       intercept: bool = True) -> "Model":
        terms: list[Term] = []
        terms.extend(Main(n) for n in factor_names)
        terms.extend(Interaction((a, b)) for a, b in combinations(factor_names, 2))
        terms.extend(Power(n, 2) for n in factor_names)
        return cls.from_terms(terms, intercept=intercept)

    @classmethod
    def main_effects(cls, factor_names: Sequence[str],
                     intercept: bool = True) -> "Model":
        return cls.from_terms([Main(n) for n in factor_names], intercept=intercept)

    @classmethod
    def scheffe_linear(cls, factor_names: Sequence[str]) -> "Model":
        """Scheffé linear mixture model: ``sum β_i x_i`` (no intercept)."""
        return cls.from_terms([Main(n) for n in factor_names], intercept=False)

    @classmethod
    def scheffe_quadratic(cls, factor_names: Sequence[str]) -> "Model":
        """Scheffé quadratic: linear + cross products ``x_i:x_j`` (no intercept)."""
        terms: list[Term] = [Main(n) for n in factor_names]
        terms.extend(Interaction((a, b)) for a, b in combinations(factor_names, 2))
        return cls.from_terms(terms, intercept=False)

    def column_names(self, df: pd.DataFrame) -> list[str]:
        return _matrix.column_names(self.terms, df)

    def matrix(self, df: pd.DataFrame):
        return _matrix.build_matrix(self.terms, df)

    @property
    def factor_names(self) -> list[str]:
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

    def to_dict(self) -> dict:
        return {"response": self.response, "terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        terms = [term_from_dict(t) for t in d.get("terms", [])]
        return cls(terms, response=d.get("response"))
