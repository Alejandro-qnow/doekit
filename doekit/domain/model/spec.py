"""Model specification: parsing, constructors, serialization."""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import pandas as pd

from .terms import Intercept, Main, Interaction, Power, Term, term_from_dict
from . import matrix as _matrix


class Model:
    """An ordered set of terms that builds a model matrix ``X``.

    Terms (:class:`Intercept`, :class:`Main`, :class:`Interaction`,
    :class:`Power`) define columns of ``X`` from a run matrix. Construct via
    :meth:`parse`, :meth:`from_terms`, or presets (:meth:`full_quadratic`,
    :meth:`main_effects`, Scheffé helpers).

    Parameters
    ----------
    terms : sequence of Term
        Ordered list of model terms (intercept first when present).
    response : str, optional
        Response variable name (metadata only; not used in matrix construction).

    Examples
    --------
    >>> import doekit as ed
    >>> m = ed.Model.parse("y ~ x1 + x2 + x1:x2")
    >>> "x1:x2" in [t.label() for t in m.terms]
    True
    """

    def __init__(self, terms: Sequence[Term], response: Optional[str] = None):
        self.terms = list(terms)
        self.response = response

    @classmethod
    def parse(cls, formula: str) -> "Model":
        """Parse a formula-like string into a :class:`Model`.

        Supports ``+`` for main effects, ``:`` for interactions, ``^`` for
        powers, and ``-1`` / ``0`` to omit the intercept.

        Parameters
        ----------
        formula : str
            Formula such as ``"y ~ x1 + x2 + x1:x2 + x3^2"`` or ``"x1 + x2"``.

        Returns
        -------
        Model
            Parsed model with an intercept unless ``-1`` or ``0`` appears.

        Examples
        --------
        >>> import doekit as ed
        >>> ed.Model.parse("y ~ x1 + x2").factor_names
        ['x1', 'x2']
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
        """Build a model from an explicit term list.

        Parameters
        ----------
        terms : sequence of Term
            Model terms (intercept may be included explicitly).
        response : str, optional
            Response variable name.
        intercept : bool, default True
            Insert :class:`Intercept` when absent.

        Returns
        -------
        Model
            Model with the given terms.
        """
        terms = list(terms)
        if intercept and not any(isinstance(t, Intercept) for t in terms):
            terms.insert(0, Intercept())
        return cls(terms, response=response)

    @classmethod
    def full_quadratic(cls, factor_names: Sequence[str],
                       intercept: bool = True) -> "Model":
        """Full quadratic response-surface model (main + interactions + squares).

        Parameters
        ----------
        factor_names : sequence of str
            Factor names for mains, pairwise interactions, and ``^2`` terms.
        intercept : bool, default True
            Include an intercept.

        Returns
        -------
        Model
            Model with all main effects, two-factor interactions, and pure
            quadratics.

        Examples
        --------
        >>> import doekit as ed
        >>> m = ed.Model.full_quadratic(["A", "B"])
        >>> len(m.terms) >= 5  # intercept + 2 mains + interaction + 2 squares
        True
        """
        terms: list[Term] = []
        terms.extend(Main(n) for n in factor_names)
        terms.extend(Interaction((a, b)) for a, b in combinations(factor_names, 2))
        terms.extend(Power(n, 2) for n in factor_names)
        return cls.from_terms(terms, intercept=intercept)

    @classmethod
    def main_effects(cls, factor_names: Sequence[str],
                     intercept: bool = True) -> "Model":
        """Main-effects-only model.

        Parameters
        ----------
        factor_names : sequence of str
            Factor names for main-effect columns.
        intercept : bool, default True
            Include an intercept.

        Returns
        -------
        Model
            Model with one :class:`Main` term per factor.

        Examples
        --------
        >>> import doekit as ed
        >>> ed.Model.main_effects(["x1", "x2"], intercept=False).terms[0]
        Main(name='x1')
        """
        return cls.from_terms([Main(n) for n in factor_names], intercept=intercept)

    @classmethod
    def scheffe_linear(cls, factor_names: Sequence[str]) -> "Model":
        """Scheffé linear mixture model (no intercept).

        Formulas
        --------
        ``y = sum_i beta_i x_i`` with ``sum_i x_i = 1``.

        Parameters
        ----------
        factor_names : sequence of str
            Mixture component names.

        Returns
        -------
        Model
            Linear Scheffé model without intercept.
        """
        return cls.from_terms([Main(n) for n in factor_names], intercept=False)

    @classmethod
    def scheffe_quadratic(cls, factor_names: Sequence[str]) -> "Model":
        """Scheffé quadratic mixture model (no intercept).

        Formulas
        --------
        Linear Scheffé terms plus cross products ``x_i x_j`` for ``i < j``.

        Parameters
        ----------
        factor_names : sequence of str
            Mixture component names.

        Returns
        -------
        Model
            Quadratic Scheffé model without intercept.
        """
        terms: list[Term] = [Main(n) for n in factor_names]
        terms.extend(Interaction((a, b)) for a, b in combinations(factor_names, 2))
        return cls.from_terms(terms, intercept=False)

    def column_names(self, df: pd.DataFrame) -> list[str]:
        """Return model-matrix column labels for a run frame.

        Parameters
        ----------
        df : DataFrame
            Run matrix whose columns supply factor values.

        Returns
        -------
        list of str
            One label per model column (matches :meth:`matrix` column order).
        """
        return _matrix.column_names(self.terms, df)

    def matrix(self, df: pd.DataFrame):
        """Build the model matrix ``X`` from a run DataFrame.

        Parameters
        ----------
        df : DataFrame
            Run matrix with columns for each factor referenced by ``terms``.

        Returns
        -------
        ndarray, shape (n_runs, n_params)
            Model matrix ``X``.
        """
        return _matrix.build_matrix(self.terms, df)

    @property
    def factor_names(self) -> list[str]:
        """Unique factor names referenced by all terms (order of first appearance)."""
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
        """Serialize the model to a plain dict.

        Returns
        -------
        dict
            Keys ``response`` and ``terms`` (each term's :meth:`~Term.to_dict`).
        """
        return {"response": self.response, "terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        """Rebuild a :class:`Model` from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict
            Serialized model.

        Returns
        -------
        Model
            Restored model instance.
        """
        terms = [term_from_dict(t) for t in d.get("terms", [])]
        return cls(terms, response=d.get("response"))
