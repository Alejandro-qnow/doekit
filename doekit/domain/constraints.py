"""First-class experimental constraints (region shape, hard-to-change, cost)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Native constraint bundle for advise, optimal, and sequential workflows.

    Flags steer design recommendation (mixture, split-plot, irregular region)
    and optional row filtering via ``exclude``. Also inferred from factor types
    when factors carry mixture or hard-to-change metadata.

    Parameters
    ----------
    mixture : bool, default False
        Prefer mixture / simplex designs in the advisor (also inferred from
        :class:`~doekit.domain.factors.MixtureFactor`).
    hard_to_change : sequence of str, optional
        Factor names that are hard to change → split-plot generation.
    split_plot : bool, default False
        Force split-plot shortlist even without explicit hard-to-change names.
    run_cost : float, default 1.0
        Relative cost per run (used by compare / sequential heuristics).
    exclude : callable, optional
        ``f(row: pd.Series) -> bool``; ``True`` means the point is forbidden.
        Applied when filtering candidate sets.
    irregular : bool, default False
        Irregular / non-rectangular region → prefer D-optimal over RSM templates.
        Replaces the legacy ``constrained=True`` flag.

    Examples
    --------
    >>> import doekit as ed
    >>> c = ed.Constraints(mixture=True)
    >>> c.mixture
    True
    """

    mixture: bool = False
    hard_to_change: Sequence[str] = field(default_factory=tuple)
    split_plot: bool = False
    run_cost: float = 1.0
    exclude: Optional[Callable[[pd.Series], bool]] = None
    irregular: bool = False

    def __post_init__(self):
        self.hard_to_change = tuple(self.hard_to_change or ())

    @property
    def wants_split_plot(self) -> bool:
        """Whether split-plot generation is requested."""
        return bool(self.split_plot or self.hard_to_change)

    def filter_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows rejected by ``exclude`` (if any).

        Parameters
        ----------
        df : DataFrame
            Candidate run table.

        Returns
        -------
        DataFrame
            Filtered copy; unchanged when ``exclude`` is ``None``.
        """
        if self.exclude is None or df.empty:
            return df
        keep = [not self.exclude(row) for _, row in df.iterrows()]
        return df.loc[keep].reset_index(drop=True)

    def to_dict(self) -> dict:
        """Serialize constraints to a JSON-safe dict (``exclude`` as flag only).

        Returns
        -------
        dict
            Plain constraint fields; ``has_exclude`` indicates a callable.
        """
        return {
            "mixture": bool(self.mixture),
            "hard_to_change": list(self.hard_to_change),
            "split_plot": bool(self.split_plot),
            "run_cost": float(self.run_cost),
            "irregular": bool(self.irregular),
            "has_exclude": self.exclude is not None,
        }


def coerce_constraints(
    constraints: Union[Constraints, dict, None] = None,
    *,
    constrained: bool = False,
) -> Constraints:
    """Normalize ``constraints`` or legacy ``constrained`` into a :class:`Constraints`.

    ``constrained=True`` maps to ``Constraints(irregular=True)`` with a soft
    deprecation path for callers that still pass the boolean.

    Parameters
    ----------
    constraints : Constraints, dict, or None, optional
        Explicit constraints object or dict of field values.
    constrained : bool, default False
        Legacy flag equivalent to ``irregular=True``.

    Returns
    -------
    Constraints
        Normalized constraint bundle.

    Raises
    ------
    TypeError
        When ``constraints`` is neither :class:`Constraints`, dict, nor ``None``.

    Examples
    --------
    >>> import doekit as ed
    >>> ed.coerce_constraints({"mixture": True}).mixture
    True
    """
    if constraints is None:
        if constrained:
            return Constraints(irregular=True)
        return Constraints()
    if isinstance(constraints, Constraints):
        if constrained and not constraints.irregular:
            return Constraints(
                mixture=constraints.mixture,
                hard_to_change=constraints.hard_to_change,
                split_plot=constraints.split_plot,
                run_cost=constraints.run_cost,
                exclude=constraints.exclude,
                irregular=True,
            )
        return constraints
    if isinstance(constraints, dict):
        return Constraints(**constraints)
    raise TypeError(
        f"constraints must be Constraints, dict, or None; got {type(constraints)!r}"
    )
