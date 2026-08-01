"""First-class experimental constraints (region shape, hard-to-change, cost)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Native constraint bundle for advise / optimal / sequential.

    Parameters
    ----------
    mixture : bool, default False
        Force mixture / simplex shortlist in the advisor (also inferred from
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
        Replaces the old ``constrained=True`` flag.
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
        return bool(self.split_plot or self.hard_to_change)

    def filter_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows rejected by ``exclude`` (if any)."""
        if self.exclude is None or df.empty:
            return df
        keep = [not self.exclude(row) for _, row in df.iterrows()]
        return df.loc[keep].reset_index(drop=True)

    def to_dict(self) -> dict:
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
    """Normalize ``constraints`` / legacy ``constrained`` into a :class:`Constraints`.

    ``constrained=True`` maps to ``Constraints(irregular=True)`` with a soft
    deprecation path for callers that still pass the boolean.
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
