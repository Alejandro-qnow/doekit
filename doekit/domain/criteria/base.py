"""Uniform criterion interface so D/A/T/G/E/I are substitutable (Liskov)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class CriterionContext:
    """Optional extras a criterion may need (e.g. I-optimality moments)."""

    moment_matrix: Optional[np.ndarray] = None
    tolerance: Optional[float] = None


class Criterion(Protocol):
    """Callable optimality score: larger is better."""

    def __call__(
        self,
        model_matrix: np.ndarray,
        context: CriterionContext | None = None,
    ) -> float: ...


def score_criterion(
    fn: Callable,
    model_matrix: np.ndarray,
    context: CriterionContext | None = None,
) -> float:
    """Invoke a criterion with a uniform ``(X, context)`` contract.

    Legacy callables that only accept ``X`` (or ``X, moment_matrix=...`` for I)
    are adapted here so search/evaluation never special-case signatures.
    """
    ctx = context or CriterionContext()
    name = getattr(fn, "__name__", "")
    if name == "i_criterion" or getattr(fn, "_needs_moments", False):
        kwargs = {}
        if ctx.tolerance is not None:
            kwargs["tolerance"] = ctx.tolerance
        return float(fn(model_matrix, moment_matrix=ctx.moment_matrix, **kwargs))
    if ctx.tolerance is not None:
        try:
            return float(fn(model_matrix, tolerance=ctx.tolerance))
        except TypeError:
            pass
    return float(fn(model_matrix))
