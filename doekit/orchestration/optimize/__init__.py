"""Optimization engine: acquisition functions + Pareto utilities.

The ``optimize`` intent of :func:`doekit.propose_next_runs` uses these to turn a
:class:`~doekit.assessment.surrogate.Surrogate` into the *next batch of runs that
move the result*, complementing the classical (information-optimal) ``learn`` path.
"""

from .acquisition import (
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
    expected_hypervolume_improvement,
    get_acquisition,
)
from .pareto import (
    dominates,
    pareto_mask,
    pareto_front,
    hypervolume,
    to_max_space,
    default_reference,
)

__all__ = [
    "expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
    "expected_hypervolume_improvement",
    "get_acquisition",
    "dominates",
    "pareto_mask",
    "pareto_front",
    "hypervolume",
    "to_max_space",
    "default_reference",
]
