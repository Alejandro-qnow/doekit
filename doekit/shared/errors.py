"""Domain-specific exceptions for doekit.

Prefer these over bare ``except Exception`` so inapplicable designs and
numerical failures stay distinguishable from programming bugs.
"""

from __future__ import annotations


class DoEError(Exception):
    """Base class for doekit domain errors."""


class InapplicableDesign(DoEError):
    """Raised (or returned as a typed outcome) when a design kind does not apply."""


class SingularMatrixError(DoEError):
    """Information / model matrix is singular."""


class RankDeficientError(DoEError):
    """Model matrix rank is less than the number of parameters."""


class UnknownCriterionError(DoEError, ValueError):
    """Unknown optimality criterion name."""


class UnknownFactorTypeError(DoEError, ValueError):
    """Unknown factor type in serialization / registry lookup."""
