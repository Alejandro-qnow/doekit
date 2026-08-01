"""Cross-cutting utilities (serialization, errors, typing)."""

from .serialize import jsonify, as_float_list
from .errors import (
    DoEError,
    InapplicableDesign,
    SingularMatrixError,
    RankDeficientError,
    UnknownCriterionError,
    UnknownFactorTypeError,
)
from .typing import Space

__all__ = [
    "jsonify",
    "as_float_list",
    "DoEError",
    "InapplicableDesign",
    "SingularMatrixError",
    "RankDeficientError",
    "UnknownCriterionError",
    "UnknownFactorTypeError",
    "Space",
]
