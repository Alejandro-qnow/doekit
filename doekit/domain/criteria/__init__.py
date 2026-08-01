"""Optimality criteria over the model matrix ``X`` (``N x p``).

Default numerical tolerance ``1e-9``. Convention: **larger is better** for every
function, so that ``optimal_design`` always maximizes the chosen criterion.
"""

from __future__ import annotations

from .base import CriterionContext, score_criterion
from .functions import (
    TOLERANCE,
    d_criterion,
    a_criterion,
    t_criterion,
    g_criterion,
    e_criterion,
    i_criterion,
    CRITERIA,
    get_criterion,
    all_criteria,
)
from .linalg import info_matrix, leverage, inv_info

__all__ = [
    "TOLERANCE",
    "CriterionContext",
    "score_criterion",
    "d_criterion",
    "a_criterion",
    "t_criterion",
    "g_criterion",
    "e_criterion",
    "i_criterion",
    "CRITERIA",
    "get_criterion",
    "all_criteria",
    "info_matrix",
    "leverage",
    "inv_info",
]
