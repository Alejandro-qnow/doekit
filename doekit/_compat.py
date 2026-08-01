"""Backward-compatibility shims (deprecated APIs).

The ``report=`` side-channel on ``evaluate`` / ``fit_linear_model`` /
``optimal_design`` inverted the dependency rule (domain importing presentation).
Callers should use :func:`doekit.report` (or ``presentation.report``) explicitly.
"""

from __future__ import annotations

import warnings
from typing import Any


def maybe_report(design, report=None, *, response=None, model=None, **extra) -> Any:
    """Deprecated bridge for the ``report=`` argument.

    Returns the written path or ``None``. Emits :class:`DeprecationWarning`.
    """
    if report is None or report is False:
        return None
    warnings.warn(
        "Passing report= to evaluate/fit_linear_model/optimal_design is deprecated; "
        "call doekit.report(design, ...) explicitly instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    from .presentation.report import run_report_arg  # noqa: PLC0415
    return run_report_arg(design, response=response, model=model, report=report, **extra)
