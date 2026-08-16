"""Sequential monitoring: convergence detection and step diagnostics.

These feed the decision engine: :func:`check_convergence` produces a result whose
``should_stop`` / ``reason`` :func:`decide_next_action` reads as a hard stop, and
:func:`diagnose_step` flags per-wave problems (prediction degradation, thin power
gain, high uncertainty) that the core's fit-level diagnostics do not cover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ...shared.serialize import jsonify as _jsonify


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Whether a sequential run has stopped improving.

    Attributes
    ----------
    converged, should_stop : bool
        ``should_stop`` is what :func:`decide_next_action` consumes.
    reason : str
        Human-readable explanation.
    metric_key : str
        Which per-step value was tracked (e.g. ``"best_so_far"`` for optimize,
        ``"delta_D_efficiency"`` for learn).
    consecutive_hits : int
        Number of trailing steps whose marginal change was within tolerance.
    last_improvements : list of float
        The trailing marginal changes examined.
    observed_points : int
        How many values were extracted from the history.
    """

    converged: bool
    should_stop: bool
    reason: str
    metric_key: str
    marginal_threshold: float
    consecutive_required: int
    consecutive_hits: int
    last_improvements: list = field(default_factory=list)
    observed_points: int = 0

    def to_dict(self) -> dict:
        return _jsonify({
            "schema": "doekit.ConvergenceResult/1",
            "converged": self.converged,
            "should_stop": self.should_stop,
            "reason": self.reason,
            "metric_key": self.metric_key,
            "marginal_threshold": self.marginal_threshold,
            "consecutive_required": self.consecutive_required,
            "consecutive_hits": self.consecutive_hits,
            "last_improvements": list(self.last_improvements),
            "observed_points": self.observed_points,
        })


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_value(item, metric_key: str) -> Optional[float]:
    """Read ``metric_key`` from a scalar, a dict (top-level or ``metrics``), or an object."""
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        if metric_key in item:
            return _as_float(item[metric_key])
        metrics = item.get("metrics")
        if isinstance(metrics, dict) and metric_key in metrics:
            return _as_float(metrics[metric_key])
        return None
    if hasattr(item, metric_key):
        return _as_float(getattr(item, metric_key))
    metrics = getattr(item, "metrics", None)
    if isinstance(metrics, dict) and metric_key in metrics:
        return _as_float(metrics[metric_key])
    return None


def check_convergence(history: Iterable[Any], metric_key: str = "best_so_far",
                      marginal_threshold: float = 0.5,
                      consecutive_required: int = 2,
                      min_points: int = 3) -> ConvergenceResult:
    """Detect convergence by consecutive marginal changes within tolerance.

    Parameters
    ----------
    history : iterable
        Per-step values: scalars, dicts (``metric_key`` at top level or under
        ``"metrics"``), or objects exposing ``metric_key`` / ``.metrics``.
    metric_key : str, default "best_so_far"
        Tracked value. Use ``"best_so_far"`` for optimize, a delta metric for learn.
    marginal_threshold : float, default 0.5
        A step "does not improve" when ``|value_i - value_{i-1}| <= threshold``.
    consecutive_required : int, default 2
        Non-improving steps in a row needed to declare convergence.
    min_points : int, default 3
        Minimum history length before a verdict is issued.

    Returns
    -------
    ConvergenceResult
    """
    values = [v for v in (_extract_value(it, metric_key) for it in history) if v is not None]
    improvements = [values[i] - values[i - 1] for i in range(1, len(values))]

    if len(values) < min_points:
        return ConvergenceResult(
            converged=False, should_stop=False,
            reason="Insufficient history to assess convergence.",
            metric_key=metric_key, marginal_threshold=marginal_threshold,
            consecutive_required=consecutive_required, consecutive_hits=0,
            last_improvements=improvements[-consecutive_required:],
            observed_points=len(values),
        )

    streak = 0
    for imp in reversed(improvements):
        if abs(imp) <= marginal_threshold:
            streak += 1
        else:
            break
    converged = streak >= consecutive_required
    if converged:
        reason = (f"Converged: {streak} consecutive marginal change(s) "
                  f"<= {marginal_threshold} in {metric_key}.")
    else:
        reason = (f"Not converged: streak={streak}, required={consecutive_required}, "
                  f"threshold={marginal_threshold}.")
    return ConvergenceResult(
        converged=converged, should_stop=converged, reason=reason,
        metric_key=metric_key, marginal_threshold=marginal_threshold,
        consecutive_required=consecutive_required, consecutive_hits=streak,
        last_improvements=improvements[-consecutive_required:],
        observed_points=len(values),
    )


# ---------------------------------------------------------------------------
# step diagnostics (complement fit-level anomalies / gates)
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticIssue:
    """A single per-step diagnostic with severity and an actionable fix."""

    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    recommendation: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonify({
            "code": self.code, "severity": self.severity,
            "message": self.message, "recommendation": self.recommendation,
            "metadata": self.metadata,
        })


@dataclass
class DiagnosticsReport:
    """Aggregated per-step diagnostics."""

    issues: list = field(default_factory=list)
    summary: str = ""

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def has_blockers(self) -> bool:
        return any(i.severity.lower() == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return _jsonify({
            "schema": "doekit.DiagnosticsReport/1",
            "summary": self.summary,
            "has_blockers": self.has_blockers,
            "issues": [i.to_dict() for i in self.issues],
        })


def diagnose_step(metrics: dict, *, budget_remaining: Optional[float] = None,
                  uncertainty: Optional[float] = None, convergence=None,
                  min_power_gain: float = 0.01, max_negative_g_eff: float = -2.0,
                  high_uncertainty: float = 0.7) -> DiagnosticsReport:
    """Flag per-wave problems the fit-level diagnostics do not cover.

    Parameters
    ----------
    metrics : dict
        Step deltas (``delta_mean_power``, ``delta_G_efficiency``, ``n_add``).
        The power/prediction gates fire only when their key is present (learn);
        optimize steps carry ``predicted_improvement`` and skip them.
    budget_remaining : float, optional
        Remaining runs; a proposal exceeding it is a blocking error.
    uncertainty : float, optional
        Normalized uncertainty; above ``high_uncertainty`` is a warning.
    convergence : ConvergenceResult, optional
        If it stopped, an informational issue is added.

    Returns
    -------
    DiagnosticsReport
    """
    issues: list[DiagnosticIssue] = []
    n_add = float(metrics.get("n_add", metrics.get("extra_runs", 0.0)))

    # Precision/prediction gates only apply when the deltas are present (learn).
    # Optimize proposals carry ``predicted_improvement`` instead — gating on a
    # missing key would raise a spurious LOW_POWER_GAIN there.
    if "delta_mean_power" in metrics:
        d_power = float(metrics["delta_mean_power"])
        if d_power < min_power_gain:
            issues.append(DiagnosticIssue(
                "LOW_POWER_GAIN", "warning",
                f"Marginal power gain ({d_power:.4f}) below the expected minimum "
                f"({min_power_gain:.4f}).",
                "Consider focused replication or a model tweak before adding runs.",
                {"delta_mean_power": d_power}))
    if "delta_G_efficiency" in metrics:
        d_g = float(metrics["delta_G_efficiency"])
        if d_g <= max_negative_g_eff:
            issues.append(DiagnosticIssue(
                "PREDICTION_DEGRADATION", "warning",
                f"Notable G-efficiency drop ({d_g:.3f}).",
                "Review the estimation-precision vs prediction trade-off.",
                {"delta_G_efficiency": d_g}))
    if budget_remaining is not None and n_add > budget_remaining:
        issues.append(DiagnosticIssue(
            "BUDGET_OVERFLOW", "error",
            f"Proposal needs {n_add:.0f} runs but only {budget_remaining:.0f} remain.",
            "Reduce n_add or replan the budget before executing.",
            {"n_add": n_add, "budget_remaining": budget_remaining}))
    if uncertainty is not None and uncertainty >= high_uncertainty:
        issues.append(DiagnosticIssue(
            "HIGH_UNCERTAINTY", "warning",
            f"High uncertainty ({uncertainty:.3f}) above threshold "
            f"{high_uncertainty:.3f}.",
            "Prioritize uncertainty-reducing runs before exploiting the model.",
            {"uncertainty": uncertainty}))
    if convergence is not None and getattr(convergence, "should_stop", False):
        issues.append(DiagnosticIssue(
            "CONVERGENCE_REACHED", "info",
            getattr(convergence, "reason", "Convergence detected."),
            "Stop expanding the design and close the iteration.",
            {"converged": getattr(convergence, "converged", None)}))

    if not issues:
        summary = "No relevant step diagnostics."
    else:
        n_err = sum(1 for i in issues if i.severity == "error")
        n_warn = sum(1 for i in issues if i.severity == "warning")
        n_info = sum(1 for i in issues if i.severity == "info")
        summary = f"Diagnostics: {n_err} error(s), {n_warn} warning(s), {n_info} info."
    return DiagnosticsReport(issues=issues, summary=summary)
