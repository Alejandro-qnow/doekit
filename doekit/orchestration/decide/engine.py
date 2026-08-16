"""Decision engine: turn experiment signals into stop / augment / refine / redesign.

One engine, not two: it *consumes* the signals doekit already produces — the
comparison deltas and ``worth_it`` for the ``learn`` intent, and the native
``predicted_improvement`` / ``explore_exploit`` (+ optional calibration) for the
``optimize`` intent — plus quality gates and the run budget. The rule-based
``gate_board`` in :mod:`doekit.presentation.workspace.conclusions` delegates its
process status to this engine, so there is a single decision logic.

Crucially, ``optimize`` is *not* scored by D-efficiency (which often drops while
the result improves): its benefit comes from the expected improvement and the
explore/exploit stance, not from information metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ...shared.serialize import jsonify as _jsonify

DecisionAction = Literal["augment", "refine", "stop", "redesign"]
_GATE_STATUS = {"augment": "augment", "refine": "augment",
                "stop": "stop", "redesign": "redesign"}


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

@dataclass
class DecisionContext:
    """Signals needed to decide the next experimental step.

    Attributes
    ----------
    budget_total, budget_spent : int
        Run budget accounting (``0`` total = unknown/unbounded).
    risk_tolerance : {"low", "moderate", "high"}
        Shifts the policy thresholds.
    intent : {"learn", "optimize"}
        Which value the proposal pursues (drives how benefit is scored).
    quality : str, optional
        Design quality gate level (e.g. ``"rank_deficient"`` forces redesign).
    inference : str, optional
        Inference gate status (``"no_response"``, ``"saturated_no_test"``, …).
    metrics : dict
        Signals: ``delta_D_efficiency``, ``delta_mean_power``,
        ``delta_G_efficiency``, ``n_add`` (learn) or ``predicted_improvement``
        (optimize).
    uncertainty : float
        Normalized uncertainty in ``[0, 1]`` (surrogate mis-calibration /
        explore stance).
    worth_it : bool, optional
        The cheap heuristic from :class:`DesignComparison`.
    """

    budget_total: int = 0
    budget_spent: int = 0
    risk_tolerance: str = "moderate"
    intent: str = "learn"
    quality: Optional[str] = None
    inference: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    uncertainty: float = 0.0
    worth_it: Optional[bool] = None
    unknowns: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def budget_remaining(self) -> int:
        if self.budget_total <= 0:
            return 10**9  # unbounded
        return max(0, self.budget_total - self.budget_spent)

    @property
    def has_scoring_metrics(self) -> bool:
        keys = ("delta_D_efficiency", "delta_mean_power", "delta_G_efficiency",
                "predicted_improvement")
        return any(k in self.metrics for k in keys)


@dataclass
class DecisionScore:
    """Composite score with a transparent breakdown."""

    composite: float
    benefit: float
    cost: float
    risk: float
    uncertainty_penalty: float
    components: dict = field(default_factory=dict)
    rationale: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _jsonify({
            "composite": self.composite, "benefit": self.benefit,
            "cost": self.cost, "risk": self.risk,
            "uncertainty_penalty": self.uncertainty_penalty,
            "components": self.components, "rationale": self.rationale,
        })


@dataclass
class Decision:
    """Recommended next action with confidence and a serializable rationale."""

    action: DecisionAction
    confidence: float
    reasoning: str
    score: Optional[DecisionScore] = None
    recommendations: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def gate_status(self) -> str:
        """Map to the ``gate_board`` process vocabulary (stop/augment/redesign)."""
        return _GATE_STATUS[self.action]

    def to_dict(self) -> dict:
        """Serialize (``schema: doekit.Decision/1``)."""
        return _jsonify({
            "schema": "doekit.Decision/1",
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommendations": list(self.recommendations),
            "score": self.score.to_dict() if self.score is not None else None,
            "metadata": dict(self.metadata),
        })

    def for_llm(self) -> str:
        lines = [f"[doekit · decision] action={self.action} "
                 f"(confidence {self.confidence:.2f})", self.reasoning]
        if self.recommendations:
            lines += ["", "Next:"] + [f"  - {r}" for r in self.recommendations]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.for_llm()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def _norm(value, scale):
    if scale <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / scale))


class ContinuationScorer:
    """Score whether continuing the experiment is worthwhile.

    ``learn`` benefit comes from D-efficiency / power gains; ``optimize`` benefit
    comes from the expected improvement (never from D-efficiency, which can drop
    while the outcome improves).
    """

    def __init__(self, benefit_weight: float = 1.0, cost_weight: float = 0.7,
                 risk_weight: float = 0.8, uncertainty_weight: float = 0.6):
        self.benefit_weight = benefit_weight
        self.cost_weight = cost_weight
        self.risk_weight = risk_weight
        self.uncertainty_weight = uncertainty_weight

    def score(self, context: DecisionContext) -> DecisionScore:
        m = context.metrics
        extra_runs = float(m.get("n_add", m.get("extra_runs", 0.0)))
        remaining = max(1.0, float(context.budget_remaining))
        cost = min(2.0, extra_runs / remaining)
        uncertainty_penalty = _clip(context.uncertainty)

        if context.intent == "optimize":
            pi = float(m.get("predicted_improvement", 0.0))
            benefit = 1.0 if pi > 1e-9 else 0.0
            risk = uncertainty_penalty  # in optimize, mis-calibration is the risk
            components = {"predicted_improvement": pi, "extra_runs": extra_runs}
            rationale = [
                f"optimize benefit={benefit:.2f} from predicted_improvement={pi:.4g}",
                f"cost={cost:.3f} (extra_runs={extra_runs:.0f}/{remaining:.0f})",
                f"risk=uncertainty={risk:.3f}",
            ]
        else:
            d_gain = float(m.get("delta_D_efficiency", 0.0))
            p_gain = float(m.get("delta_mean_power", 0.0))
            g_delta = float(m.get("delta_G_efficiency", 0.0))
            benefit = 0.6 * _norm(d_gain, 20.0) + 0.4 * _norm(p_gain, 0.2)
            risk = max(0.0, -g_delta / 10.0)
            components = {"d_eff_gain": d_gain, "power_gain": p_gain,
                          "g_eff_delta": g_delta, "extra_runs": extra_runs}
            rationale = [
                f"learn benefit={benefit:.3f} (d_eff={d_gain:.2f}, power={p_gain:.3f})",
                f"cost={cost:.3f} (extra_runs={extra_runs:.0f}/{remaining:.0f})",
                f"risk={risk:.3f} (g_eff_delta={g_delta:.2f})",
            ]

        composite = (self.benefit_weight * benefit - self.cost_weight * cost
                     - self.risk_weight * risk
                     - self.uncertainty_weight * uncertainty_penalty)
        return DecisionScore(composite=composite, benefit=benefit, cost=cost,
                             risk=risk, uncertainty_penalty=uncertainty_penalty,
                             components=components, rationale=rationale)


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

class DecisionPolicy:
    """Base policy: map a :class:`DecisionScore` to a :class:`Decision`."""

    def decide(self, context: DecisionContext, score: DecisionScore) -> Decision:  # noqa: D401
        raise NotImplementedError


class ThresholdPolicy(DecisionPolicy):
    """Threshold on the composite score: augment / refine / stop."""

    def __init__(self, continue_threshold: float = 0.15,
                 refine_threshold: float = -0.05):
        self.continue_threshold = continue_threshold
        self.refine_threshold = refine_threshold

    def decide(self, context: DecisionContext, score: DecisionScore) -> Decision:
        c = float(score.composite)
        if c >= self.continue_threshold:
            action, recs = "augment", [
                "Run the proposed extra runs.",
                "Re-evaluate metrics after the next wave.",
            ]
        elif c >= self.refine_threshold:
            action, recs = "refine", [
                "Revisit the model spec / active terms before spending runs.",
            ]
        else:
            action, recs = "stop", [
                "Stop expanding the current design.",
                "Analyze current results / rethink the region.",
            ]
        confidence = _clip(0.5 + abs(c))
        reasoning = (f"composite={c:.3f} vs continue>={self.continue_threshold:.2f}, "
                     f"refine>={self.refine_threshold:.2f} (intent={context.intent})")
        return Decision(action=action, confidence=confidence, reasoning=reasoning,
                        score=score, recommendations=recs,
                        metadata={"policy": "ThresholdPolicy"})


class RiskAdaptivePolicy(DecisionPolicy):
    """Shift the thresholds by risk tolerance (low = more conservative)."""

    def __init__(self, base_continue: float = 0.15, base_refine: float = -0.05):
        self.base_continue = base_continue
        self.base_refine = base_refine

    def decide(self, context: DecisionContext, score: DecisionScore) -> Decision:
        if context.risk_tolerance == "low":
            cont, refi = self.base_continue + 0.10, self.base_refine + 0.05
        elif context.risk_tolerance == "high":
            cont, refi = self.base_continue - 0.08, self.base_refine - 0.04
        else:
            cont, refi = self.base_continue, self.base_refine
        d = ThresholdPolicy(cont, refi).decide(context, score)
        d.metadata["policy"] = "RiskAdaptivePolicy"
        return d


class BudgetAwarePolicy(DecisionPolicy):
    """Let the run budget dominate before scoring benefit."""

    def __init__(self, min_remaining_for_continue: int = 3):
        self.min_remaining_for_continue = min_remaining_for_continue

    def decide(self, context: DecisionContext, score: DecisionScore) -> Decision:
        if context.budget_remaining <= 0:
            return Decision("stop", 0.95, "Budget exhausted: no remaining runs.",
                            score=score, recommendations=["Close the iteration."],
                            metadata={"policy": "BudgetAwarePolicy"})
        if (context.budget_remaining < self.min_remaining_for_continue
                and score.composite < 0.35):
            return Decision("refine", 0.75,
                            "Low remaining budget with marginal score.",
                            score=score,
                            recommendations=["Reduce model uncertainty first."],
                            metadata={"policy": "BudgetAwarePolicy"})
        d = ThresholdPolicy().decide(context, score)
        d.metadata["policy"] = "BudgetAwarePolicy"
        return d


# ---------------------------------------------------------------------------
# gate-mode rules (no scoring metrics available) — mirror the legacy gate
# ---------------------------------------------------------------------------

def _gate_decision(context: DecisionContext) -> Decision:
    """Deterministic decision when only quality/inference/worth_it are known."""
    if context.worth_it is True:
        return Decision("augment", 0.7,
                        "Comparison judges the extra runs worthwhile.",
                        recommendations=["Run the proposed extra runs."],
                        metadata={"mode": "gate"})
    if context.inference == "no_response":
        return Decision("stop", 0.6, "Awaiting response before deciding.",
                        recommendations=["Collect responses, then re-evaluate."],
                        metadata={"mode": "gate", "reason": "awaiting_response"})
    if context.inference == "saturated_no_test":
        return Decision("augment", 0.6, "Saturated fit: add runs for residual df.",
                        recommendations=["Augment to enable inference."],
                        metadata={"mode": "gate", "reason": "saturated_fit"})
    return Decision("stop", 0.55,
                    "Default stop; augment only with worth_it or explicit budget.",
                    metadata={"mode": "gate"})


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def decide_next_action(context: DecisionContext, scorer: Optional[ContinuationScorer] = None,
                       policy: Optional[DecisionPolicy] = None,
                       convergence=None) -> Decision:
    """Decide the next experimental action from the available signals.

    Hard gates (rank deficiency, exhausted budget, detected convergence) win
    first; otherwise the score-based policy runs when scoring metrics exist, or
    the deterministic gate rules when they do not.

    Parameters
    ----------
    context : DecisionContext
    scorer : ContinuationScorer, optional
    policy : DecisionPolicy, optional
        Defaults to :class:`ThresholdPolicy`.
    convergence : object, optional
        Anything exposing ``should_stop`` / ``reason`` (see M3 monitoring).

    Returns
    -------
    Decision
    """
    if context.quality == "rank_deficient":
        return Decision("redesign", 0.9,
                        "Design is rank-deficient: the model is not estimable.",
                        recommendations=["Reduce the model or add runs."],
                        metadata={"reason": "rank_deficient"})
    if context.budget_total > 0 and context.budget_remaining <= 0:
        return Decision("stop", 0.95, "Budget exhausted: no remaining runs.",
                        recommendations=["Analyze current results; close the iteration."],
                        metadata={"reason": "budget_exhausted"})
    if convergence is not None and getattr(convergence, "should_stop", False):
        return Decision("stop", 0.85,
                        f"Convergence: {getattr(convergence, 'reason', 'no marginal gain')}.",
                        recommendations=["Stop; further runs add little."],
                        metadata={"reason": "converged"})

    if not context.has_scoring_metrics:
        return _gate_decision(context)

    scorer = scorer or ContinuationScorer()
    policy = policy or ThresholdPolicy()
    score = scorer.score(context)
    return policy.decide(context, score)


def _calibration_uncertainty(surrogate) -> Optional[float]:
    """Overconfidence gap at the 95% interval (higher = less trustworthy sigma)."""
    if surrogate is None or not hasattr(surrogate, "calibration"):
        return None
    try:
        cov = surrogate.calibration().get("coverage", {})
    except Exception:  # noqa: BLE001 - calibration is best-effort
        return None
    c95 = cov.get(0.95)
    if c95 is None:
        return None
    return _clip(0.95 - float(c95))


def context_from_proposal(proposal, *, budget_total: int = 0, budget_spent: int = 0,
                          risk_tolerance: str = "moderate",
                          use_calibration: bool = False) -> DecisionContext:
    """Build a :class:`DecisionContext` from a :class:`NextRunsProposal`.

    Reads the comparison deltas (learn) or the native optimize fields
    (``predicted_improvement``, ``explore_exploit``, optional calibration).
    """
    cmp = getattr(proposal, "comparison", None)
    delta = dict(getattr(cmp, "delta", {}) or {})
    intent = getattr(proposal, "intent", "learn")
    n_add = getattr(getattr(proposal, "added", None), "n_runs", 0)

    metrics: dict = {"n_add": n_add}
    uncertainty = 0.0
    if intent == "optimize":
        metrics["predicted_improvement"] = getattr(proposal, "predicted_improvement", 0.0) or 0.0
        ee = dict(getattr(proposal, "explore_exploit", {}) or {})
        mode = ee.get("mode")
        uncertainty = {"exploring": 0.3, "balanced": 0.15,
                       "exploiting": 0.05}.get(mode, 0.2)
        if use_calibration:
            cu = _calibration_uncertainty(getattr(proposal, "surrogate", None))
            if cu is not None:
                uncertainty = cu
    else:
        for src, dst in (("D_efficiency", "delta_D_efficiency"),
                         ("mean_power", "delta_mean_power"),
                         ("G_efficiency", "delta_G_efficiency")):
            if delta.get(src) is not None:
                metrics[dst] = delta[src]

    return DecisionContext(
        budget_total=budget_total, budget_spent=budget_spent,
        risk_tolerance=risk_tolerance, intent=intent, metrics=metrics,
        uncertainty=uncertainty,
        worth_it=(getattr(cmp, "worth_it", None) if cmp is not None else None),
    )
