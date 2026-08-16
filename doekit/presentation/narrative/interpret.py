"""Uniform interpretation of a single doekit result for humans and agents.

Every doekit result already carries the facts (``to_dict``), a one-line
``summary``/``rationale`` and ``caveats``. :func:`interpret` *composes* those into
a small, uniform :class:`Interpretation` — a headline, the reasoning, warnings,
next actions and the raw facts — without re-deriving any statistics.

``Interpretation.for_llm()`` renders the block an agent appends to an LLM's
context (the "context addition"); ``to_dict()`` serializes it for handoff.

    from doekit.presentation.narrative import interpret
    rec = ed.recommend_design(goal="optimization", factors=3, budget=20)
    view = interpret(rec)
    print(view.for_llm())          # ready to add to an agent's context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...shared.serialize import jsonify as _jsonify


@dataclass
class Interpretation:
    """A uniform, agent-ready reading of a doekit result.

    Attributes
    ----------
    kind : str
        Result type: ``recommendation`` / ``evaluation`` / ``fit`` /
        ``proposal`` / ``comparison``.
    headline : str
        One-sentence takeaway.
    reasoning : str
        Why — taken from the result's own ``rationale`` / ``summary``.
    warnings : list of str
        Caveats and threshold flags (from the result's ``caveats`` / diagnostics).
    recommendations : list of str
        Concrete next actions.
    facts : dict
        The underlying numbers (a compact slice of the result's ``to_dict``);
        the interpretation never invents figures.
    confidence : str
        Qualitative confidence with a short reason.
    """

    kind: str
    headline: str
    reasoning: str
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    confidence: str = ""

    def to_dict(self) -> dict:
        """Serialize (``schema: doekit.Interpretation/1``)."""
        return _jsonify({
            "schema": "doekit.Interpretation/1",
            "kind": self.kind,
            "headline": self.headline,
            "reasoning": self.reasoning,
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "facts": dict(self.facts),
            "confidence": self.confidence,
        })

    def for_llm(self) -> str:
        """Render the block an agent adds to an LLM's context (context addition)."""
        lines = [f"[doekit · {self.kind}]", self.headline, "", f"Why: {self.reasoning}"]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        if self.recommendations:
            lines += ["", "Next:"] + [f"  - {r}" for r in self.recommendations]
        if self.facts:
            facts = ", ".join(f"{k}={_fmt(v)}" for k, v in self.facts.items())
            lines += ["", f"Facts: {facts}"]
        if self.confidence:
            lines += ["", f"Confidence: {self.confidence}"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.for_llm()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _round(v, ndigits: int = 4):
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------------------
# per-type interpreters (compose the result's own fields; never re-derive)
# ---------------------------------------------------------------------------

def _interpret_recommendation(rec) -> Interpretation:
    scenario = dict(getattr(rec, "scenario", {}) or {})
    n_runs = getattr(getattr(rec, "design", None), "n_runs", None)
    facts = {"method": rec.method, "n_runs": n_runs}
    facts.update({k: scenario[k] for k in ("goal", "factors", "model", "budget")
                  if k in scenario})
    headline = f"Recommended design: {rec.method}" + (
        f" ({n_runs} runs)" if n_runs is not None else "")
    return Interpretation(
        kind="recommendation",
        headline=headline,
        reasoning=getattr(rec, "rationale", "") or "",
        warnings=list(getattr(rec, "caveats", []) or []),
        recommendations=[
            "Evaluate before committing to the lab (ed.evaluate).",
            "Review the alternatives table and choose by your priorities.",
        ],
        facts=facts,
        confidence="Rule-based advisor ranking; a trade-off, not a unique optimum.",
    )


def _interpret_evaluation(ev) -> Interpretation:
    e = dict(getattr(ev, "efficiencies", {}) or {})
    warnings: list[str] = []
    if e.get("rank_deficient"):
        warnings.append("Saturated / rank-deficient: the model is not estimable.")
    facts = {
        "n_runs": getattr(ev, "n_runs", None),
        "dof": getattr(ev, "dof", None),
        "D_eff": _round(e.get("D_efficiency")),
        "G_eff": _round(e.get("G_efficiency")),
        "spv_mean": _round(e.get("spv_mean")),
    }
    d, g = e.get("D_efficiency"), e.get("G_efficiency")
    if d is not None and g is not None and not e.get("rank_deficient"):
        headline = f"Design quality: D-eff {d:.0f}%, G-eff {g:.0f}%."
        if getattr(ev, "dof", 1) is not None and ev.dof <= 0:
            warnings.append("dof<=0: no residual df; power uses the prior sigma.")
    else:
        headline = "Design quality could not be scored (see warnings)."
    return Interpretation(
        kind="evaluation",
        headline=headline,
        reasoning=ev.summary() if hasattr(ev, "summary") else "",
        warnings=warnings,
        recommendations=["If precision/power is short, augment with ed.propose_next_runs."],
        facts=facts,
        confidence="Computed in coded units against the fitted model.",
    )


def _interpret_fit(fit) -> Interpretation:
    r2 = getattr(fit, "r_squared", float("nan"))
    dof = getattr(fit, "dof", None)
    warnings: list[str] = []
    try:
        anomalies = fit.anomalies()
        n_anom = len(anomalies)
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        n_anom = 0
    if n_anom:
        warnings.append(f"{n_anom} atypical/influential run(s): review before concluding.")
    active = []
    try:
        for name, p in zip(fit.names, fit.pvalues):
            if name in ("(Intercept)", "Intercept"):
                continue
            if p == p and p < 0.05:  # p is finite and significant
                active.append(str(name))
    except (AttributeError, TypeError):
        pass
    headline = f"Model fit: R²={r2:.3f} (dof={dof})."
    recs = []
    if active:
        recs.append(f"Significant terms (p<0.05): {', '.join(active)}.")
    return Interpretation(
        kind="fit",
        headline=headline,
        reasoning=(f"OLS fit with {dof} residual degrees of freedom; "
                   f"R²={r2:.3f}."),
        warnings=warnings,
        recommendations=recs or ["No term reached p<0.05; consider more runs / a simpler model."],
        facts={"r_squared": _round(r2), "dof": dof, "n_active": len(active)},
        confidence=("High" if dof and dof >= 3 else "Low") + " — depends on residual dof.",
    )


def _interpret_proposal(prop) -> Interpretation:
    intent = getattr(prop, "intent", "learn")
    n_add = getattr(getattr(prop, "added", None), "n_runs", None)
    warnings = list(getattr(prop, "caveats", []) or [])
    if intent == "optimize":
        best = getattr(prop, "best_so_far", None)
        ee = dict(getattr(prop, "explore_exploit", {}) or {})
        headline = (f"Optimize: propose {n_add} run(s) by "
                    f"{getattr(prop, 'acquisition', '?')} "
                    f"({ee.get('mode', 'n/a')}).")
        facts = {
            "intent": intent,
            "acquisition": getattr(prop, "acquisition", None),
            "n_add": n_add,
            "best_so_far": best if not isinstance(best, dict) else None,
            "predicted_improvement": _round(getattr(prop, "predicted_improvement", None)),
            "mode": ee.get("mode"),
        }
        recs = ["Audit surrogate.calibration() before trusting best_so_far."]
    else:
        cmp = getattr(prop, "comparison", None)
        worth = getattr(cmp, "worth_it", None)
        headline = (f"Learn: propose {n_add} run(s) by "
                    f"{getattr(prop, 'criterion', 'D')}-optimal augmentation.")
        facts = {"intent": intent, "criterion": getattr(prop, "criterion", None),
                 "n_add": n_add, "worth_it": worth}
        recs = ["Decide the extra runs from comparison deltas, not intuition."]
    return Interpretation(
        kind="proposal",
        headline=headline,
        reasoning=getattr(prop, "rationale", "") or "",
        warnings=warnings,
        recommendations=recs,
        facts=facts,
        confidence=("Optimize conditions on the surrogate; calibration audits it."
                    if intent == "optimize"
                    else "Classical augmentation conditioned on current runs."),
    )


def _interpret_comparison(cmp) -> Interpretation:
    worth = getattr(cmp, "worth_it", None)
    delta = dict(getattr(cmp, "delta", {}) or {})
    facts = {
        "worth_it": worth,
        "delta_D_efficiency": _round(delta.get("D_efficiency")),
        "delta_G_efficiency": _round(delta.get("G_efficiency")),
        "delta_mean_power": _round(delta.get("mean_power")),
        "delta_n_runs": delta.get("n_runs"),
    }
    verdict = {True: "worth it", False: "not clearly worth it",
               None: "inconclusive"}.get(worth, "inconclusive")
    return Interpretation(
        kind="comparison",
        headline=f"Extra runs verdict: {verdict}.",
        reasoning=getattr(cmp, "summary", "") or "",
        warnings=[],
        recommendations=["Weigh the metric deltas against the extra run cost."],
        facts=facts,
        confidence="Heuristic verdict from quality deltas; set run_cost to tune it.",
    )


# duck-typed dispatch: (predicate, builder). Order matters (proposal before
# comparison, since a proposal also owns a comparison).
_DISPATCH = [
    (lambda o: _has(o, "method", "design", "rationale", "scenario"), _interpret_recommendation),
    (lambda o: _has(o, "efficiencies", "power", "vif"), _interpret_evaluation),
    (lambda o: _has(o, "r_squared", "coef", "names", "resid"), _interpret_fit),
    (lambda o: _has(o, "added", "combined", "comparison"), _interpret_proposal),
    (lambda o: _has(o, "a_label", "b_label", "delta", "worth_it"), _interpret_comparison),
]


def _has(obj, *attrs) -> bool:
    return all(hasattr(obj, a) for a in attrs)


def interpret(result, context: Optional[dict] = None) -> Interpretation:
    """Interpret a single doekit result into a uniform :class:`Interpretation`.

    Composes each result's own ``summary``/``rationale``, ``caveats`` and
    ``to_dict`` facts into a headline, warnings, next actions and confidence
    label — without re-deriving statistics.

    Parameters
    ----------
    result : Recommendation, DesignEvaluation, FitResult, NextRunsProposal or DesignComparison
        The object to read. Dispatch is by structure (duck-typed).
    context : dict, optional
        Reserved for caller-supplied framing (currently unused).

    Returns
    -------
    Interpretation
        Uniform view; use :meth:`Interpretation.for_llm` for agent context.

    Raises
    ------
    TypeError
        If ``result`` is not a recognized doekit result type.

    Examples
    --------
    >>> import doekit as ed
    >>> from doekit import interpret
    >>> pb = ed.plackett_burman(5)
    >>> view = interpret(ed.fit_linear_model(pb, pb.matrix["factor1"]))
    >>> view.kind == "fit" and "R²" in view.headline
    True
    """
    _ = context
    for predicate, builder in _DISPATCH:
        if predicate(result):
            return builder(result)
    raise TypeError(
        f"cannot interpret object of type {type(result).__name__!r}; expected a "
        "doekit Recommendation / DesignEvaluation / FitResult / NextRunsProposal / "
        "DesignComparison"
    )
