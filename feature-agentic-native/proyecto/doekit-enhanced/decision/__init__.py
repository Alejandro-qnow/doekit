"""
decision - API unificada del motor de decision autonoma.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from decision.core import Decision, DecisionContext, DecisionPolicy
from decision.pipeline import (
    DecisionPipelineConfig,
    DecisionPipelineResult,
    run_decision_pipeline,
)
from decision.policies import ThresholdPolicy
from decision.scoring import ContinuationScorer, DecisionScore, MultiObjectiveScorer
from decision.uncertainty import UncertaintyEstimate, UncertaintyQuantifier

# Integraciones opcionales re-exportadas para conveniencia
from memory import (
    ExperimentRecord,
    ExperimentStore,
    PriorEstimate,
    PriorLearner,
    HistoricalRecommendation,
    HistoricalRecommender,
)
from integrations import (
    CandidateScore,
    BayesianOptProposal,
    BayesianOptAdapter,
    propose_with_bayesian_opt,
)


def build_context(
    budget_total: int,
    budget_spent: int,
    risk_tolerance: str = "moderate",
    metrics: Optional[Dict[str, float]] = None,
    uncertainty: float = 0.0,
    proposal: Any = None,
    comparison: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionContext:
    """Factory helper para DecisionContext."""
    return DecisionContext(
        budget_total=budget_total,
        budget_spent=budget_spent,
        risk_tolerance=risk_tolerance,
        metrics=metrics or {},
        uncertainty=uncertainty,
        proposal=proposal,
        comparison=comparison,
        metadata=metadata or {},
    )


def decide_next_action(
    context: DecisionContext,
    scorer: Any = None,
    policy: Optional[DecisionPolicy] = None,
    uncertainty_estimate: Optional[UncertaintyEstimate] = None,
    convergence_result: Any = None,
    diagnostics_report: Any = None,
    event_bus: Any = None,
) -> Decision:
    """
    API principal de Fase 2.

    Args:
        context: DecisionContext con metricas y restricciones.
        scorer: Scorer con metodo score(context) -> DecisionScore.
        policy: Politica de decision. Default: ThresholdPolicy.
        uncertainty_estimate: Estimacion opcional de incertidumbre (Fase 3).
        convergence_result: Resultado opcional de checker de convergencia.
        diagnostics_report: Reporte opcional de diagnosticos.
        event_bus: Bus de eventos opcional (monitoring.events.EventBus).

    Returns:
        Decision final con accion, confianza y prompt explicativo.
    """
    scorer = scorer or ContinuationScorer()
    policy = policy or ThresholdPolicy()

    if uncertainty_estimate is not None:
        context.uncertainty = uncertainty_estimate.normalized_uncertainty
        context.metadata.setdefault("uncertainty", uncertainty_estimate.to_dict())

    score: DecisionScore = scorer.score(context)

    if event_bus is not None and hasattr(event_bus, "publish"):
        from monitoring.events import create_event

        event_bus.publish(
            create_event(
                "decision.scored",
                payload={"composite": score.composite, "components": score.components},
            )
        )

    decision = policy.decide(context, score)

    if convergence_result is not None:
        if event_bus is not None and hasattr(event_bus, "publish"):
            from monitoring.events import create_event

            event_bus.publish(
                create_event(
                    "convergence.checked",
                    payload={
                        "converged": getattr(convergence_result, "converged", None),
                        "should_stop": getattr(convergence_result, "should_stop", None),
                        "reason": getattr(convergence_result, "reason", ""),
                    },
                )
            )

        if getattr(convergence_result, "should_stop", False):
            decision.action = "stop"
            decision.reasoning = (
                f"{decision.reasoning}. Override por monitoring: "
                f"{getattr(convergence_result, 'reason', 'convergencia detectada')}"
            )
            decision.recommendations.insert(
                0,
                "Detener expansion del diseno por criterio de convergencia",
            )
            decision.prompt_injection = decision._build_prompt()

    if diagnostics_report is not None:
        issues = getattr(diagnostics_report, "issues", []) or []

        if event_bus is not None and hasattr(event_bus, "publish"):
            from monitoring.events import create_event

            event_bus.publish(
                create_event(
                    "diagnostics.generated",
                    payload={
                        "issue_count": len(issues),
                        "has_blockers": getattr(diagnostics_report, "has_blockers", False),
                        "summary": getattr(diagnostics_report, "summary", ""),
                    },
                )
            )

        if getattr(diagnostics_report, "has_blockers", False):
            decision.action = "stop"
            decision.reasoning = (
                f"{decision.reasoning}. Override por diagnostico bloqueante: "
                f"{getattr(diagnostics_report, 'summary', 'issues criticos detectados')}"
            )
            decision.recommendations.insert(
                0,
                "Resolver issues bloqueantes antes de nuevas corridas",
            )
            decision.prompt_injection = decision._build_prompt()

    decision.metadata.setdefault("scorer", type(scorer).__name__)
    decision.metadata.setdefault("policy", type(policy).__name__)
    if uncertainty_estimate is not None:
        decision.metadata.setdefault("uncertainty", uncertainty_estimate.to_dict())
    if convergence_result is not None:
        decision.metadata.setdefault(
            "convergence",
            {
                "converged": getattr(convergence_result, "converged", None),
                "should_stop": getattr(convergence_result, "should_stop", None),
                "reason": getattr(convergence_result, "reason", ""),
            },
        )
    if diagnostics_report is not None:
        decision.metadata.setdefault(
            "diagnostics",
            {
                "summary": getattr(diagnostics_report, "summary", ""),
                "issue_count": len(getattr(diagnostics_report, "issues", []) or []),
                "has_blockers": getattr(diagnostics_report, "has_blockers", False),
            },
        )

    if event_bus is not None and hasattr(event_bus, "publish"):
        from monitoring.events import create_event

        event_bus.publish(
            create_event(
                "decision.finalized",
                payload={
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "policy": decision.metadata.get("policy"),
                },
            )
        )

    return decision


def estimate_uncertainty(
    expected_gain: float,
    sigma_hat: float,
    threshold: float = 0.0,
    reference_sigma: float = 0.25,
) -> UncertaintyEstimate:
    """Helper directo para estimar incertidumbre desde media y sigma."""
    quantifier = UncertaintyQuantifier(reference_sigma=reference_sigma)
    return quantifier.estimate(
        expected_gain=expected_gain,
        sigma_hat=sigma_hat,
        threshold=threshold,
    )


def estimate_uncertainty_from_proposal(
    proposal: Any,
    comparison: Any = None,
    threshold: float = 0.0,
    reference_sigma: float = 0.25,
) -> UncertaintyEstimate:
    """Helper para extraer incertidumbre desde propose_next_runs/compare_designs."""
    quantifier = UncertaintyQuantifier(reference_sigma=reference_sigma)
    return quantifier.from_proposal(
        proposal=proposal,
        comparison=comparison,
        threshold=threshold,
    )


__all__ = [
    "Decision",
    "DecisionContext",
    "DecisionPolicy",
    "DecisionScore",
    "ContinuationScorer",
    "MultiObjectiveScorer",
    "ThresholdPolicy",
    "UncertaintyEstimate",
    "UncertaintyQuantifier",
    "DecisionPipelineConfig",
    "DecisionPipelineResult",
    "run_decision_pipeline",
    "ExperimentRecord",
    "ExperimentStore",
    "PriorEstimate",
    "PriorLearner",
    "HistoricalRecommendation",
    "HistoricalRecommender",
    "CandidateScore",
    "BayesianOptProposal",
    "BayesianOptAdapter",
    "propose_with_bayesian_opt",
    "build_context",
    "decide_next_action",
    "estimate_uncertainty",
    "estimate_uncertainty_from_proposal",
]
