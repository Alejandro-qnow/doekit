"""
decision.pipeline - Pipeline opcional y configurable de decision.

Mantiene la arquitectura modular y ofrece una capa de orquestacion para
casos estandar sin perder trazabilidad por etapa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from decision.core import Decision, DecisionContext
from decision.policies import ThresholdPolicy
from decision.scoring import ContinuationScorer

from monitoring.convergence import DefaultConvergenceChecker
from monitoring.diagnostics import DefaultDiagnosticsAnalyzer
from monitoring.events import EventBus


@dataclass
class DecisionPipelineConfig:
    """Configuracion del pipeline opcional."""

    enable_uncertainty: bool = True
    enable_convergence: bool = True
    enable_diagnostics: bool = True
    enable_events: bool = True

    convergence_metric_key: str = "delta_D_efficiency"
    convergence_marginal_threshold: float = 0.5
    convergence_consecutive_required: int = 2
    convergence_min_points: int = 3


@dataclass
class DecisionPipelineResult:
    """Salida completa del pipeline con artefactos intermedios."""

    decision: Decision

    uncertainty_estimate: Any = None
    convergence_result: Any = None
    diagnostics_report: Any = None

    event_bus: Optional[EventBus] = None
    executed_stages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_decision_pipeline(
    context: DecisionContext,
    history: Optional[Iterable[Any]] = None,
    config: Optional[DecisionPipelineConfig] = None,
    scorer: Any = None,
    policy: Any = None,
    convergence_checker: Any = None,
    diagnostics_analyzer: Any = None,
    event_bus: Optional[EventBus] = None,
) -> DecisionPipelineResult:
    """
    Ejecuta flujo opcional por etapas para producir decision final.

    Etapas (configurables):
    1) uncertainty
    2) convergence
    3) diagnostics
    4) decision
    """
    config = config or DecisionPipelineConfig()
    scorer = scorer or ContinuationScorer()
    policy = policy or ThresholdPolicy()

    stages: List[str] = []

    if config.enable_events:
        bus = event_bus or EventBus()
    else:
        bus = None

    uncertainty_estimate = None
    if config.enable_uncertainty:
        from decision import estimate_uncertainty_from_proposal

        if context.proposal is not None:
            uncertainty_estimate = estimate_uncertainty_from_proposal(
                proposal=context.proposal,
                comparison=context.comparison,
            )
            stages.append("uncertainty")

    convergence_result = None
    if config.enable_convergence and history is not None:
        checker = convergence_checker or DefaultConvergenceChecker(
            metric_key=config.convergence_metric_key,
            marginal_threshold=config.convergence_marginal_threshold,
            consecutive_required=config.convergence_consecutive_required,
            min_points=config.convergence_min_points,
        )
        convergence_result = checker.check(history)
        stages.append("convergence")

    diagnostics_report = None
    if config.enable_diagnostics:
        analyzer = diagnostics_analyzer or DefaultDiagnosticsAnalyzer()
        uncertainty_value = context.uncertainty
        if uncertainty_estimate is not None:
            uncertainty_value = uncertainty_estimate.normalized_uncertainty

        diagnostics_report = analyzer.analyze(
            metrics=context.metrics,
            budget_remaining=context.budget_remaining,
            uncertainty=uncertainty_value,
            convergence_result=convergence_result,
        )
        stages.append("diagnostics")

    from decision import decide_next_action

    decision = decide_next_action(
        context=context,
        scorer=scorer,
        policy=policy,
        uncertainty_estimate=uncertainty_estimate,
        convergence_result=convergence_result,
        diagnostics_report=diagnostics_report,
        event_bus=bus,
    )
    stages.append("decision")

    return DecisionPipelineResult(
        decision=decision,
        uncertainty_estimate=uncertainty_estimate,
        convergence_result=convergence_result,
        diagnostics_report=diagnostics_report,
        event_bus=bus,
        executed_stages=stages,
        metadata={
            "pipeline": "optional-configurable",
            "events_enabled": config.enable_events,
        },
    )
