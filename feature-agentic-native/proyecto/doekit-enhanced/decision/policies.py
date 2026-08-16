"""
decision.policies - Politicas de decision para motor autonomo.
"""

from __future__ import annotations

from decision.core import Decision, DecisionContext, DecisionPolicy


class ThresholdPolicy(DecisionPolicy):
    """Politica simple basada en umbrales del score compuesto."""

    def __init__(self, continue_threshold: float = 0.15, refine_threshold: float = -0.05):
        self.continue_threshold = continue_threshold
        self.refine_threshold = refine_threshold

    def decide(self, context: DecisionContext, score) -> Decision:
        composite = float(score.composite)

        if composite >= self.continue_threshold:
            action = "continue"
            recommendations = [
                "Ejecutar corridas adicionales propuestas",
                "Re-evaluar metricas tras la siguiente wave",
            ]
        elif composite >= self.refine_threshold:
            action = "refine_model"
            recommendations = [
                "Revisar especificacion de modelo y terminos activos",
                "Ajustar criterio de propuesta antes de nuevas corridas",
            ]
        else:
            action = "stop"
            recommendations = [
                "Detener expansion del diseno actual",
                "Replantear estrategia o region experimental",
            ]

        confidence = min(1.0, max(0.0, 0.5 + abs(composite)))
        reasoning = (
            f"Decision by thresholds: composite={composite:.3f}, "
            f"continue>={self.continue_threshold:.3f}, "
            f"refine>={self.refine_threshold:.3f}"
        )

        return Decision(
            action=action,
            confidence=confidence,
            score=score,
            reasoning=reasoning,
            recommendations=recommendations,
            metadata={"policy": "ThresholdPolicy"},
        )


class RiskAdaptivePolicy(DecisionPolicy):
    """Ajusta umbrales segun tolerancia al riesgo."""

    def __init__(self):
        self._base_continue = 0.15
        self._base_refine = -0.05

    def decide(self, context: DecisionContext, score) -> Decision:
        composite = float(score.composite)

        if context.risk_tolerance == "low":
            continue_threshold = self._base_continue + 0.10
            refine_threshold = self._base_refine + 0.05
        elif context.risk_tolerance == "high":
            continue_threshold = self._base_continue - 0.08
            refine_threshold = self._base_refine - 0.04
        else:
            continue_threshold = self._base_continue
            refine_threshold = self._base_refine

        return ThresholdPolicy(
            continue_threshold=continue_threshold,
            refine_threshold=refine_threshold,
        ).decide(context, score)


class BudgetAwarePolicy(DecisionPolicy):
    """Prioriza restriccion presupuestaria antes de seguir experimentando."""

    def __init__(self, min_remaining_for_continue: int = 3):
        self.min_remaining_for_continue = min_remaining_for_continue

    def decide(self, context: DecisionContext, score) -> Decision:
        if context.budget_remaining <= 0:
            return Decision(
                action="stop",
                confidence=0.95,
                score=score,
                reasoning="Budget exhausted: no remaining runs.",
                recommendations=[
                    "Detener ejecucion de nuevas corridas",
                    "Analizar resultados actuales y cerrar iteracion",
                ],
                metadata={"policy": "BudgetAwarePolicy"},
            )

        if context.budget_remaining < self.min_remaining_for_continue and score.composite < 0.35:
            return Decision(
                action="refine_model",
                confidence=0.75,
                score=score,
                reasoning=(
                    "Low remaining budget with marginal score; refine model before spending final runs."
                ),
                recommendations=[
                    "Reducir incertidumbre del modelo con diagnostico",
                    "Usar corridas finales solo tras nueva priorizacion",
                ],
                metadata={"policy": "BudgetAwarePolicy"},
            )

        return ThresholdPolicy().decide(context, score)
