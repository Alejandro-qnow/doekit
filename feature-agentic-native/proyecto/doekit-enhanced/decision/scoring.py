"""
decision.scoring - Sistemas de scoring para decidir continuidad experimental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from decision.core import DecisionContext


@dataclass
class DecisionScore:
    """Score compuesto con trazabilidad de componentes."""

    composite: float
    benefit: float
    cost: float
    risk: float
    uncertainty_penalty: float

    components: Dict[str, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)


class ContinuationScorer:
    """Scorer base para decidir si conviene continuar experimentando."""

    def __init__(
        self,
        benefit_weight: float = 1.0,
        cost_weight: float = 0.7,
        risk_weight: float = 0.8,
        uncertainty_weight: float = 0.6,
    ):
        self.benefit_weight = benefit_weight
        self.cost_weight = cost_weight
        self.risk_weight = risk_weight
        self.uncertainty_weight = uncertainty_weight

    def score(self, context: DecisionContext) -> DecisionScore:
        m = context.metrics

        d_eff_gain = float(m.get("delta_D_efficiency", 0.0))
        power_gain = float(m.get("delta_mean_power", 0.0))
        g_eff_delta = float(m.get("delta_G_efficiency", 0.0))
        extra_runs = float(m.get("n_add", m.get("extra_runs", 0.0)))

        benefit = 0.6 * self._normalize(d_eff_gain, 20.0) + 0.4 * self._normalize(power_gain, 0.2)

        remaining = max(1.0, float(context.budget_remaining))
        cost = min(2.0, extra_runs / remaining)

        risk = max(0.0, -g_eff_delta / 10.0)
        uncertainty_penalty = max(0.0, min(1.0, context.uncertainty))

        composite = (
            self.benefit_weight * benefit
            - self.cost_weight * cost
            - self.risk_weight * risk
            - self.uncertainty_weight * uncertainty_penalty
        )

        rationale = [
            f"benefit={benefit:.3f} from d_eff_gain={d_eff_gain:.3f}, power_gain={power_gain:.3f}",
            f"cost={cost:.3f} using extra_runs={extra_runs:.1f} and budget_remaining={remaining:.1f}",
            f"risk={risk:.3f} from g_eff_delta={g_eff_delta:.3f}",
            f"uncertainty_penalty={uncertainty_penalty:.3f}",
        ]

        return DecisionScore(
            composite=composite,
            benefit=benefit,
            cost=cost,
            risk=risk,
            uncertainty_penalty=uncertainty_penalty,
            components={
                "d_eff_gain": d_eff_gain,
                "power_gain": power_gain,
                "g_eff_delta": g_eff_delta,
                "extra_runs": extra_runs,
            },
            rationale=rationale,
        )

    def _normalize(self, value: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        x = value / scale
        return max(-1.0, min(1.0, x))


class MultiObjectiveScorer:
    """Scorer multi-objetivo con prioridades configurables."""

    def __init__(self, objective_weights: Dict[str, float] | None = None):
        self.objective_weights = objective_weights or {
            "precision": 0.5,
            "prediction": 0.3,
            "cost": 0.2,
        }

    def score(self, context: DecisionContext) -> DecisionScore:
        objectives = context.metrics.get("objectives", {})

        precision = float(objectives.get("precision", 0.0))
        prediction = float(objectives.get("prediction", 0.0))
        cost = float(objectives.get("cost", 0.0))

        composite = (
            self.objective_weights.get("precision", 0.0) * precision
            + self.objective_weights.get("prediction", 0.0) * prediction
            - self.objective_weights.get("cost", 0.0) * cost
        )

        return DecisionScore(
            composite=composite,
            benefit=precision + prediction,
            cost=cost,
            risk=max(0.0, context.uncertainty),
            uncertainty_penalty=max(0.0, min(1.0, context.uncertainty)),
            components={
                "precision": precision,
                "prediction": prediction,
                "cost": cost,
            },
            rationale=[
                "Composite built from weighted objectives",
                f"weights={self.objective_weights}",
            ],
        )
