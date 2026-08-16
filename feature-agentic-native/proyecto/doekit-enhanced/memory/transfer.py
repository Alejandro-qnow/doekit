"""
memory.transfer - Transfer learning de priors desde historial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List

from memory.store import ExperimentStore


@dataclass
class PriorEstimate:
    """Priors derivados de experimentos similares."""

    objective: str
    n_sources: int

    expected_delta_d_efficiency: float = 0.0
    expected_delta_mean_power: float = 0.0
    expected_uncertainty: float = 0.5

    metadata: Dict[str, float] = field(default_factory=dict)


class PriorLearner:
    """Aprende priors simples a partir de registros históricos similares."""

    def __init__(self, store: ExperimentStore):
        self.store = store

    def learn(
        self,
        objective: str,
        factor_names: List[str],
        top_k: int = 5,
    ) -> PriorEstimate:
        similar = self.store.find_similar(objective=objective, factor_names=factor_names, top_k=top_k)

        if not similar:
            return PriorEstimate(
                objective=objective,
                n_sources=0,
                expected_delta_d_efficiency=0.0,
                expected_delta_mean_power=0.0,
                expected_uncertainty=0.5,
                metadata={"fallback": 1.0},
            )

        d_eff_values = [rec.metrics.get("delta_D_efficiency", 0.0) for rec in similar]
        power_values = [rec.metrics.get("delta_mean_power", 0.0) for rec in similar]
        uncertainty_values = [rec.metrics.get("uncertainty", 0.5) for rec in similar]

        return PriorEstimate(
            objective=objective,
            n_sources=len(similar),
            expected_delta_d_efficiency=mean(d_eff_values) if d_eff_values else 0.0,
            expected_delta_mean_power=mean(power_values) if power_values else 0.0,
            expected_uncertainty=mean(uncertainty_values) if uncertainty_values else 0.5,
            metadata={"top_k": float(top_k)},
        )
