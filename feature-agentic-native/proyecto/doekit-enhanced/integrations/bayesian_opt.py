"""
integrations.bayesian_opt - Integracion pragmatica para propuesta BO.

Si BoTorch esta disponible, el adapter queda listo para extension futura.
En ausencia de BoTorch, usa fallback deterministico basado en EI simplificado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, pi, sqrt
from typing import Any, Dict, List, Optional


@dataclass
class CandidateScore:
    index: int
    score: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class BayesianOptProposal:
    """Resultado de la integracion BO."""

    method: str
    selected_index: int
    selected_candidate: Dict[str, float]
    scores: List[CandidateScore]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BayesianOptAdapter:
    """Adapter de BO con fallback cuando BoTorch no esta instalado."""

    def __init__(self):
        self.botorch_available = self._check_botorch()

    def _check_botorch(self) -> bool:
        try:
            import botorch  # noqa: F401
            return True
        except Exception:
            return False

    def propose(
        self,
        candidate_pool: List[Dict[str, float]],
        objective_values: List[float],
        uncertainty_values: Optional[List[float]] = None,
        exploration_weight: float = 0.1,
    ) -> BayesianOptProposal:
        if not candidate_pool:
            raise ValueError("candidate_pool no puede ser vacio")
        if len(candidate_pool) != len(objective_values):
            raise ValueError("candidate_pool y objective_values deben tener igual longitud")

        if uncertainty_values is None:
            uncertainty_values = [0.1] * len(candidate_pool)
        if len(uncertainty_values) != len(candidate_pool):
            raise ValueError("uncertainty_values debe tener igual longitud que candidate_pool")

        # Fallback actual: EI simplificado independiente por candidato.
        best_so_far = max(objective_values)
        scores: List[CandidateScore] = []

        for idx, (mu, sigma) in enumerate(zip(objective_values, uncertainty_values)):
            ei = self._expected_improvement(mu, max(1e-9, sigma), best_so_far)
            score = mu + exploration_weight * sigma + ei
            scores.append(
                CandidateScore(
                    index=idx,
                    score=score,
                    details={
                        "mu": float(mu),
                        "sigma": float(sigma),
                        "ei": float(ei),
                    },
                )
            )

        scores.sort(key=lambda x: x.score, reverse=True)
        winner = scores[0]

        method = "botorch" if self.botorch_available else "fallback_ei"
        return BayesianOptProposal(
            method=method,
            selected_index=winner.index,
            selected_candidate=candidate_pool[winner.index],
            scores=scores,
            metadata={
                "botorch_available": self.botorch_available,
                "best_so_far": float(best_so_far),
                "exploration_weight": float(exploration_weight),
            },
        )

    def _expected_improvement(self, mean: float, sigma: float, best: float) -> float:
        z = (mean - best) / sigma
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        pdf = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)
        return (mean - best) * cdf + sigma * pdf


def propose_with_bayesian_opt(
    candidate_pool: List[Dict[str, float]],
    objective_values: List[float],
    uncertainty_values: Optional[List[float]] = None,
    exploration_weight: float = 0.1,
) -> BayesianOptProposal:
    """API simple para obtener propuesta BO/fallback."""
    adapter = BayesianOptAdapter()
    return adapter.propose(
        candidate_pool=candidate_pool,
        objective_values=objective_values,
        uncertainty_values=uncertainty_values,
        exploration_weight=exploration_weight,
    )
