"""
decision.core - Estructuras base para decision autonoma en DoE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


DecisionAction = Literal["continue", "stop", "refine_model"]
RiskTolerance = Literal["low", "moderate", "high"]


@dataclass
class DecisionContext:
    """Contexto operativo para decidir el siguiente paso experimental."""

    budget_total: int
    budget_spent: int
    risk_tolerance: RiskTolerance = "moderate"

    metrics: Dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0

    proposal: Any = None
    comparison: Any = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.budget_total - self.budget_spent)

    @property
    def budget_usage_ratio(self) -> float:
        if self.budget_total <= 0:
            return 1.0
        return min(1.0, max(0.0, self.budget_spent / float(self.budget_total)))


@dataclass
class Decision:
    """Resultado final de una politica de decision."""

    action: DecisionAction
    confidence: float
    score: Any

    reasoning: str
    recommendations: List[str] = field(default_factory=list)

    context_addition: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()

        if not self.context_addition:
            self.context_addition = self._build_prompt()

    def _build_prompt(self) -> str:
        lines = [
            "DECISION AUTONOMA:",
            f"Accion: {self.action}",
            f"Confianza: {self.confidence:.2f}",
            "",
            "RAZONAMIENTO:",
            self.reasoning,
        ]

        if self.recommendations:
            lines += ["", "RECOMENDACIONES:"]
            lines += [f"- {rec}" for rec in self.recommendations]

        if self.score is not None and hasattr(self.score, "composite"):
            lines += ["", "SCORING:", f"composite={self.score.composite:.3f}"]

        return "\n".join(lines)


class DecisionPolicy(ABC):
    """Interfaz base para politicas de decision."""

    @abstractmethod
    def decide(self, context: DecisionContext, score: Any) -> Decision:
        """Retorna una decision final basada en contexto y score."""
