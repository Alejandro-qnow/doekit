"""
monitoring.convergence - Deteccion de convergencia en experimentacion secuencial.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ConvergenceResult:
    """Resultado de evaluacion de convergencia."""

    converged: bool
    should_stop: bool

    reason: str
    metric_key: str

    marginal_threshold: float
    consecutive_required: int
    consecutive_hits: int

    last_improvements: List[float] = field(default_factory=list)
    observed_points: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConvergenceChecker(ABC):
    """Interfaz base para checkers de convergencia."""

    @abstractmethod
    def check(self, history: Iterable[Any]) -> ConvergenceResult:
        """Evalua convergencia sobre un historial temporal."""


class DefaultConvergenceChecker(ConvergenceChecker):
    """
    Checker por mejora marginal consecutiva.

    Regla de parada:
    - Si las ultimas N mejoras absolutas son menores o iguales al umbral,
      se considera convergido y se recomienda detener.
    """

    def __init__(
        self,
        metric_key: str = "delta_D_efficiency",
        marginal_threshold: float = 0.5,
        consecutive_required: int = 2,
        min_points: int = 3,
    ):
        self.metric_key = metric_key
        self.marginal_threshold = float(marginal_threshold)
        self.consecutive_required = int(consecutive_required)
        self.min_points = int(min_points)

    def check(self, history: Iterable[Any]) -> ConvergenceResult:
        values = self._extract_values(history)
        improvements = self._compute_improvements(values)

        if len(values) < self.min_points:
            return ConvergenceResult(
                converged=False,
                should_stop=False,
                reason="Historial insuficiente para evaluar convergencia",
                metric_key=self.metric_key,
                marginal_threshold=self.marginal_threshold,
                consecutive_required=self.consecutive_required,
                consecutive_hits=0,
                last_improvements=improvements[-self.consecutive_required :],
                observed_points=len(values),
                metadata={"values": values},
            )

        streak = self._count_marginal_streak(improvements)
        converged = streak >= self.consecutive_required

        if converged:
            reason = (
                f"Convergencia detectada: {streak} mejoras consecutivas <= "
                f"{self.marginal_threshold} en {self.metric_key}"
            )
        else:
            reason = (
                f"Aun sin convergencia: streak={streak}, requerido={self.consecutive_required}, "
                f"umbral={self.marginal_threshold}"
            )

        return ConvergenceResult(
            converged=converged,
            should_stop=converged,
            reason=reason,
            metric_key=self.metric_key,
            marginal_threshold=self.marginal_threshold,
            consecutive_required=self.consecutive_required,
            consecutive_hits=streak,
            last_improvements=improvements[-self.consecutive_required :],
            observed_points=len(values),
            metadata={"values": values, "improvements": improvements},
        )

    def _extract_values(self, history: Iterable[Any]) -> List[float]:
        values: List[float] = []

        for item in history:
            value = self._extract_single_value(item)
            if value is not None:
                values.append(float(value))

        return values

    def _extract_single_value(self, item: Any) -> Optional[float]:
        if isinstance(item, dict):
            if self.metric_key in item:
                return self._as_float(item[self.metric_key])

            metrics = item.get("metrics")
            if isinstance(metrics, dict) and self.metric_key in metrics:
                return self._as_float(metrics[self.metric_key])

            return None

        if hasattr(item, self.metric_key):
            return self._as_float(getattr(item, self.metric_key))

        if hasattr(item, "metrics") and isinstance(getattr(item, "metrics"), dict):
            metrics = getattr(item, "metrics")
            if self.metric_key in metrics:
                return self._as_float(metrics[self.metric_key])

        return None

    def _compute_improvements(self, values: List[float]) -> List[float]:
        if len(values) < 2:
            return []

        improvements = []
        for i in range(1, len(values)):
            improvements.append(values[i] - values[i - 1])

        return improvements

    def _count_marginal_streak(self, improvements: List[float]) -> int:
        streak = 0
        for imp in reversed(improvements):
            if abs(imp) <= self.marginal_threshold:
                streak += 1
            else:
                break
        return streak

    def _as_float(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
