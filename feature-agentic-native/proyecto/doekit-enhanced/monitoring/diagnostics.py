"""
monitoring.diagnostics - Diagnosticos automaticos para experimentacion secuencial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticIssue:
    """Issue diagnostico con severidad y recomendacion accionable."""

    code: str
    severity: str
    message: str
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticsReport:
    """Resultado agregado de diagnostico."""

    issues: List[DiagnosticIssue] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def has_blockers(self) -> bool:
        return any(issue.severity.lower() == "error" for issue in self.issues)


class DefaultDiagnosticsAnalyzer:
    """Analizador baseline con reglas explicitas por metricas clave."""

    def __init__(
        self,
        min_power_gain: float = 0.01,
        max_negative_g_eff: float = -2.0,
        high_uncertainty_threshold: float = 0.7,
    ):
        self.min_power_gain = min_power_gain
        self.max_negative_g_eff = max_negative_g_eff
        self.high_uncertainty_threshold = high_uncertainty_threshold

    def analyze(
        self,
        metrics: Dict[str, float],
        budget_remaining: Optional[float] = None,
        uncertainty: Optional[float] = None,
        convergence_result: Any = None,
    ) -> DiagnosticsReport:
        issues: List[DiagnosticIssue] = []

        d_power = float(metrics.get("delta_mean_power", 0.0))
        d_g_eff = float(metrics.get("delta_G_efficiency", 0.0))
        n_add = float(metrics.get("n_add", metrics.get("extra_runs", 0.0)))

        if d_power < self.min_power_gain:
            issues.append(
                DiagnosticIssue(
                    code="LOW_POWER_GAIN",
                    severity="warning",
                    message=(
                        f"Ganancia de poder marginal ({d_power:.4f}) por debajo del minimo esperado "
                        f"({self.min_power_gain:.4f})."
                    ),
                    recommendation="Evaluar replicacion focalizada o ajuste de modelo antes de agregar corridas.",
                    metadata={"delta_mean_power": d_power},
                )
            )

        if d_g_eff <= self.max_negative_g_eff:
            issues.append(
                DiagnosticIssue(
                    code="PREDICTION_DEGRADATION",
                    severity="warning",
                    message=(
                        f"Degradacion relevante en G-efficiency ({d_g_eff:.3f})."
                    ),
                    recommendation="Revisar trade-off entre precision de estimacion y capacidad predictiva global.",
                    metadata={"delta_G_efficiency": d_g_eff},
                )
            )

        if budget_remaining is not None and n_add > budget_remaining:
            issues.append(
                DiagnosticIssue(
                    code="BUDGET_OVERFLOW",
                    severity="error",
                    message=(
                        f"Propuesta requiere {n_add:.1f} corridas con solo {budget_remaining:.1f} disponibles."
                    ),
                    recommendation="Reducir n_add o replanificar presupuesto antes de ejecutar.",
                    metadata={"n_add": n_add, "budget_remaining": budget_remaining},
                )
            )

        if uncertainty is not None and uncertainty >= self.high_uncertainty_threshold:
            issues.append(
                DiagnosticIssue(
                    code="HIGH_UNCERTAINTY",
                    severity="warning",
                    message=(
                        f"Incertidumbre elevada ({uncertainty:.3f}) sobre umbral {self.high_uncertainty_threshold:.3f}."
                    ),
                    recommendation="Priorizar corridas de reduccion de incertidumbre antes de explotar el modelo.",
                    metadata={"uncertainty": uncertainty},
                )
            )

        if convergence_result is not None and getattr(convergence_result, "should_stop", False):
            issues.append(
                DiagnosticIssue(
                    code="CONVERGENCE_REACHED",
                    severity="info",
                    message=getattr(convergence_result, "reason", "Convergencia detectada."),
                    recommendation="Detener expansion del diseno y cerrar iteracion con analisis final.",
                    metadata={"converged": getattr(convergence_result, "converged", None)},
                )
            )

        summary = self._build_summary(issues)
        return DiagnosticsReport(
            issues=issues,
            summary=summary,
            metadata={
                "rules": {
                    "min_power_gain": self.min_power_gain,
                    "max_negative_g_eff": self.max_negative_g_eff,
                    "high_uncertainty_threshold": self.high_uncertainty_threshold,
                }
            },
        )

    def _build_summary(self, issues: List[DiagnosticIssue]) -> str:
        if not issues:
            return "Sin issues diagnosticos relevantes."

        errors = sum(1 for i in issues if i.severity.lower() == "error")
        warnings = sum(1 for i in issues if i.severity.lower() == "warning")
        infos = sum(1 for i in issues if i.severity.lower() == "info")

        return f"Diagnostico: {errors} errores, {warnings} warnings, {infos} info."
