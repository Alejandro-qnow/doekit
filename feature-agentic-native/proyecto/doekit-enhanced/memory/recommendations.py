"""
memory.recommendations - Recomendaciones basadas en historial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from memory.store import ExperimentStore


@dataclass
class HistoricalRecommendation:
    """Sugerencia accionable derivada del historial."""

    title: str
    rationale: str
    actions: List[str] = field(default_factory=list)


class HistoricalRecommender:
    """Genera recomendaciones desde patrones históricos similares."""

    def __init__(self, store: ExperimentStore):
        self.store = store

    def suggest(self, objective: str, factor_names: List[str], top_k: int = 5) -> HistoricalRecommendation:
        similar = self.store.find_similar(objective=objective, factor_names=factor_names, top_k=top_k)

        if not similar:
            return HistoricalRecommendation(
                title="Sin historial comparable",
                rationale="No se encontraron experimentos similares para transferir estrategia.",
                actions=[
                    "Iniciar con politica conservadora y recolectar evidencia en las primeras waves",
                    "Registrar metricas para habilitar meta-aprendizaje posterior",
                ],
            )

        avg_d_eff = sum(rec.metrics.get("delta_D_efficiency", 0.0) for rec in similar) / len(similar)

        if avg_d_eff >= 5.0:
            actions = [
                "Priorizar expansion de diseño en primeras iteraciones",
                "Aplicar criterio de convergencia con umbral moderado",
            ]
            title = "Historial favorable a expansión"
            rationale = f"Experimentos similares muestran mejora media de D-efficiency {avg_d_eff:.2f}."
        else:
            actions = [
                "Priorizar refinamiento de modelo antes de agregar corridas",
                "Aplicar umbral de parada mas estricto en convergencia",
            ]
            title = "Historial sugiere cautela"
            rationale = f"Mejora media de D-efficiency en historial similar es limitada ({avg_d_eff:.2f})."

        return HistoricalRecommendation(title=title, rationale=rationale, actions=actions)
