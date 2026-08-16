"""
memory.store - Almacen de experimentos para meta-aprendizaje.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """Registro de un experimento histórico."""

    experiment_id: str
    objective: str
    factor_names: List[str]

    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ExperimentStore:
    """Store en memoria con búsquedas por similaridad semántica simple."""

    def __init__(self):
        self._records: Dict[str, ExperimentRecord] = {}

    def add(self, record: ExperimentRecord) -> None:
        self._records[record.experiment_id] = record

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        return self._records.get(experiment_id)

    def list_all(self) -> List[ExperimentRecord]:
        return list(self._records.values())

    def find_similar(
        self,
        objective: str,
        factor_names: List[str],
        top_k: int = 5,
    ) -> List[ExperimentRecord]:
        scored = []
        target_factors = set(f.lower() for f in factor_names)

        for rec in self._records.values():
            score = 0.0
            if rec.objective.lower() == objective.lower():
                score += 0.6

            rec_factors = set(f.lower() for f in rec.factor_names)
            if target_factors:
                overlap = len(target_factors.intersection(rec_factors)) / float(len(target_factors))
                score += 0.4 * overlap

            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for score, rec in scored[:top_k] if score > 0.0]

    def size(self) -> int:
        return len(self._records)
