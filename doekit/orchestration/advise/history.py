"""Meta-learning from past experiments: priors and historical advice.

The persistent store is the traceable workspace (``ExperimentProject`` → waves);
this module reads those into lightweight :class:`ExperimentRecord`s and derives
priors / advice for a new case. It does not introduce a parallel database.

    hist = ExperimentHistory.from_project(project)
    prior = learn_priors(hist, objective="optimization", factor_names=["T", "pH"])
    advice = historical_recommendation(hist, "optimization", ["T", "pH"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from ...shared.serialize import jsonify as _jsonify


@dataclass
class ExperimentRecord:
    """A compact record of a past experiment for similarity search.

    Attributes
    ----------
    experiment_id : str
        Stable identifier (e.g. a wave id).
    objective : str
        Goal label (``"screening"`` / ``"optimization"`` / …).
    factor_names : list of str
        Factor names involved.
    metrics : dict
        Outcome signals (``delta_D_efficiency``, ``delta_mean_power``,
        ``uncertainty``, …).
    """

    experiment_id: str
    objective: str
    factor_names: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonify({
            "schema": "doekit.ExperimentRecord/1",
            "experiment_id": self.experiment_id,
            "objective": self.objective,
            "factor_names": list(self.factor_names),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        })

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        return cls(
            experiment_id=str(d.get("experiment_id", "")),
            objective=str(d.get("objective", "")),
            factor_names=list(d.get("factor_names", []) or []),
            metrics=dict(d.get("metrics", {}) or {}),
            metadata=dict(d.get("metadata", {}) or {}),
        )


@dataclass
class PriorEstimate:
    """Priors transferred from similar past experiments."""

    objective: str
    n_sources: int
    expected_delta_d_efficiency: float = 0.0
    expected_delta_mean_power: float = 0.0
    expected_uncertainty: float = 0.5
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonify({
            "schema": "doekit.PriorEstimate/1",
            "objective": self.objective,
            "n_sources": self.n_sources,
            "expected_delta_d_efficiency": self.expected_delta_d_efficiency,
            "expected_delta_mean_power": self.expected_delta_mean_power,
            "expected_uncertainty": self.expected_uncertainty,
            "metadata": dict(self.metadata),
        })


@dataclass
class HistoricalRecommendation:
    """Actionable advice derived from similar history."""

    title: str
    rationale: str
    actions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _jsonify({
            "schema": "doekit.HistoricalRecommendation/1",
            "title": self.title,
            "rationale": self.rationale,
            "actions": list(self.actions),
        })


class ExperimentHistory:
    """A small, similarity-searchable collection of :class:`ExperimentRecord`."""

    def __init__(self, records: Optional[list] = None):
        self._records: dict = {}
        for r in records or []:
            self.add(r)

    def add(self, record: ExperimentRecord) -> None:
        self._records[record.experiment_id] = record

    def all(self) -> list:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def find_similar(self, objective: str, factor_names, top_k: int = 5) -> list:
        """Rank records by objective match (0.6) + factor overlap (0.4)."""
        target = {f.lower() for f in factor_names}
        scored = []
        for rec in self._records.values():
            score = 0.6 if rec.objective.lower() == objective.lower() else 0.0
            if target:
                rec_f = {f.lower() for f in rec.factor_names}
                score += 0.4 * (len(target & rec_f) / len(target))
            if score > 0.0:
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    # -- construction from the traceable workspace --------------------------
    @classmethod
    def from_project(cls, project) -> "ExperimentHistory":
        """Build history from an :class:`ExperimentProject`'s waves (best-effort)."""
        records = []
        for wave in project.waves():
            rec = _record_from_wave(wave)
            if rec is not None:
                records.append(rec)
        return cls(records)


def _record_from_wave(wave) -> Optional[ExperimentRecord]:
    """Extract an :class:`ExperimentRecord` from a wave's snapshot / conclusions."""
    try:
        exp = wave.load_experiment()
    except (FileNotFoundError, ValueError, KeyError):
        return None
    meta = dict(getattr(exp, "metadata", {}) or {})
    objective = str(meta.get("goal") or meta.get("objective") or "")
    factor_names = list(getattr(exp.design, "factor_names", []) or [])
    metrics: dict = {}
    ev = getattr(exp, "evaluation", None)
    if ev is not None:
        eff = getattr(ev, "efficiencies", {}) or {}
        if eff.get("D_efficiency") is not None:
            metrics["D_efficiency"] = eff["D_efficiency"]
    return ExperimentRecord(
        experiment_id=getattr(wave, "wave_id", "") or str(id(wave)),
        objective=objective, factor_names=factor_names, metrics=metrics,
        metadata={"source": "wave"},
    )


def learn_priors(history: ExperimentHistory, objective: str, factor_names,
                 top_k: int = 5) -> PriorEstimate:
    """Average outcome signals over similar past experiments into priors."""
    similar = history.find_similar(objective, factor_names, top_k=top_k)
    if not similar:
        return PriorEstimate(objective=objective, n_sources=0,
                             metadata={"fallback": True})
    d_eff = [r.metrics.get("delta_D_efficiency", 0.0) for r in similar]
    power = [r.metrics.get("delta_mean_power", 0.0) for r in similar]
    unc = [r.metrics.get("uncertainty", 0.5) for r in similar]
    return PriorEstimate(
        objective=objective, n_sources=len(similar),
        expected_delta_d_efficiency=mean(d_eff),
        expected_delta_mean_power=mean(power),
        expected_uncertainty=mean(unc),
        metadata={"top_k": top_k},
    )


def historical_recommendation(history: ExperimentHistory, objective: str,
                              factor_names, top_k: int = 5) -> HistoricalRecommendation:
    """Turn similar history into a short strategy suggestion for a new case."""
    similar = history.find_similar(objective, factor_names, top_k=top_k)
    if not similar:
        return HistoricalRecommendation(
            title="No comparable history",
            rationale="No similar past experiments to transfer strategy from.",
            actions=[
                "Start conservatively and gather evidence in the first waves.",
                "Record metrics to enable meta-learning later.",
            ],
        )
    avg_d = mean(r.metrics.get("delta_D_efficiency", 0.0) for r in similar)
    if avg_d >= 5.0:
        return HistoricalRecommendation(
            title="History favors expansion",
            rationale=f"Similar experiments show a mean D-efficiency gain of {avg_d:.2f}.",
            actions=[
                "Prioritize design expansion in early iterations.",
                "Use a moderate convergence threshold.",
            ],
        )
    return HistoricalRecommendation(
        title="History suggests caution",
        rationale=f"Mean D-efficiency gain in similar history is limited ({avg_d:.2f}).",
        actions=[
            "Refine the model before adding runs.",
            "Use a stricter convergence stop threshold.",
        ],
    )
