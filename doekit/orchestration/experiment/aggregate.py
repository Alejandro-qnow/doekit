"""Stateful experiment aggregate — end-to-end product contract (v0.7).

Composes recommend / evaluate / analyze / sequential / report / export without
embedding presentation side-effects in domain functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...domain.design import Design
from ...domain.model import Model
from ...assessment.evaluation import evaluate, DesignEvaluation
from ...assessment.analysis import fit_linear_model, FitResult
from ..advise import recommend_design, Recommendation
from ..sequential import propose_next_runs, compare_designs, NextRunsProposal, DesignComparison


def _as_response_frame(response, n_runs: int,
                       names: Sequence[str]) -> pd.DataFrame:
    """Normalize array / dict / DataFrame responses to a DataFrame (n_runs x k)."""
    if isinstance(response, pd.DataFrame):
        df = response.copy()
        if len(df) != n_runs:
            raise ValueError(f"response has {len(df)} rows; design has {n_runs}")
        return df
    if isinstance(response, Mapping):
        df = pd.DataFrame(response)
        if len(df) != n_runs:
            raise ValueError(f"response has {len(df)} rows; design has {n_runs}")
        return df
    arr = np.asarray(response, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_runs:
            raise ValueError(
                f"response length ({arr.shape[0]}) must match n_runs ({n_runs})"
            )
        name = names[0] if names else "y"
        return pd.DataFrame({name: arr})
    if arr.ndim == 2:
        if arr.shape[0] != n_runs:
            raise ValueError(
                f"response has {arr.shape[0]} rows; design has {n_runs}"
            )
        cols = list(names) if names and len(names) == arr.shape[1] else [
            f"y{i + 1}" for i in range(arr.shape[1])
        ]
        return pd.DataFrame(arr, columns=cols)
    raise TypeError("response must be 1d/2d array, Mapping, or DataFrame")


def desirability_scores(frame: pd.DataFrame,
                        goals: Optional[Mapping[str, str]] = None) -> pd.Series:
    """Simple per-run Derringer-style desirability (larger-the-better default).

    For each column, values are scaled to ``[0, 1]`` within the observed range
    (maximize) or inverted (minimize). Overall desirability is the geometric
    mean across responses.
    """
    if frame.empty:
        return pd.Series(dtype=float)
    goals = goals or {}
    parts = []
    for col in frame.columns:
        x = frame[col].to_numpy(dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            d = np.ones(len(x), dtype=float)
        else:
            d = (x - lo) / (hi - lo)
            if goals.get(col, "max") == "min":
                d = 1.0 - d
        parts.append(np.clip(d, 1e-12, 1.0))
    stacked = np.vstack(parts)
    # geometric mean
    geo = np.exp(np.mean(np.log(stacked), axis=0))
    return pd.Series(geo, name="desirability")


@dataclass
class Experiment:
    """End-to-end experiment handle: design → evaluate → ingest → next → report.

    Example::

        exp = ed.experiment(goal="screening", factors=6, budget=12)
        exp.evaluate()
        print(exp.plan)                 # run sheet
        exp.export_csv("runs.csv")
        exp.ingest(y)
        nxt = exp.next(n_add=4)
        exp.report(output_dir="report/")
    """

    design: Design
    model: Optional[Model] = None
    response_names: list = field(default_factory=lambda: ["y"])
    responses: Optional[pd.DataFrame] = None
    evaluation: Optional[DesignEvaluation] = None
    fits: dict = field(default_factory=dict)
    recommendation: Optional[Recommendation] = None
    metadata: dict = field(default_factory=dict)
    # back-compat single-response accessors
    response: Optional[np.ndarray] = None
    fit: Optional[FitResult] = None

    @classmethod
    def from_goal(cls, goal: str, factors, budget: Optional[int] = None,
                  model_order: Optional[str] = None,
                  responses: Optional[Sequence[str]] = None,
                  **kwargs) -> "Experiment":
        """Build an experiment from :func:`recommend_design`."""
        rec = recommend_design(goal=goal, factors=factors, budget=budget,
                               model_order=model_order, **kwargs)
        names = list(responses) if responses else ["y"]
        return cls(
            design=rec.design, model=rec.model, recommendation=rec,
            response_names=names,
            metadata={"goal": goal, "budget": budget,
                      "response_names": names},
        )

    @classmethod
    def from_design(cls, design: Design, model: Optional[Model] = None,
                    responses: Optional[Sequence[str]] = None) -> "Experiment":
        """Wrap an existing design."""
        names = list(responses) if responses else ["y"]
        return cls(design=design, model=model or design.model,
                   response_names=names,
                   metadata={"response_names": names})

    @property
    def plan(self) -> pd.DataFrame:
        """Lab collection template (factors + empty response columns)."""
        from ...presentation.export import run_sheet  # noqa: PLC0415
        return run_sheet(self.design, response_names=self.response_names)

    def evaluate(self, **kwargs) -> DesignEvaluation:
        """Run quality evaluation and cache the result."""
        self.evaluation = evaluate(self.design, model=self.model, **kwargs)
        return self.evaluation

    def ingest(self, response, *, fit: bool = True, **kwargs) -> "Experiment":
        """Attach measured responses (1d, 2d, dict or DataFrame); optionally fit."""
        frame = _as_response_frame(response, self.design.n_runs, self.response_names)
        self.responses = frame
        self.response_names = list(frame.columns)
        # primary response for sequential / report
        self.response = frame.iloc[:, 0].to_numpy(dtype=float)
        self.fits = {}
        self.fit = None
        if fit:
            for col in frame.columns:
                fr = fit_linear_model(self.design, frame[col].to_numpy(),
                                      model=self.model, **kwargs)
                self.fits[col] = fr
            self.fit = self.fits[frame.columns[0]]
        self.metadata["response_names"] = list(frame.columns)
        return self

    def desirability(self, goals: Optional[Mapping[str, str]] = None) -> pd.Series:
        """Overall desirability across ingested responses (requires multi-y)."""
        if self.responses is None:
            raise ValueError("call ingest(...) before desirability()")
        return desirability_scores(self.responses, goals=goals)

    def multi_response_summary(self, goals: Optional[Mapping[str, str]] = None) -> dict:
        """Per-response fit quality + which response the design serves best.

        Returns a dict with ``per_response`` (R², sigma) and a short ``note``
        like \"strong on A / weak on B\" for the agent/report layer.
        """
        if not self.fits:
            raise ValueError("call ingest(..., fit=True) first")
        per = {}
        for name, fr in self.fits.items():
            per[name] = {
                "r_squared": float(fr.r_squared) if np.isfinite(fr.r_squared) else None,
                "sigma": (float(np.sqrt(fr.sigma2))
                          if np.isfinite(fr.sigma2) and fr.sigma2 >= 0 else None),
                "dof": int(fr.dof),
            }
        # rank by R²
        ranked = sorted(
            ((k, v["r_squared"] if v["r_squared"] is not None else -1) for k, v in per.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        if len(ranked) == 1:
            note = f"Single response '{ranked[0][0]}' (R²={ranked[0][1]:.3f})."
        else:
            best, worst = ranked[0], ranked[-1]
            note = (f"Stronger fit on '{best[0]}' (R²={best[1]:.3f}); "
                    f"weaker on '{worst[0]}' (R²={worst[1]:.3f}).")
        des = None
        if self.responses is not None and self.responses.shape[1] > 1:
            d = self.desirability(goals=goals)
            des = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
        return {"per_response": per, "note": note, "desirability": des}

    def next(self, n_add: int = 4, **kwargs) -> NextRunsProposal:
        """Propose the next batch of runs (uses primary ingested response)."""
        if self.response is None:
            raise ValueError("call ingest(y) before next()")
        return propose_next_runs(self.design, response=self.response,
                                 n_add=n_add, model=self.model, **kwargs)

    def compare(self, n_add: int = 4, **kwargs) -> DesignComparison:
        """Ask whether ``n_add`` more runs are worth it (via propose + compare)."""
        prop = self.next(n_add=n_add, **kwargs) if self.response is not None else None
        if prop is not None:
            return prop.comparison
        from ..sequential import augment_design  # noqa: PLC0415
        combined = augment_design(self.design, n_add=n_add, model=self.model, **kwargs)
        return compare_designs(self.design, combined, model=self.model)

    def report(self, **kwargs) -> Any:
        """Generate an HTML report via the presentation layer (explicit IO)."""
        from ...presentation.report import report_html  # noqa: PLC0415
        return report_html(self.design, response=self.response,
                           model=self.model, **kwargs)

    def export_csv(self, path: Union[str, Path], **kwargs) -> Path:
        """Write the collection template as CSV."""
        from ...presentation.export import export_csv  # noqa: PLC0415
        return export_csv(self.design, path, response_names=self.response_names, **kwargs)

    def export_excel(self, path: Union[str, Path], **kwargs) -> Path:
        """Write the collection template as Excel (``doekit[export]``)."""
        from ...presentation.export import export_excel  # noqa: PLC0415
        return export_excel(self.design, path, response_names=self.response_names, **kwargs)

    def to_dict(self) -> dict:
        """JSON-safe snapshot (``schema: doekit.Experiment/1``)."""
        from ...shared.serialize import jsonify  # noqa: PLC0415
        multi = None
        if self.fits:
            try:
                multi = self.multi_response_summary()
            except ValueError:
                multi = None
        return jsonify({
            "schema": "doekit.Experiment/1",
            "design": self.design.to_dict() if self.design is not None else None,
            "model": self.model.to_dict() if self.model is not None else None,
            "response_names": list(self.response_names),
            "response": (self.response.tolist()
                         if self.response is not None else None),
            "responses": (self.responses.to_dict("list")
                          if self.responses is not None else None),
            "evaluation": (self.evaluation.to_dict()
                           if self.evaluation is not None else None),
            "fit": self.fit.to_dict() if self.fit is not None else None,
            "fits": {k: v.to_dict() for k, v in self.fits.items()},
            "multi_response": multi,
            "recommendation": (self.recommendation.to_dict()
                               if self.recommendation is not None else None),
            "metadata": dict(self.metadata),
        })

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Experiment":
        """Rebuild an :class:`Experiment` from :meth:`to_dict` output."""
        if d.get("schema") not in (None, "doekit.Experiment/1"):
            raise ValueError(f"unsupported Experiment schema: {d.get('schema')!r}")
        if not d.get("design"):
            raise ValueError("Experiment snapshot missing design")
        design = Design.from_dict(d["design"])
        model = Model.from_dict(d["model"]) if d.get("model") else design.model
        names = list(d.get("response_names") or ["y"])
        exp = cls(
            design=design,
            model=model,
            response_names=names,
            metadata=dict(d.get("metadata") or {"response_names": names}),
        )
        if d.get("evaluation"):
            exp.evaluation = DesignEvaluation.from_dict(d["evaluation"])
        # Prefer multi-column responses; fall back to primary response vector
        if d.get("responses"):
            frame = pd.DataFrame(d["responses"])
            exp.responses = frame
            exp.response_names = list(frame.columns)
            exp.response = frame.iloc[:, 0].to_numpy(dtype=float)
        elif d.get("response") is not None:
            arr = np.asarray(d["response"], dtype=float)
            name = names[0] if names else "y"
            exp.responses = pd.DataFrame({name: arr})
            exp.response = arr
            exp.response_names = [name]
        fits_raw = d.get("fits") or {}
        if fits_raw:
            exp.fits = {k: FitResult.from_dict(v) for k, v in fits_raw.items()}
            first = exp.response_names[0] if exp.response_names else next(iter(exp.fits))
            exp.fit = exp.fits.get(first) or next(iter(exp.fits.values()))
        elif d.get("fit"):
            exp.fit = FitResult.from_dict(d["fit"])
            key = exp.response_names[0] if exp.response_names else "y"
            exp.fits = {key: exp.fit}
        # Recommendation is optional narrative; keep payload in metadata if present
        if d.get("recommendation"):
            exp.metadata = {
                **exp.metadata,
                "recommendation_snapshot": d["recommendation"],
            }
        return exp

    def save(
        self,
        target: Union[str, Path, Any],
        *,
        thresholds: Optional[Mapping[str, float]] = None,
        seed: Optional[int] = None,
        write_report: bool = False,
        comparison=None,
        next_runs=None,
    ) -> Any:
        """Persist into a :class:`~doekit.presentation.workspace.Wave` or project.

        ``target`` may be a :class:`~doekit.presentation.workspace.Wave`, an
        :class:`~doekit.presentation.workspace.ExperimentProject` (creates a new
        wave), or a filesystem path to either.
        """
        from ...presentation.workspace import (  # noqa: PLC0415
            ExperimentProject, Wave, open_project,
        )
        if isinstance(target, Wave):
            wave = target
        elif isinstance(target, ExperimentProject):
            wave = target.new_wave(self, thresholds=thresholds, seed=seed)
            return wave
        else:
            path = Path(target)
            if (path / "PROJECT.json").exists():
                proj = open_project(path)
                return proj.new_wave(self, thresholds=thresholds, seed=seed)
            if (path / "manifest.json").exists():
                wave = Wave(path)
            else:
                raise FileNotFoundError(
                    f"save target is neither a project nor a wave: {path}"
                )
        wave.sync(
            self,
            write_report=write_report,
            comparison=comparison,
            next_runs=next_runs,
            thresholds=thresholds,
            seed=seed,
        )
        return wave

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Experiment":
        """Load from a wave directory or ``doe-configuration/experiment.json``."""
        from ...presentation.workspace import Wave  # noqa: PLC0415
        path = Path(path)
        if path.is_file() and path.name == "experiment.json":
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if (path / "manifest.json").exists():
            return Wave(path).load_experiment()
        cfg = path / "doe-configuration" / "experiment.json"
        if cfg.exists():
            return cls.from_dict(json.loads(cfg.read_text(encoding="utf-8")))
        raise FileNotFoundError(f"cannot load Experiment from {path}")

    def conclude(
        self,
        wave: Union[str, Path, Any],
        *,
        thresholds: Optional[Mapping[str, float]] = None,
        lang: str = "en",
        write_html: bool = False,
        comparison=None,
    ) -> dict:
        """Write automatic conclusions into a wave (semantic handoff artifact)."""
        from ...presentation.workspace import Wave  # noqa: PLC0415
        if not isinstance(wave, Wave):
            wave = Wave(wave)
        return wave.conclude(
            self,
            thresholds=thresholds,
            lang=lang,
            write_html=write_html,
            comparison=comparison,
        )


def experiment(goal: str = "screening", factors=None, budget: Optional[int] = None,
               design: Optional[Design] = None, model: Optional[Model] = None,
               responses: Optional[Sequence[str]] = None, **kwargs) -> Experiment:
    """Factory: ``ed.experiment(...)`` → :class:`Experiment`.

    Pass ``design=`` to wrap an existing design, or ``goal`` + ``factors`` to
    run the advisor.
    """
    if design is not None:
        return Experiment.from_design(design, model=model, responses=responses)
    if factors is None:
        raise ValueError("pass factors=... or design=...")
    return Experiment.from_goal(goal=goal, factors=factors, budget=budget,
                                responses=responses, **kwargs)
