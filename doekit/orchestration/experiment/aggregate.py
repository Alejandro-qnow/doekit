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
    """Compute per-run Derringer-style desirability across responses.

    For each response column, values are scaled to ``[0, 1]`` within the
    observed range (maximize) or inverted (minimize). Overall desirability is
    the geometric mean across responses.

    Formulas
    --------
    - Per column ``j``: ``d_j = (x - min) / (max - min)`` (or ``1 - d_j`` for min goals).
    - Overall: ``D = exp(mean(log(d_j)))`` (geometric mean).

    Parameters
    ----------
    frame : DataFrame
        Multi-response matrix (n_runs x k).
    goals : mapping, optional
        ``{column: "max"|"min"}``; defaults to maximize every column.

    Returns
    -------
    Series
        Overall desirability per run (name ``"desirability"``).

    Examples
    --------
    >>> import doekit as ed
    >>> import pandas as pd
    >>> df = pd.DataFrame({"y1": [1.0, 2.0, 3.0], "y2": [3.0, 2.0, 1.0]})
    >>> d = ed.desirability_scores(df)
    >>> len(d) == 3 and float(d.max()) <= 1.0
    True
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
    """End-to-end experiment handle: design through report and persistence.

    Stateful aggregate composing recommend, evaluate, ingest, sequential propose,
    decision, and export without embedding presentation side-effects in domain
    functions.

    Attributes
    ----------
    design : Design
        Current experimental design.
    model : Model, optional
        Fitted model specification.
    response_names : list of str
        Names of response columns.
    responses : DataFrame, optional
        Multi-column ingested responses.
    evaluation : DesignEvaluation, optional
        Cached design-quality evaluation.
    fits : dict
        Per-response :class:`~doekit.assessment.analysis.FitResult` objects.
    recommendation : Recommendation, optional
        Advisor result when built via :meth:`from_goal`.
    metadata : dict
        Goal, budget, and other experiment context.
    response : ndarray, optional
        Primary (first) response vector for sequential / report.
    fit : FitResult, optional
        Fit for the primary response.

    Examples
    --------
    >>> import doekit as ed
    >>> exp = ed.experiment(goal="screening", factors=4, budget=12)
    >>> ev = exp.evaluate(seed=0)
    >>> ev.d_efficiency > 0
    True
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
        """Build an experiment from :func:`~doekit.recommend_design`.

        Parameters
        ----------
        goal : str
            ``"screening"`` or ``"optimization"``.
        factors
            Factor count or explicit factor specification.
        budget : int, optional
            Run budget passed to the advisor.
        model_order : str, optional
            Model order for the advisor.
        responses : sequence of str, optional
            Response column names (default ``["y"]``).
        **kwargs
            Forwarded to :func:`~doekit.recommend_design`.

        Returns
        -------
        Experiment
            Initialized with design, model, and recommendation metadata.

        Examples
        --------
        >>> import doekit as ed
        >>> exp = ed.Experiment.from_goal("screening", factors=3, budget=12, seed=0)
        >>> exp.design.n_runs > 0
        True
        """
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
        """Wrap an existing :class:`Design`.

        Parameters
        ----------
        design : Design
            Design to manage.
        model : Model, optional
            Model specification; taken from ``design.model`` if omitted.
        responses : sequence of str, optional
            Response column names (default ``["y"]``).

        Returns
        -------
        Experiment

        Examples
        --------
        >>> import doekit as ed
        >>> d = ed.plackett_burman(6)
        >>> exp = ed.Experiment.from_design(d)
        >>> exp.design.n_runs == d.n_runs
        True
        """
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
        """Run design-quality evaluation and cache the result.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~doekit.evaluate` (``n_region``, ``seed``, etc.).

        Returns
        -------
        DesignEvaluation
            Cached on ``self.evaluation``.

        Examples
        --------
        >>> import doekit as ed
        >>> exp = ed.experiment(design=ed.plackett_burman(6))
        >>> ev = exp.evaluate(n_region=500, seed=0)
        >>> ev.n_runs == exp.design.n_runs
        True
        """
        self.evaluation = evaluate(self.design, model=self.model, **kwargs)
        return self.evaluation

    def ingest(self, response, *, fit: bool = True, **kwargs) -> "Experiment":
        """Attach measured responses and optionally fit the model.

        Parameters
        ----------
        response : array-like, Mapping, or DataFrame
            Measured responses (1d, 2d, dict, or DataFrame).
        fit : bool, default True
            When ``True``, fit a linear model per response column.
        **kwargs
            Forwarded to :func:`~doekit.fit_linear_model`.

        Returns
        -------
        Experiment
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If response row count does not match ``design.n_runs``.

        Examples
        --------
        >>> import doekit as ed
        >>> import numpy as np
        >>> exp = ed.experiment(design=ed.plackett_burman(6))
        >>> exp.ingest(np.random.default_rng(0).normal(size=6))
        >>> exp.response is not None
        True
        """
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
        """Overall desirability across ingested multi-response data.

        Parameters
        ----------
        goals : mapping, optional
            ``{column: "max"|"min"}`` per response.

        Returns
        -------
        Series
            Per-run desirability from :func:`desirability_scores`.

        Raises
        ------
        ValueError
            If :meth:`ingest` has not been called.

        Examples
        --------
        >>> import doekit as ed
        >>> import numpy as np
        >>> exp = ed.experiment(design=ed.plackett_burman(6), responses=["y1", "y2"])
        >>> exp.ingest(np.column_stack([
        ...     np.random.default_rng(0).normal(size=6),
        ...     np.random.default_rng(1).normal(size=6),
        ... ]))
        >>> len(exp.desirability()) == 6
        True
        """
        if self.responses is None:
            raise ValueError("call ingest(...) before desirability()")
        return desirability_scores(self.responses, goals=goals)

    def multi_response_summary(self, goals: Optional[Mapping[str, str]] = None) -> dict:
        """Summarize per-response fit quality and overall desirability.

        Parameters
        ----------
        goals : mapping, optional
            Passed to :meth:`desirability` when multiple responses exist.

        Returns
        -------
        dict
            Keys ``per_response`` (R², sigma, dof per column), ``note`` (human
            verdict on which response fits best), and optional ``desirability``
            stats when multiple responses were ingested.

        Raises
        ------
        ValueError
            If :meth:`ingest` with ``fit=True`` has not been called.

        Examples
        --------
        >>> import doekit as ed
        >>> import numpy as np
        >>> exp = ed.experiment(design=ed.plackett_burman(6))
        >>> exp.ingest(np.random.default_rng(0).normal(size=6))
        >>> "per_response" in exp.multi_response_summary()
        True
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

    def next(self, n_add: int = 4, *, intent: str = "learn",
             **kwargs) -> NextRunsProposal:
        """Propose the next batch of runs.

        ``intent="learn"`` augments for information using the primary response;
        ``intent="optimize"`` fits a surrogate and proposes runs that move the
        result (multi-objective when multiple responses were ingested).

        Parameters
        ----------
        n_add : int, default 4
            Number of new runs to propose.
        intent : {"learn", "optimize"}, default "learn"
            Information vs optimization intent.
        **kwargs
            Forwarded to :func:`~doekit.propose_next_runs`.

        Returns
        -------
        NextRunsProposal

        Raises
        ------
        ValueError
            If :meth:`ingest` has not been called.

        Examples
        --------
        >>> import doekit as ed
        >>> import numpy as np
        >>> exp = ed.experiment(design=ed.plackett_burman(6))
        >>> exp.ingest(np.random.default_rng(0).normal(size=6))
        >>> prop = exp.next(n_add=2, seed=0)
        >>> prop.added.n_runs == 2
        True
        """
        if self.response is None:
            raise ValueError("call ingest(y) before next()")
        resp = self.response
        if (intent == "optimize" and self.responses is not None
                and self.responses.shape[1] > 1):
            resp = self.responses
        return propose_next_runs(self.design, response=resp, n_add=n_add,
                                 model=self.model, intent=intent, **kwargs)

    def decide_next(self, n_add: int = 4, *, intent: str = "learn",
                    budget: Optional[int] = None, risk_tolerance: str = "moderate",
                    proposal: Optional[NextRunsProposal] = None,
                    use_calibration: bool = False, history=None,
                    scorer=None, policy=None, **kwargs):
        """Decide the next action (stop / augment / refine / redesign).

        Proposes the next batch (unless ``proposal`` is given) and feeds its
        signals to the decision engine — comparison deltas + ``worth_it`` for
        ``learn``, ``predicted_improvement`` / ``explore_exploit`` for
        ``optimize`` — together with the run budget and design quality. When
        ``history`` is given, convergence is checked and can force a stop.

        Parameters
        ----------
        n_add : int, default 4
            Runs to propose when ``proposal`` is omitted.
        intent : {"learn", "optimize"}, default "learn"
            Passed to :meth:`next`.
        budget : int, optional
            Total run budget; falls back to ``metadata["budget"]``.
        risk_tolerance : {"low", "moderate", "high"}, default "moderate"
            Decision policy sensitivity.
        proposal : NextRunsProposal, optional
            Precomputed proposal; :meth:`next` is called when omitted.
        use_calibration : bool, default False
            Use surrogate calibration for optimize uncertainty.
        history : iterable, optional
            Per-generation values for :func:`~doekit.check_convergence`
            (``best_so_far`` for optimize, a delta metric for learn).
        scorer, policy
            Custom :class:`~doekit.ContinuationScorer` / :class:`~doekit.DecisionPolicy`.
        **kwargs
            Forwarded to :meth:`next` when building the proposal.

        Returns
        -------
        Decision
            Recommended action with diagnostics in ``metadata["diagnostics"]``.

        Examples
        --------
        >>> import doekit as ed
        >>> import numpy as np
        >>> exp = ed.experiment(goal="screening", factors=4, budget=16)
        >>> exp.ingest(np.random.default_rng(0).normal(size=exp.design.n_runs))
        >>> dec = exp.decide_next(n_add=2, seed=0)
        >>> dec.action in ("augment", "refine", "stop", "redesign")
        True
        """
        from ..decide import (  # noqa: PLC0415
            decide_next_action, context_from_proposal, check_convergence,
            diagnose_step,
        )
        if proposal is None:
            proposal = self.next(n_add=n_add, intent=intent, **kwargs)
        budget_total = budget if budget is not None else int(self.metadata.get("budget") or 0)
        ctx = context_from_proposal(
            proposal, budget_total=budget_total, budget_spent=self.design.n_runs,
            risk_tolerance=risk_tolerance, use_calibration=use_calibration,
        )
        if (self.evaluation is not None
                and self.evaluation.efficiencies.get("rank_deficient")):
            ctx.quality = "rank_deficient"
        convergence = None
        if history is not None:
            metric_key = "best_so_far" if intent == "optimize" else "delta_D_efficiency"
            convergence = check_convergence(history, metric_key=metric_key)
        decision = decide_next_action(ctx, scorer=scorer, policy=policy,
                                      convergence=convergence)
        # Graduated from doekit-enhanced: run per-step diagnostics automatically
        # and attach them so every decision carries its warnings (power, G-eff,
        # budget, uncertainty) without an extra call. Reuses the same signals
        # already in the context; non-breaking (rides in metadata).
        report = diagnose_step(
            ctx.metrics, budget_remaining=ctx.budget_remaining,
            uncertainty=ctx.uncertainty, convergence=convergence)
        decision.metadata["diagnostics"] = report.to_dict()
        return decision

    def compare(self, n_add: int = 4, **kwargs) -> DesignComparison:
        """Ask whether ``n_add`` more runs are worth it.

        Parameters
        ----------
        n_add : int, default 4
            Proposed augmentation size.
        **kwargs
            Forwarded to :meth:`next` or :func:`~doekit.augment_design`.

        Returns
        -------
        DesignComparison
            Metric deltas and ``worth_it`` heuristic.

        Examples
        --------
        >>> import doekit as ed
        >>> exp = ed.experiment(design=ed.plackett_burman(6))
        >>> cmp = exp.compare(n_add=2, seed=0)
        >>> "delta" in cmp.to_dict()
        True
        """
        prop = self.next(n_add=n_add, **kwargs) if self.response is not None else None
        if prop is not None:
            return prop.comparison
        from ..sequential import augment_design  # noqa: PLC0415
        combined = augment_design(self.design, n_add=n_add, model=self.model, **kwargs)
        return compare_designs(self.design, combined, model=self.model)

    def report(self, **kwargs) -> Any:
        """Generate an HTML report via the presentation layer.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~doekit.report`.

        Returns
        -------
        Path or object
            Report artifact from the presentation layer.
        """
        from ...presentation.report import report_html  # noqa: PLC0415
        return report_html(self.design, response=self.response,
                           model=self.model, **kwargs)

    def export_csv(self, path: Union[str, Path], **kwargs) -> Path:
        """Write the lab collection template as CSV.

        Parameters
        ----------
        path : str or Path
            Output file path.
        **kwargs
            Forwarded to :func:`~doekit.export_csv`.

        Returns
        -------
        Path
            Written file path.
        """
        from ...presentation.export import export_csv  # noqa: PLC0415
        return export_csv(self.design, path, response_names=self.response_names, **kwargs)

    def export_excel(self, path: Union[str, Path], **kwargs) -> Path:
        """Write the lab collection template as Excel (requires ``doekit[export]``).

        Parameters
        ----------
        path : str or Path
            Output file path.
        **kwargs
            Forwarded to :func:`~doekit.export_excel`.

        Returns
        -------
        Path
            Written file path.
        """
        from ...presentation.export import export_excel  # noqa: PLC0415
        return export_excel(self.design, path, response_names=self.response_names, **kwargs)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (``schema: doekit.Experiment/1``).

        Returns
        -------
        dict
            Snapshot including design, model, responses, evaluation, and fits.
        """
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
        """Rebuild an :class:`Experiment` from :meth:`to_dict` output.

        Parameters
        ----------
        d : mapping
            Serialized experiment snapshot.

        Returns
        -------
        Experiment

        Raises
        ------
        ValueError
            For unsupported schema or missing design.
        """
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
        """Persist the experiment into a workspace wave or project.

        ``target`` may be a :class:`~doekit.presentation.workspace.Wave`, an
        :class:`~doekit.presentation.workspace.ExperimentProject` (creates a new
        wave), or a filesystem path to either.

        Parameters
        ----------
        target : Wave, ExperimentProject, str, or Path
            Persistence destination.
        thresholds : mapping, optional
            Quality thresholds for the wave manifest.
        seed : int, optional
            RNG seed stored in wave metadata.
        write_report : bool, default False
            Generate an HTML report during sync.
        comparison, next_runs
            Optional artifacts to attach to the wave.

        Returns
        -------
        Wave
            Synced wave (or new wave when ``target`` is a project).

        Raises
        ------
        FileNotFoundError
            When ``target`` is neither a project nor a wave directory.
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
        """Load an experiment from a wave directory or snapshot file.

        Parameters
        ----------
        path : str or Path
            Wave directory, project path, or ``experiment.json`` file.

        Returns
        -------
        Experiment

        Raises
        ------
        FileNotFoundError
            When no experiment snapshot is found at ``path``.
        """
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
        """Write automatic conclusions into a wave (semantic handoff artifact).

        Parameters
        ----------
        wave : Wave, str, or Path
            Target wave directory.
        thresholds : mapping, optional
            Quality thresholds for conclusions.
        lang : str, default "en"
            Language for narrative text.
        write_html : bool, default False
            Emit an HTML conclusions page.
        comparison : DesignComparison, optional
            Comparison artifact to include.

        Returns
        -------
        dict
            Conclusions payload written by the wave.
        """
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
    """Factory: ``ed.experiment(...)`` returns an :class:`Experiment`.

    Pass ``design=`` to wrap an existing design, or ``goal`` + ``factors`` to
    run the advisor.

    Parameters
    ----------
    goal : str, default "screening"
        Advisor goal when building from factors.
    factors
        Factor count or specification (required unless ``design`` is given).
    budget : int, optional
        Run budget for the advisor.
    design : Design, optional
        Existing design to wrap.
    model : Model, optional
        Model specification.
    responses : sequence of str, optional
        Response column names.
    **kwargs
        Forwarded to :meth:`Experiment.from_goal`.

    Returns
    -------
    Experiment

    Raises
    ------
    ValueError
        If neither ``factors`` nor ``design`` is provided.

    Examples
    --------
    >>> import doekit as ed
    >>> exp = ed.experiment(goal="screening", factors=4, budget=12, seed=0)
    >>> exp.design.n_runs <= 12
    True
    """
    if design is not None:
        return Experiment.from_design(design, model=model, responses=responses)
    if factors is None:
        raise ValueError("pass factors=... or design=...")
    return Experiment.from_goal(goal=goal, factors=factors, budget=budget,
                                responses=responses, **kwargs)
