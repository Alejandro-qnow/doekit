"""Sequential / adaptive DoE: augment designs and propose the next run batch.

Closes the loop ``evaluate → run → analyze → propose_next`` while keeping the
same quality language as :mod:`doekit.assessment.evaluation` (D/A/G-efficiency, SPV, power).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ...domain import criteria as _crit
from ...domain.criteria.base import CriterionContext, score_criterion
from ...domain.model import Model
from ...domain.design import Design
from ...generation.catalog.random_design import random_design
from ...generation.catalog.factorial import full_factorial
from ...assessment.evaluation import efficiencies, power_analysis
from ...assessment.units import resolve_model as _resolve_model
from ...domain.factors import ContinuousFactor, DiscreteFactor, CategoricalFactor
from ...shared.serialize import jsonify as _jsonify


def _factor_columns(design: Design) -> list[str]:
    """Factor columns only (drop block / grouping metadata columns)."""
    drop = set()
    blocking = design.metadata.get("blocking")
    if isinstance(blocking, str):
        drop.add(blocking)
    elif isinstance(blocking, dict) and blocking.get("column"):
        drop.add(blocking["column"])
    return [c for c in design.matrix.columns if c not in drop]


def _default_candidates(design: Design, model: Model, n_cand: int,
                        seed: Optional[int]) -> Design:
    """Build a candidate set covering the factor region of ``design``."""
    facs = list(design.factors or [])
    names = _factor_columns(design)
    if not facs:
        # Infer continuous factors from column ranges
        for name in names:
            col = design.matrix[name]
            if pd.api.types.is_numeric_dtype(col):
                lo, hi = float(col.min()), float(col.max())
                if lo == hi:
                    lo, hi = lo - 1.0, hi + 1.0
                facs.append(ContinuousFactor(name, lo, hi))
            else:
                facs.append(CategoricalFactor(name, list(pd.unique(col))))

    # Prefer a modest grid when the combinatorial size is small
    try:
        levels = {}
        ok_grid = True
        for f in facs:
            if isinstance(f, ContinuousFactor):
                levels[f.name] = [f.low, (f.low + f.high) / 2.0, f.high]
            elif isinstance(f, DiscreteFactor):
                levels[f.name] = list(f.levels)
            elif isinstance(f, CategoricalFactor):
                levels[f.name] = list(f.levels)
            else:
                ok_grid = False
                break
        ncomb = int(np.prod([len(v) for v in levels.values()])) if ok_grid else 10**9
        if ok_grid and ncomb <= max(n_cand, 81):
            cand = full_factorial(levels)
        else:
            cand = random_design(facs, n=n_cand, seed=seed)
    except (ValueError, TypeError, KeyError):
        cand = random_design(facs, n=n_cand, seed=seed)

    keep = [c for c in names if c in cand.matrix.columns]
    matrix = cand.matrix.loc[:, keep].copy() if keep else cand.matrix
    return cand.replace(factors=facs, model=model, matrix=matrix)


def _score(X: np.ndarray, criterion: str, X_region: Optional[np.ndarray]) -> float:
    c = criterion.strip().upper()
    ctx = CriterionContext(moment_matrix=X_region if c == "I" else None)
    return score_criterion(_crit.get_criterion(c), X, ctx)


def _select_augment_rows(X_fixed: np.ndarray, X_cand: np.ndarray, n_add: int,
                         criterion: str = "D", n_starts: int = 3,
                         seed: Optional[int] = None,
                         max_exchange: int = 200) -> list[int]:
    """Greedy + exchange selection of ``n_add`` candidate rows given fixed rows."""
    Nc = X_cand.shape[0]
    if n_add <= 0:
        return []
    if n_add > Nc:
        raise ValueError(f"n_add={n_add} exceeds candidate set size ({Nc})")

    rng = np.random.default_rng(seed)
    X_region = X_cand  # I-opt moments over the candidate region

    def score_idx(idx):
        X = np.vstack([X_fixed, X_cand[idx]]) if len(X_fixed) else X_cand[idx]
        return _score(X, criterion, X_region)

    best_idx, best_s = None, -np.inf
    for start in range(max(1, n_starts)):
        # --- greedy growth ---
        available = list(range(Nc))
        rng.shuffle(available)
        selected: list[int] = []
        # diversify starts: pick a random first point
        if available:
            first = int(available.pop(rng.integers(0, len(available))))
            selected.append(first)
        while len(selected) < n_add:
            best_i, best_local = None, -np.inf
            for i in available:
                s = score_idx(selected + [i])
                if s > best_local:
                    best_local, best_i = s, i
            if best_i is None:
                break
            selected.append(best_i)
            available.remove(best_i)

        # --- exchange among the new points only (fixed stays fixed) ---
        for _ in range(max_exchange):
            improved = False
            cur = score_idx(selected)
            for j, old in enumerate(list(selected)):
                for i in available:
                    trial = list(selected)
                    trial[j] = i
                    s = score_idx(trial)
                    if s > cur + 1e-12:
                        available.append(old)
                        available.remove(i)
                        selected[j] = i
                        cur = s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        s = score_idx(selected)
        if s > best_s:
            best_s, best_idx = s, list(selected)

    return best_idx or []


def augment_design(design: Design, n_add: int, model: Optional[Model] = None,
                   criterion: str = "D", candidates: Optional[Design] = None,
                   n_candidates: int = 200, n_starts: int = 5,
                   seed: Optional[int] = None) -> Design:
    """Augment an existing design with ``n_add`` D/I-optimal new runs.

    The current runs are **fixed**; new points are chosen from a candidate set
    to maximize the chosen criterion on the *combined* design.

    Parameters
    ----------
    design : Design
        Already-executed (or planned) design to keep.
    n_add : int
        Number of new runs to append.
    model : Model, optional
        Model for the information matrix; taken from ``design.model`` if omitted.
    criterion : {"D", "I", "A", "G", "E", "T"}, default "D"
        Optimality criterion for the combined design.
    candidates : Design, optional
        Candidate set; a grid/random cover of the factor region is built if omitted.
    n_candidates : int, default 200
        Size of the auto-generated candidate set when ``candidates`` is omitted.
    n_starts : int, default 5
        Independent greedy starts (best is kept).
    seed : int, optional
        RNG seed.

    Returns
    -------
    Design
        Combined design (original rows + new rows). Metadata includes
        ``n_original``, ``n_added``, ``added_rows`` (relative to candidates) and
        ``criterion``.
    """
    if n_add < 1:
        raise ValueError("n_add must be >= 1")
    model = _resolve_model(design, model)
    cols = _factor_columns(design)
    fixed_frame = design.matrix.loc[:, cols]
    X_fixed = np.asarray(model.matrix(fixed_frame), dtype=float)

    cand = candidates or _default_candidates(design, model, n_candidates, seed)
    cand_frame = cand.matrix.loc[:, [c for c in cols if c in cand.matrix.columns]]
    if cand_frame.shape[1] != len(cols):
        # align missing columns
        for c in cols:
            if c not in cand_frame.columns:
                raise ValueError(
                    f"candidate set missing factor column {c!r}; "
                    "pass candidates with the same factors as design"
                )
        cand_frame = cand_frame.loc[:, cols]
    X_cand = np.asarray(model.matrix(cand_frame), dtype=float)

    idx = _select_augment_rows(X_fixed, X_cand, n_add, criterion=criterion,
                               n_starts=n_starts, seed=seed)
    new_rows = cand_frame.iloc[idx].reset_index(drop=True)
    combined = pd.concat([fixed_frame.reset_index(drop=True), new_rows],
                         ignore_index=True)
    meta = dict(design.metadata)
    meta.update({
        "kind": "AugmentedDesign",
        "n_original": int(design.n_runs),
        "n_added": int(n_add),
        "added_from_candidates": list(idx),
        "criterion": criterion.strip().upper(),
        "parent_kind": design.metadata.get("kind"),
    })
    return Design(matrix=combined, factors=list(design.factors or cand.factors or []),
                  model=model, metadata=meta)


# ---------------------------------------------------------------------------
# compare + propose
# ---------------------------------------------------------------------------

@dataclass
class DesignComparison:
    """Side-by-side quality diff of two designs (``compare_designs``).

    Attributes
    ----------
    a_label, b_label : str
        Labels for the two designs.
    a, b : dict
        Efficiency dicts (and mean power) for each design.
    delta : dict
        ``b - a`` for numeric metrics (positive Δ D-eff = B is better on D).
    worth_it : bool or None
        Heuristic: True if B gains precision/prediction enough to justify extra runs.
    summary : str
        One-line human verdict.
    """

    a_label: str
    b_label: str
    a: dict
    b: dict
    delta: dict
    worth_it: Optional[bool] = None
    summary: str = ""
    table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        """Serialize (``schema: doekit.DesignComparison/1``)."""
        return _jsonify({
            "schema": "doekit.DesignComparison/1",
            "a_label": self.a_label,
            "b_label": self.b_label,
            "a": self.a,
            "b": self.b,
            "delta": self.delta,
            "worth_it": self.worth_it,
            "summary": self.summary,
            "table": self.table.to_dict("records"),
        })


def compare_designs(a: Design, b: Design, model: Optional[Model] = None,
                    effect_size=1.0, sigma: float = 1.0, alpha: float = 0.05,
                    n_region: int = 4000, seed: Optional[int] = None,
                    a_label: str = "current", b_label: str = "proposed",
                    run_cost: float = 1.0) -> DesignComparison:
    """Compare two designs on efficiencies, SPV and mean power.

    Answers *"is it worth paying for the extra runs in B?"* with a transparent
    metric delta table (same language as :func:`~doekit.assessment.evaluation.evaluate`).

    Parameters
    ----------
    a, b : Design
        Designs to compare (typically current vs augmented).
    model : Model, optional
        Shared model; resolved from ``a`` / ``b`` if omitted.
    effect_size, sigma, alpha
        Power-analysis assumptions.
    n_region, seed
        Region sampling for G/I metrics.
    a_label, b_label : str
        Display labels.
    run_cost : float, default 1.0
        Relative cost per extra run (used only in the heuristic verdict).

    Returns
    -------
    DesignComparison
    """
    model = model or a.model or b.model or _resolve_model(a, None)
    ea = efficiencies(a, model=model, n_region=n_region, seed=seed)
    eb = efficiencies(b, model=model, n_region=n_region, seed=seed)

    def _mean_power(d):
        try:
            pw = power_analysis(d, model=model, effect_size=effect_size,
                                sigma=sigma, alpha=alpha)
            vals = pw.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if len(vals) else float("nan")
        except ValueError:
            return float("nan")

    pa, pb = _mean_power(a), _mean_power(b)
    ea = {**ea, "mean_power": pa}
    eb = {**eb, "mean_power": pb}

    keys = ["n_runs", "D_efficiency", "A_efficiency", "G_efficiency",
            "spv_mean", "mean_power"]
    delta = {}
    rows = []
    for k in keys:
        va, vb = ea.get(k), eb.get(k)
        try:
            dlt = float(vb) - float(va)
        except (TypeError, ValueError):
            dlt = float("nan")
        delta[k] = dlt
        rows.append({"metric": k, a_label: va, b_label: vb, "delta": dlt})

    extra_runs = int(eb["n_runs"] - ea["n_runs"])
    # Heuristic: worth it if D or G rises >= 5 pts, or mean SPV drops >= 10%,
    # or mean power rises >= 0.05 — relative to extra_runs * run_cost.
    d_gain = delta.get("D_efficiency") or 0.0
    g_gain = delta.get("G_efficiency") or 0.0
    spv_a, spv_b = ea.get("spv_mean"), eb.get("spv_mean")
    spv_improve = (
        (spv_a - spv_b) / spv_a
        if spv_a and spv_a > 0 and spv_b is not None and np.isfinite(spv_b)
        else 0.0
    )
    pow_gain = delta.get("mean_power") or 0.0
    quality = (d_gain >= 5) or (g_gain >= 5) or (spv_improve >= 0.10) or (pow_gain >= 0.05)
    if extra_runs <= 0:
        worth = True if quality else None
        summary = (f"{b_label} improves quality without extra runs."
                   if quality else f"{b_label} does not clearly beat {a_label}.")
    else:
        # penalize large batches: need stronger gains
        threshold = 1.0 + 0.15 * extra_runs * run_cost
        score = (max(0, d_gain) / 5 + max(0, g_gain) / 5
                 + max(0, spv_improve) / 0.10 + max(0, pow_gain) / 0.05)
        worth = bool(score >= threshold and quality)
        summary = (
            f"{'Yes' if worth else 'Maybe not'}: {extra_runs} extra run(s) "
            f"(ΔD={d_gain:+.1f} pts, ΔG={g_gain:+.1f} pts, "
            f"ΔSPV_mean={delta.get('spv_mean'):+.3g}, Δpower={pow_gain:+.3f})."
        )

    return DesignComparison(
        a_label=a_label, b_label=b_label, a=ea, b=eb, delta=delta,
        worth_it=worth, summary=summary, table=pd.DataFrame(rows),
    )


@dataclass
class NextRunsProposal:
    """Result of :func:`propose_next_runs`.

    Attributes
    ----------
    added : Design
        Design containing **only** the proposed new runs.
    combined : Design
        Original design with the new runs appended.
    comparison : DesignComparison
        Metric delta current vs combined.
    criterion : str
        Criterion used for augmentation.
    rationale : str
        Human-readable justification.
    caveats : list of str
        Assumptions and limitations.
    active_terms : list of str
        Terms flagged as active when ``response`` was provided (empty otherwise).
    sigma_hat : float or None
        Residual sigma from the fit when ``response`` was provided.
    """

    added: Design
    combined: Design
    comparison: DesignComparison
    criterion: str
    rationale: str
    caveats: list = field(default_factory=list)
    active_terms: list = field(default_factory=list)
    sigma_hat: Optional[float] = None
    # --- optimize intent (surrogate + acquisition); None for the learn path ---
    intent: str = "learn"
    acquisition: Optional[str] = None
    best_so_far: Optional[object] = None
    predicted_improvement: Optional[float] = None
    pareto_front: Optional[list] = None
    explore_exploit: Optional[dict] = None
    surrogate: object = field(default=None, repr=False)
    acquisition_values: Optional[np.ndarray] = field(default=None, repr=False)

    def _surrogate_summary(self) -> Optional[dict]:
        if self.surrogate is None:
            return None
        sur = self.surrogate
        summary = {"kind": type(sur).__name__}
        model = getattr(sur, "model", None)
        if model is not None:
            summary["model"] = repr(model)
        try:
            summary["calibration"] = sur.calibration()
        except (ValueError, AttributeError, np.linalg.LinAlgError):
            summary["calibration"] = None
        return summary

    def to_dict(self) -> dict:
        """Serialize (``schema: doekit.NextRunsProposal/1``)."""
        return _jsonify({
            "schema": "doekit.NextRunsProposal/1",
            "intent": self.intent,
            "criterion": self.criterion,
            "rationale": self.rationale,
            "caveats": list(self.caveats),
            "active_terms": list(self.active_terms),
            "sigma_hat": self.sigma_hat,
            "acquisition": self.acquisition,
            "best_so_far": self.best_so_far,
            "predicted_improvement": self.predicted_improvement,
            "pareto_front": self.pareto_front,
            "explore_exploit": self.explore_exploit,
            "surrogate": self._surrogate_summary(),
            "added": self.added.to_dict(),
            "combined": self.combined.to_dict(),
            "comparison": self.comparison.to_dict(),
        })

    def summary(self) -> str:
        if self.intent == "optimize":
            head = (f"Next runs proposal ({self.added.n_runs} new, "
                    f"intent=optimize, acquisition={self.acquisition})")
        else:
            head = (f"Next runs proposal ({self.added.n_runs} new, "
                    f"criterion={self.criterion})")
        lines = [head, "-" * 48, self.rationale, "", self.comparison.summary, ""]
        if self.intent == "optimize" and self.best_so_far is not None:
            lines.append(f"Best so far: {self.best_so_far}")
            if self.predicted_improvement is not None:
                lines.append(
                    f"Predicted improvement of best candidate: "
                    f"{self.predicted_improvement:+.4g}"
                )
            lines.append("")
        lines += ["Proposed runs:", repr(self.added.matrix)]
        if self.caveats:
            lines += ["", "Caveats:"] + [f"  - {c}" for c in self.caveats]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def propose_next_runs(design: Design, response=None, n_add: int = 4,
                      model: Optional[Model] = None, criterion: str = "D",
                      candidates: Optional[Design] = None,
                      budget: Optional[int] = None,
                      priorities: Optional[dict] = None,
                      effect_size=1.0, sigma: float = 1.0,
                      alpha: float = 0.05, n_region: int = 4000,
                      n_candidates: int = 200, n_starts: int = 5,
                      seed: Optional[int] = None,
                      active_p: float = 0.05, *,
                      intent: str = "learn",
                      objectives: Optional[list] = None,
                      goals: Optional[dict] = None,
                      goal: str = "max",
                      acquisition: Optional[str] = None,
                      surrogate: str = "auto",
                      kappa: float = 2.0, xi: float = 0.01) -> NextRunsProposal:
    """Propose the next batch of runs for a sequential experiment.

    Two intents share one entry point:

    - ``intent="learn"`` (default) — classical **D/I-optimal augmentation**:
      runs that sharpen the *model* (unchanged behavior). Without ``response``
      it augments by information; with ``response`` it estimates residual sigma
      and flags active terms (p < ``active_p``).
    - ``intent="optimize"`` — **surrogate + acquisition**: fits a
      :class:`~doekit.assessment.surrogate.Surrogate` (GP with the OLS surface as
      prior mean, or plain OLS), then proposes runs that move the *result*
      toward the optimum. Supports multi-objective via Pareto/EHVI.

    Parameters
    ----------
    design : Design
        Current design (runs already done or committed).
    response : array-like, optional
        Measured responses. For ``intent="optimize"`` this is required and may
        be multi-column (2d array / DataFrame) for multi-objective.
    n_add : int, default 4
        Number of new runs to propose. Capped by ``budget - n_runs`` when set.
    model : Model, optional
        Model for augmentation / the surrogate prior mean.
    criterion : str, default "D"
        Augmentation criterion (learn intent only).
    candidates : Design, optional
        Candidate set for new runs (both intents).
    budget : int, optional
        Maximum total runs (current + new).
    priorities : dict, optional
        Reserved for future ranking of criteria; accepted for API stability.
    effect_size, sigma, alpha
        Power assumptions (``sigma`` overridden by residual sigma when
        ``response`` is given and dof > 0).
    n_region, n_candidates, n_starts, seed
        Evaluation / search controls.
    active_p : float, default 0.05
        p-value cutoff for listing active terms when ``response`` is given.
    intent : {"learn", "optimize"}, default "learn"
        Whether to augment for information or optimize the response.
    objectives : list of str, optional
        Column names when ``response`` is multi-column (optimize intent).
    goals : dict, optional
        ``{column: "max"|"min"}`` per objective (optimize intent).
    goal : {"max", "min"}, default "max"
        Direction for a single objective when ``goals`` is not given.
    acquisition : str, optional
        Acquisition function: ``"ei"`` / ``"ucb"`` / ``"pi"`` (single) or
        ``"ehvi"`` (multi). Defaults to ``"ei"`` (single) / ``"ehvi"`` (multi).
    surrogate : {"auto", "ols", "gp"}, default "auto"
        Surrogate backend (optimize intent).
    kappa : float, default 2.0
        UCB exploration weight.
    xi : float, default 0.01
        EI/PI exploration margin.

    Returns
    -------
    NextRunsProposal
    """
    _ = priorities  # API placeholder for multi-objective next-batch ranking
    intent = str(intent).strip().lower()
    if intent not in ("learn", "optimize"):
        raise ValueError(f"intent must be 'learn' or 'optimize', got {intent!r}")
    model = _resolve_model(design, model)

    if budget is not None:
        remaining = budget - design.n_runs
        if remaining <= 0:
            raise ValueError(
                f"budget={budget} already exhausted (n_runs={design.n_runs})"
            )
        n_add = min(n_add, remaining)

    if intent == "optimize":
        return _propose_optimize(
            design, response=response, n_add=n_add, model=model,
            candidates=candidates, objectives=objectives, goals=goals,
            goal=goal, acquisition=acquisition, surrogate_kind=surrogate,
            kappa=kappa, xi=xi, n_candidates=n_candidates, n_region=n_region,
            seed=seed,
        )

    caveats = [
        "Proposal is conditional on the assumed model and criterion; "
        "re-run after updating the model if the active set changes.",
        "Augmentation is classical (D/I-optimal conditioned on current runs), "
        "not a black-box Bayesian optimizer — use intent='optimize' for a "
        "surrogate + acquisition loop.",
    ]

    sigma_hat = None
    active_terms: list[str] = []
    if response is not None:
        from ...assessment.analysis import fit_linear_model  # noqa: PLC0415
        y = np.asarray(response, dtype=float).reshape(-1)
        if y.shape[0] != design.n_runs:
            raise ValueError(
                f"response length ({y.shape[0]}) must match n_runs ({design.n_runs})"
            )
        fit = fit_linear_model(design, y, model=model)
        if fit.dof > 0 and np.isfinite(fit.sigma2) and fit.sigma2 > 0:
            sigma_hat = float(np.sqrt(fit.sigma2))
            sigma = sigma_hat
        for name, p in zip(fit.names, fit.pvalues):
            if name in ("(Intercept)", "Intercept"):
                continue
            if np.isfinite(p) and p < active_p:
                active_terms.append(str(name))
        if fit.dof <= 0:
            caveats.append(
                "Current design is saturated (dof<=0); sigma/power use the "
                "supplied prior sigma, not a residual estimate."
            )
        if active_terms:
            caveats.append(
                f"Active terms at p<{active_p}: {', '.join(active_terms)}. "
                "Consider focusing the next wave / model on these."
            )

    combined = augment_design(
        design, n_add=n_add, model=model, criterion=criterion,
        candidates=candidates, n_candidates=n_candidates, n_starts=n_starts,
        seed=seed,
    )
    n_orig = design.n_runs
    added_mat = combined.matrix.iloc[n_orig:].reset_index(drop=True)
    added = Design(
        matrix=added_mat,
        factors=list(combined.factors or []),
        model=model,
        metadata={
            "kind": "ProposedRuns",
            "n_added": int(n_add),
            "criterion": criterion.strip().upper(),
            "parent_kind": design.metadata.get("kind"),
        },
    )

    comparison = compare_designs(
        design, combined, model=model, effect_size=effect_size, sigma=sigma,
        alpha=alpha, n_region=n_region, seed=seed,
        a_label="current", b_label="augmented",
    )

    rationale = (
        f"Propose {n_add} new run(s) by {criterion.strip().upper()}-optimal "
        f"augmentation of the current {design.n_runs}-run design"
    )
    if sigma_hat is not None:
        rationale += f" (sigma_hat={sigma_hat:.4g} from residual df={fit.dof})"
    rationale += f". {comparison.summary}"

    return NextRunsProposal(
        added=added, combined=combined, comparison=comparison,
        criterion=criterion.strip().upper(), rationale=rationale,
        caveats=caveats, active_terms=active_terms, sigma_hat=sigma_hat,
    )


# ---------------------------------------------------------------------------
# optimize intent: surrogate + acquisition
# ---------------------------------------------------------------------------

def _response_frame(response, n_runs: int, objectives) -> pd.DataFrame:
    """Normalize a 1d/2d/DataFrame response to a DataFrame (n_runs x k)."""
    if response is None:
        raise ValueError("intent='optimize' requires response=... (the measured y)")
    if isinstance(response, pd.DataFrame):
        frame = response.reset_index(drop=True)
    else:
        arr = np.asarray(response, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = list(objectives) if objectives else (
            [f"y{i + 1}" for i in range(arr.shape[1])] if arr.shape[1] > 1 else ["y"]
        )
        frame = pd.DataFrame(arr, columns=names)
    if len(frame) != n_runs:
        raise ValueError(
            f"response has {len(frame)} rows; design has {n_runs}"
        )
    if objectives:
        frame = frame.loc[:, list(objectives)]
    return frame


def _optimize_candidates(design: Design, cols: list, factors: list,
                         n: int, seed: Optional[int],
                         model: Optional[Model] = None) -> pd.DataFrame:
    """Continuous candidate cover for optimization (interior points allowed).

    Unlike the learn-path grid, continuous factors are sampled *continuously*
    across their range so the acquisition can reach an interior optimum.
    Discrete / categorical factors are sampled from their levels, and mixture
    designs are sampled uniformly *on the simplex* (so candidates are feasible).
    """
    rng = np.random.default_rng(seed)
    from ...domain.region import region_from_design, SimplexRegion  # noqa: PLC0415
    try:
        region = region_from_design(design, model)
    except (ValueError, KeyError):
        region = None
    if isinstance(region, SimplexRegion):
        pts = region.sample(n, rng)
        for c in cols:
            if c not in pts.columns:
                pts[c] = 0.0
        return pts.loc[:, cols].reset_index(drop=True)
    by_name = {getattr(f, "name", None): f for f in factors}
    data: dict = {}
    for c in cols:
        f = by_name.get(c)
        col = design.matrix[c]
        if isinstance(f, ContinuousFactor):
            data[c] = rng.uniform(float(f.low), float(f.high), size=n)
        elif isinstance(f, (DiscreteFactor, CategoricalFactor)):
            levels = np.array(list(f.levels), dtype=object)
            data[c] = rng.choice(levels, size=n)
        elif pd.api.types.is_numeric_dtype(col):
            lo, hi = float(col.min()), float(col.max())
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
            data[c] = rng.uniform(lo, hi, size=n)
        else:
            levels = np.array(list(pd.unique(col)), dtype=object)
            data[c] = rng.choice(levels, size=n)
    return pd.DataFrame(data, columns=cols)


def _within_region(cand_frame: pd.DataFrame, design: Design,
                   model: Model, n_add: int):
    """Filter candidates to a constrained region (simplex) when one applies."""
    from ...domain.region import region_from_design, SimplexRegion  # noqa: PLC0415
    try:
        region = region_from_design(design, model)
    except (ValueError, KeyError):
        return cand_frame
    # Only enforce genuinely constrained regions; a plain hypercube on natural
    # units would wrongly reject everything (learn path does not filter either).
    if not isinstance(region, SimplexRegion):
        return cand_frame
    mask = region.contains(cand_frame)
    if int(mask.sum()) >= max(n_add, 1):
        return cand_frame.loc[mask].reset_index(drop=True)
    return cand_frame


def _propose_optimize(design: Design, response, n_add: int, model: Model,
                      candidates: Optional[Design], objectives, goals,
                      goal: str, acquisition: Optional[str], surrogate_kind: str,
                      kappa: float, xi: float, n_candidates: int,
                      n_region: int, seed: Optional[int]) -> NextRunsProposal:
    """Surrogate + acquisition proposal (see :func:`propose_next_runs`)."""
    from ...assessment.surrogate import fit_surrogate  # noqa: PLC0415
    from ..optimize import (  # noqa: PLC0415
        expected_improvement, upper_confidence_bound,
        probability_of_improvement, expected_hypervolume_improvement,
        pareto_front as _pareto_front,
    )

    cols = _factor_columns(design)
    fixed_frame = design.matrix.loc[:, cols].reset_index(drop=True)
    factors = list(design.factors or [])
    y_frame = _response_frame(response, design.n_runs, objectives)
    obj_names = list(y_frame.columns)
    multi = len(obj_names) > 1
    goals = dict(goals or {})
    if not multi and obj_names[0] not in goals:
        goals[obj_names[0]] = goal

    acq_name = (acquisition or ("ehvi" if multi else "ei")).strip().lower()

    # -- candidate pool (continuous cover; respect constrained regions) ------
    if candidates is not None:
        cand_frame = candidates.matrix.loc[
            :, [c for c in cols if c in candidates.matrix.columns]]
        for c in cols:
            if c not in cand_frame.columns:
                raise ValueError(f"candidate set missing factor column {c!r}")
        cand_frame = cand_frame.loc[:, cols].reset_index(drop=True)
    else:
        cand_frame = _optimize_candidates(design, cols, factors,
                                          n_candidates, seed, model)
    cand_frame = _within_region(cand_frame, design, model, n_add)
    if len(cand_frame) < n_add:
        raise ValueError(
            f"only {len(cand_frame)} candidates available for n_add={n_add}; "
            "increase n_candidates or relax constraints"
        )

    def _fit_all(train_frame, y_cols, n_restarts):
        surs = []
        for name in obj_names:
            surs.append(fit_surrogate(
                train_frame, y_cols[name].to_numpy(dtype=float),
                kind=surrogate_kind, model=model, factors=factors,
                n_restarts=n_restarts, seed=seed,
            ))
        return surs

    surrogates = _fit_all(fixed_frame, y_frame, n_restarts=5)
    pool_std_mean = float(np.mean(np.column_stack(
        [s.predict(cand_frame)[1] for s in surrogates])))

    def _predict(surs, frame):
        means = np.column_stack([surs[i].predict(frame)[0] for i in range(len(surs))])
        stds = np.column_stack([surs[i].predict(frame)[1] for i in range(len(surs))])
        return means, stds

    def _best(y_cols):
        out = {}
        for name in obj_names:
            v = y_cols[name].to_numpy(dtype=float)
            out[name] = float(np.max(v) if goals.get(name, "max") == "max"
                              else np.min(v))
        return out

    def _score(means, stds, y_cols):
        if multi:
            return expected_hypervolume_improvement(
                means, stds, y_cols[obj_names].to_numpy(dtype=float),
                goals=goals, columns=obj_names, seed=seed)
        mean, std = means[:, 0], stds[:, 0]
        best = _best(y_cols)[obj_names[0]]
        g = goals.get(obj_names[0], "max")
        if acq_name == "ucb":
            return upper_confidence_bound(mean, std, kappa=kappa, goal=g)
        if acq_name == "pi":
            return probability_of_improvement(mean, std, best, goal=g, xi=xi)
        return expected_improvement(mean, std, best, goal=g, xi=xi)

    # -- batch selection via Kriging Believer (constant-liar) ----------------
    train_frame = fixed_frame.copy()
    train_y = y_frame.copy()
    remaining = list(range(len(cand_frame)))
    selected: list[int] = []
    first_scores = None
    predicted_improvement = None
    sel_std_vals: list[float] = []

    for step in range(n_add):
        means, stds = _predict(surrogates, cand_frame.iloc[remaining])
        scores = _score(means, stds, train_y)
        if first_scores is None:
            # acquisition surface over the full pool for plotting / reporting
            m0, s0 = _predict(surrogates, cand_frame)
            first_scores = _score(m0, s0, train_y)
        local = int(np.argmax(scores))
        idx = remaining[local]
        selected.append(idx)
        sel_std_vals.append(float(np.mean(stds[local])))
        # believe the surrogate at the chosen point (liar = predicted mean)
        liar_mean = means[local]
        if step == 0:
            # acquisition value of the best candidate: EI (>=0) / EHVI (HV gain)
            # / UCB bound, depending on the chosen acquisition.
            predicted_improvement = float(scores[local])
        new_row = cand_frame.iloc[[idx]].reset_index(drop=True)
        train_frame = pd.concat([train_frame, new_row], ignore_index=True)
        liar = {name: liar_mean[j] for j, name in enumerate(obj_names)}
        train_y = pd.concat(
            [train_y, pd.DataFrame([liar], columns=obj_names)], ignore_index=True)
        remaining.pop(local)
        if step < n_add - 1:
            surrogates = _fit_all(train_frame, train_y, n_restarts=0)

    new_rows = cand_frame.iloc[selected].reset_index(drop=True)
    combined_mat = pd.concat([fixed_frame, new_rows], ignore_index=True)
    meta_common = {"parent_kind": design.metadata.get("kind"),
                   "intent": "optimize", "acquisition": acq_name}
    added = Design(
        matrix=new_rows, factors=factors, model=model,
        metadata={"kind": "ProposedRuns", "n_added": int(n_add), **meta_common},
    )
    combined = Design(
        matrix=combined_mat, factors=factors, model=model,
        metadata={"kind": "AugmentedDesign", "n_original": int(design.n_runs),
                  "n_added": int(n_add), **meta_common},
    )

    comparison = compare_designs(
        design, combined, model=model, n_region=n_region, seed=seed,
        a_label="current", b_label="optimize",
    )

    best_now = _best(y_frame)
    best_report = best_now if multi else best_now[obj_names[0]]
    front = None
    if multi:
        pf = _pareto_front(y_frame[obj_names].to_numpy(dtype=float),
                           goals=goals, columns=obj_names)
        front = [dict(zip(obj_names, row)) for row in pf.tolist()]

    sel_std = float(np.mean(sel_std_vals)) if sel_std_vals else float("nan")
    ratio = sel_std / pool_std_mean if pool_std_mean > 0 else float("nan")
    if np.isfinite(ratio):
        mode = ("exploring" if ratio > 1.15 else
                "exploiting" if ratio < 0.85 else "balanced")
    else:
        mode = "unknown"
    explore_exploit = {
        "selected_std_mean": sel_std,
        "pool_std_mean": pool_std_mean,
        "ratio": ratio,
        "mode": mode,
    }

    surrogate_obj = surrogates[0] if not multi else surrogates
    sur_kind = type(surrogates[0]).__name__
    if multi:
        rationale = (
            f"Propose {n_add} run(s) by {acq_name.upper()} on a {sur_kind} "
            f"over {len(obj_names)} objectives {obj_names}; best-so-far="
            f"{best_report}. {mode.capitalize()} the region."
        )
    else:
        rationale = (
            f"Propose {n_add} run(s) by {acq_name.upper()} on a {sur_kind} "
            f"(goal={goals[obj_names[0]]}); best-so-far={best_report:.4g}, "
            f"top acquisition score={predicted_improvement:.4g}. "
            f"{mode.capitalize()} the region."
        )

    caveats = [
        "Optimize intent conditions on the surrogate; calibration (see "
        "surrogate.calibration / parity plot) audits whether sigma(x) is trustworthy.",
        "Acquisition assumes the response is to be optimized, not just learned; "
        "use intent='learn' to sharpen the model instead.",
    ]
    if sur_kind == "OLSSurrogate":
        caveats.append(
            "OLS surrogate in use (scikit-learn not installed): sigma(x) is the "
            "linear prediction SE, not a GP posterior. Install doekit[bo] for a GP.")

    return NextRunsProposal(
        added=added, combined=combined, comparison=comparison,
        criterion="", rationale=rationale, caveats=caveats,
        intent="optimize", acquisition=acq_name, best_so_far=best_report,
        predicted_improvement=predicted_improvement, pareto_front=front,
        explore_exploit=explore_exploit, surrogate=surrogate_obj,
        acquisition_values=first_scores,
    )
