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

    def to_dict(self) -> dict:
        """Serialize (``schema: doekit.NextRunsProposal/1``)."""
        return _jsonify({
            "schema": "doekit.NextRunsProposal/1",
            "criterion": self.criterion,
            "rationale": self.rationale,
            "caveats": list(self.caveats),
            "active_terms": list(self.active_terms),
            "sigma_hat": self.sigma_hat,
            "added": self.added.to_dict(),
            "combined": self.combined.to_dict(),
            "comparison": self.comparison.to_dict(),
        })

    def summary(self) -> str:
        lines = [
            f"Next runs proposal ({self.added.n_runs} new, "
            f"criterion={self.criterion})",
            "-" * 48,
            self.rationale,
            "",
            self.comparison.summary,
            "",
            "Proposed runs:",
            repr(self.added.matrix),
        ]
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
                      active_p: float = 0.05) -> NextRunsProposal:
    """Propose the next batch of runs for a sequential experiment.

    Without ``response``, augments by information (D/I/…). With ``response``,
    estimates residual sigma, flags active terms (p < ``active_p``), and uses
    the empirical sigma for the power side of :func:`compare_designs`.

    Parameters
    ----------
    design : Design
        Current design (runs already done or committed).
    response : array-like, optional
        Measured responses for the current design rows.
    n_add : int, default 4
        Number of new runs to propose. Capped by ``budget - n_runs`` when set.
    model : Model, optional
        Model for augmentation / evaluation.
    criterion : str, default "D"
        Augmentation criterion.
    candidates : Design, optional
        Candidate set for new runs.
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

    Returns
    -------
    NextRunsProposal
    """
    _ = priorities  # API placeholder for multi-objective next-batch ranking
    model = _resolve_model(design, model)
    caveats = [
        "Proposal is conditional on the assumed model and criterion; "
        "re-run after updating the model if the active set changes.",
        "Augmentation is classical (D/I-optimal conditioned on current runs), "
        "not a black-box Bayesian optimizer — see doekit.bo for a thin BO bridge.",
    ]

    if budget is not None:
        remaining = budget - design.n_runs
        if remaining <= 0:
            raise ValueError(
                f"budget={budget} already exhausted (n_runs={design.n_runs})"
            )
        n_add = min(n_add, remaining)

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
