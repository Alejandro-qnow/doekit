"""Experimental-design advisor: recommends the best method for a case.

``recommend_design`` is a **transparent advisor** (rules + evaluation), not a
"magic AutoML". Its logic:

1. **Rules** narrow the space to a *shortlist* of plausible methods according
   to the goal, factor types (incl. mixture / hard-to-change) and model order.
2. **Evaluation** ranks that shortlist with ``doekit`` metrics (D/A/G-efficiency,
   prediction variance, runs) according to the user's **priorities**.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...domain.factors import as_factors, CategoricalFactor, MixtureFactor
from ...domain.model import Model, Main, Interaction, Power
from ...domain.design import Design
from ...domain.constraints import Constraints, coerce_constraints
from ...generation.catalog.factorial import full_factorial
from ...generation.catalog.screening import plackett_burman
from ...generation.catalog.response_surface import box_behnken, central_composite
from ...generation.catalog.definitive import definitive_screening
from ...generation.catalog.random_design import random_design
from ...generation.catalog.mixture import simplex_lattice, simplex_centroid
from ...generation.catalog.split_plot import split_plot_design
from ...generation.search.optimal import optimal_design
from ...assessment.evaluation import efficiencies as _efficiencies
from ...shared.serialize import jsonify as _jsonify
from ...shared.errors import InapplicableDesign
from .rank import rank_candidates


_DEFAULT_PRIORITIES = {"runs": 1.0, "precision": 1.0, "prediction": 1.0}


def _model_for(order: str, names) -> Model:
    if order == "linear":
        return Model.main_effects(names)
    if order == "interactions":
        terms = [Main(n) for n in names] + [Interaction((a, b)) for a, b in combinations(names, 2)]
        return Model.from_terms(terms, intercept=True)
    if order == "quadratic":
        return Model.full_quadratic(names)
    raise ValueError(f"unknown model_order: {order!r} (linear/interactions/quadratic)")


def _levels_dict(facs, nlev):
    from ...domain.factors import ContinuousFactor, DiscreteFactor
    lv = {}
    for f in facs:
        if isinstance(f, ContinuousFactor):
            lv[f.name] = ([f.low, f.high] if nlev == 2
                          else [f.low, (f.low + f.high) / 2.0, f.high])
        elif isinstance(f, DiscreteFactor):
            lv[f.name] = list(f.levels)
        elif isinstance(f, MixtureFactor):
            lv[f.name] = [f.lower, (f.lower + f.upper) / 2.0, f.upper]
        else:
            lv[f.name] = list(f.levels)
    return lv


def _is_mixture_case(facs, constraints: Constraints) -> bool:
    return bool(constraints.mixture or all(isinstance(f, MixtureFactor) for f in facs))


def _candidate_designs(goal, factors, facs, names, k, constraints: Constraints,
                       model, seed, hard_to_change: Sequence[str]):
    """Rule-based shortlist of plausible designs."""
    from ...domain.factors import ContinuousFactor
    cands: list[tuple[str, Design]] = []
    has_cat = any(isinstance(f, CategoricalFactor) for f in facs)
    all_continuous = all(isinstance(f, ContinuousFactor) for f in facs)

    def _try(label, builder):
        try:
            cands.append((label, builder()))
        except (ValueError, TypeError, InapplicableDesign, np.linalg.LinAlgError):
            pass

    # --- mixture branch ---
    if _is_mixture_case(facs, constraints):
        mix_facs = [f if isinstance(f, MixtureFactor) else MixtureFactor(f.name)
                    for f in facs]
        order_hint = "quadratic" if goal == "optimization" else "linear"
        _try("Simplex lattice", lambda: simplex_lattice(
            mix_facs, degree=2 if order_hint == "quadratic" else 1))
        _try("Simplex centroid", lambda: simplex_centroid(mix_facs))
        return cands

    # --- split-plot branch ---
    if constraints.wants_split_plot:
        htc = list(hard_to_change or constraints.hard_to_change)
        if not htc:
            # default: first factor is hard-to-change
            htc = [names[0]] if names else []
        wp = [f for f in facs if f.name in htc]
        sp = [f for f in facs if f.name not in htc]
        if not wp or not sp:
            # cannot split — fall through with a caveat via empty + D-opt later
            _try("D-optimal", lambda: _build_optimal(facs, model, seed))
            return cands
        _try("Split-plot", lambda: split_plot_design(wp, sp, whole_plot_reps=2, seed=seed))
        # also offer a pooled D-optimal as alternative (ignores HTC structure)
        _try("D-optimal", lambda: _build_optimal(facs, model, seed))
        return cands

    if constraints.irregular:
        _try("D-optimal", lambda: _build_optimal(facs, model, seed))
        return cands

    if has_cat:
        _try("Full factorial", lambda: full_factorial(_levels_dict(facs, 2)))
        _try("D-optimal", lambda: _build_optimal(facs, model, seed))
        return cands

    if goal == "screening":
        _try("Plackett-Burman", lambda: plackett_burman(k, names=list(names)))
        if k >= 2:
            _try("Definitive Screening", lambda: definitive_screening(factors))
        if k <= 5 and all_continuous:
            _try("Full factorial", lambda: full_factorial(_levels_dict(facs, 2)))
    elif goal == "optimization":
        if k >= 3:
            _try("Box-Behnken", lambda: box_behnken(factors))
        if k >= 2:
            _try("Central Composite", lambda: central_composite(factors))
            _try("Definitive Screening", lambda: definitive_screening(factors))
        _try("D-optimal", lambda: _build_optimal(facs, model, seed))
    else:
        raise ValueError(f"unknown goal: {goal!r} (use 'screening' or 'optimization')")
    return cands


def _build_optimal(facs, model, seed) -> Design:
    lv = _levels_dict(facs, 3)
    ncombos = int(np.prod([len(v) for v in lv.values()]))
    if ncombos <= 1000:
        cand = full_factorial(lv)
    else:
        cand = random_design(facs, n=400, seed=seed)
    cand = cand.replace(factors=list(facs), model=model)
    p = len(model.column_names(cand.matrix))
    n_runs = min(len(cand.matrix), max(p + 1, p + 3))
    return optimal_design(cand, n_runs=n_runs, model=model, n_starts=6, seed=seed)


@dataclass
class Recommendation:
    """Result of :func:`recommend_design`."""

    method: str
    design: Design
    model: Model
    rationale: str
    table: pd.DataFrame = field(repr=False)
    caveats: list = field(default_factory=list)
    scenario: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Recommendation: {self.method}", "-" * 46, self.rationale, "",
                 "Evaluated alternatives:", self.table.to_string(index=False)]
        if self.caveats:
            lines += ["", "Caveats:"] + [f"  - {c}" for c in self.caveats]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return _jsonify({
            "schema": "doekit.Recommendation/1",
            "method": self.method,
            "rationale": self.rationale,
            "caveats": list(self.caveats),
            "scenario": dict(self.scenario),
            "alternatives": self.table.to_dict("records"),
            "design": self.design.to_dict(),
            "model": self.model.to_dict() if self.model is not None else None,
        })

    def __repr__(self) -> str:
        return self.summary()


def recommend_design(
    goal: str,
    factors,
    budget: Optional[int] = None,
    model_order: Optional[str] = None,
    priorities: Optional[dict] = None,
    constrained: bool = False,
    constraints: Union[Constraints, dict, None] = None,
    mixture: bool = False,
    hard_to_change: Optional[Sequence[str]] = None,
    effect_size=1.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    n_region: int = 4000,
) -> Recommendation:
    """Recommend the best experimental-design method for a given case.

    Parameters
    ----------
    goal : {"screening", "optimization"}
    factors : int or dict or sequence of Factor
    budget : int, optional
    model_order : {"linear", "interactions", "quadratic"}, optional
    priorities : dict, optional
    constrained : bool, default False
        **Deprecated.** Use ``constraints=Constraints(irregular=True)``.
    constraints : Constraints or dict, optional
        Native constraints (mixture, hard_to_change, irregular, run_cost, …).
    mixture : bool, default False
        Shortcut for mixture / simplex shortlist.
    hard_to_change : sequence of str, optional
        Shortcut for split-plot whole-plot factor names.
    effect_size, sigma : float
    seed, n_region
    """
    if constrained and constraints is None:
        warnings.warn(
            "constrained=True is deprecated; pass "
            "constraints=Constraints(irregular=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    cons = coerce_constraints(constraints, constrained=constrained)
    if mixture:
        cons = Constraints(
            mixture=True,
            hard_to_change=cons.hard_to_change,
            split_plot=cons.split_plot,
            run_cost=cons.run_cost,
            exclude=cons.exclude,
            irregular=cons.irregular,
        )
    if hard_to_change:
        cons = Constraints(
            mixture=cons.mixture,
            hard_to_change=tuple(hard_to_change),
            split_plot=True,
            run_cost=cons.run_cost,
            exclude=cons.exclude,
            irregular=cons.irregular,
        )

    facs = as_factors(factors)
    # Infer mixture from factor types
    if all(isinstance(f, MixtureFactor) for f in facs) and facs:
        cons = Constraints(
            mixture=True,
            hard_to_change=cons.hard_to_change,
            split_plot=cons.split_plot,
            run_cost=cons.run_cost,
            exclude=cons.exclude,
            irregular=cons.irregular,
        )

    is_mix = _is_mixture_case(facs, cons)
    order = model_order or ("linear" if goal == "screening" else "quadratic")
    names = [f.name for f in facs]
    k = len(names)
    prio = {**_DEFAULT_PRIORITIES, **(priorities or {})}
    caveats = _base_caveats(order, facs, effect_size, cons, is_mix)

    if is_mix:
        model = (Model.scheffe_linear(names) if order == "linear"
                 else Model.scheffe_quadratic(names))
    else:
        model = _model_for(order, names)

    htc = list(cons.hard_to_change or hard_to_change or [])
    cands = _candidate_designs(goal, factors, facs, names, k, cons, model, seed, htc)

    rows = []
    for label, d in cands:
        try:
            # For split-plot, evaluate on treatment columns only
            eval_model = model
            if d.metadata.get("kind") == "SplitPlot":
                treat = d.metadata.get("whole_plot", []) + d.metadata.get("subplot", [])
                eval_model = Model.main_effects(treat)
            eff = _efficiencies(d, model=eval_model, n_region=n_region, seed=seed)
            evaluable = not eff["rank_deficient"]
        except (ValueError, TypeError, np.linalg.LinAlgError):
            eff, evaluable = None, False
        fits_budget = budget is None or d.n_runs <= budget
        feasible = evaluable and fits_budget
        rows.append({
            "method": label, "runs": d.n_runs,
            "D_eff": round(eff["D_efficiency"], 1) if evaluable else None,
            "G_eff": round(eff["G_efficiency"], 1) if evaluable else None,
            "SPV_mean": round(eff["spv_mean"], 2) if evaluable else None,
            "in_budget": fits_budget, "supports_model": evaluable,
            "_feasible": feasible, "_eff": eff, "_design": d,
        })

    if not rows:
        raise InapplicableDesign(
            "no catalog design applies to this factor / constraint combination"
        )

    winner = rank_candidates(rows, prio, budget, caveats)
    table = pd.DataFrame([{k2: v for k2, v in r.items() if not k2.startswith("_")} for r in rows])
    rationale = _rationale(goal, k, order, budget, winner, prio, is_mix, cons)
    # Attach winning model used for evaluation when split-plot
    win_design = winner["_design"]
    win_model = win_design.model or model
    return Recommendation(
        method=winner["method"], design=win_design, model=win_model,
        rationale=rationale, table=table, caveats=caveats,
        scenario={
            "goal": goal, "n_factors": k, "budget": budget, "model_order": order,
            "mixture": is_mix, "split_plot": cons.wants_split_plot,
            "constraints": cons.to_dict(),
        },
    )


def _base_caveats(order, facs, effect_size, cons: Constraints, is_mix: bool) -> list:
    cav = [
        (f"Recommendation is conditional on model_order='{order}' "
         f"(and effect_size={effect_size} for power)."),
        ("\"Best\" is a multi-objective trade-off (runs vs precision vs prediction): "
         "adjust 'priorities' for your case."),
        ("After the first experimental wave, use propose_next_runs / augment_design "
         "to choose the next batch (sequential DoE) instead of restarting from scratch."),
    ]
    if is_mix:
        cav.append(
            "Mixture case: shortlist uses simplex lattice / centroid with Scheffé "
            "models; evaluate samples the simplex (not a hypercube)."
        )
    elif cons.wants_split_plot:
        cav.append(
            "Split-plot case: analyse with fit_mixed_model(groups='whole_plot_id') "
            "(or blocks= for fixed plots). D-optimal alternative ignores HTC structure."
        )
    if any(isinstance(f, CategoricalFactor) for f in facs) and not is_mix:
        cav.append(
            "Categorical factors present: RSM designs (Box-Behnken/CCD) assume "
            "continuous factors; prefer full factorial or D-optimal."
        )
    return cav


def _rationale(goal, k, order, budget, winner, prio, is_mix, cons) -> str:
    if is_mix:
        obj = "model a mixture response (Scheffé) on the simplex"
    elif cons.wants_split_plot:
        obj = "respect hard-to-change factors via a split-plot structure"
    elif goal == "screening":
        obj = "identify influential factors"
    else:
        obj = "model the response surface and locate the optimum"
    pres = f" with a budget of {budget} runs" if budget else ""
    met = winner["method"]
    if winner.get("D_eff") is not None:
        detalle = (f" ({winner['runs']} runs, D-efficiency {winner['D_eff']}%, "
                   f"G-efficiency {winner['G_eff']}%)")
    else:
        detalle = f" ({winner['runs']} runs)"
    return (f"To {obj} with {k} factors{pres} and a '{order}' model, the best "
            f"compromise under your priorities is {met}{detalle}.")
