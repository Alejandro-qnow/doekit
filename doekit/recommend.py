"""Experimental-design advisor: recommends the best method for a case.

``recommend_design`` is a **transparent advisor** (rules + evaluation), not a
"magic AutoML". Its logic:

1. **Rules** (based on the classical methodology: Montgomery; NIST/SEMATECH
   handbook; JMP guides) narrow the space to a *shortlist* of plausible methods
   according to the goal, the number/type of factors and the model order.
2. **Evaluation** ranks that shortlist with ``doekit``'s metrics (D/A/G-efficiency,
   prediction variance, number of runs) according to the user's **priorities**.

IMPORTANT CAVEATS (by design, not defects):

- **"The best" is a multi-objective trade-off**, not an absolute: it depends on
  whether you prioritize *few runs*, *coefficient precision* or *prediction over
  the region*. That is why the result exposes the table of alternatives and their
  metrics, and accepts ``priorities``.
- The recommendation is **conditional** on the assumptions you give it
  (``model_order`` and, for the power, ``effect_size``/``sigma``). They are listed
  in ``Recommendation.caveats``.
- **Coverage limited to the ``doekit`` catalog**: it does not yet cover **mixture
  designs** (simplex, factors summing to a constant) nor **split-plot** (hard-to-
  change factors). If your case is one of those, the advisor flags it in
  ``caveats`` rather than staying silent. For already-collected split-plot data,
  analyse with :func:`doekit.analysis.fit_mixed_model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from .factors import as_factors, CategoricalFactor
from .model import Model, Main, Interaction, Power
from .designs.base import Design
from .designs.factorial import full_factorial
from .designs.screening import plackett_burman
from .designs.response_surface import box_behnken, central_composite
from .designs.definitive import definitive_screening
from .designs.random_design import random_design
from .designs.optimal import optimal_design
from .evaluate import efficiencies as _efficiencies


_DEFAULT_PRIORITIES = {"runs": 1.0, "precision": 1.0, "prediction": 1.0}


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _model_for(order: str, names) -> Model:
    """Build the model implied by ``order`` (linear/interactions/quadratic)."""
    if order == "linear":
        return Model.main_effects(names)
    if order == "interactions":
        terms = [Main(n) for n in names] + [Interaction((a, b)) for a, b in combinations(names, 2)]
        return Model.from_terms(terms, intercept=True)
    if order == "quadratic":
        return Model.full_quadratic(names)
    raise ValueError(f"unknown model_order: {order!r} (linear/interactions/quadratic)")


def _levels_dict(facs, nlev):
    """Per-factor levels to build grids/factorials (nlev = 2 or 3)."""
    from .factors import ContinuousFactor, DiscreteFactor
    lv = {}
    for f in facs:
        if isinstance(f, ContinuousFactor):
            lv[f.name] = ([f.low, f.high] if nlev == 2
                          else [f.low, (f.low + f.high) / 2.0, f.high])
        elif isinstance(f, DiscreteFactor):
            lv[f.name] = list(f.levels)
        else:  # categorical
            lv[f.name] = list(f.levels)
    return lv


def _candidate_designs(goal, factors, facs, names, k, constrained, model, seed):
    """Rule-based shortlist of plausible designs. Returns a list of (label, Design)."""
    from .factors import ContinuousFactor
    cands: list[tuple[str, Design]] = []
    has_cat = any(isinstance(f, CategoricalFactor) for f in facs)
    all_continuous = all(isinstance(f, ContinuousFactor) for f in facs)

    def _try(label, builder):
        try:
            cands.append((label, builder()))
        except Exception:
            pass  # the design does not apply to this factor type -> skip it

    if constrained:
        _try("D-optimal", lambda: _build_optimal(facs, model, seed))
        return cands

    if has_cat:
        # RSM/PB/DSD assume continuous/2-level factors; with categoricals only
        # the full factorial and D-optimal (dummy) designs are valid.
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
    """D-optimal design over a 3-level grid (well-behaved for RSM)."""
    lv = _levels_dict(facs, 3)
    ncombos = int(np.prod([len(v) for v in lv.values()]))
    if ncombos <= 1000:
        cand = full_factorial(lv)
    else:
        cand = random_design(facs, n=400, seed=seed)
    cand.factors = list(facs)   # factor metadata -> evaluation codes correctly
    cand.model = model
    p = len(model.column_names(cand.matrix))
    n_runs = min(len(cand.matrix), max(p + 1, p + 3))
    return optimal_design(cand, n_runs=n_runs, model=model, n_starts=6, seed=seed)


@dataclass
class Recommendation:
    """Result of :func:`recommend_design`: chosen method + alternatives + rationale.

    Attributes
    ----------
    method : str
        Label of the recommended design method.
    design : Design
        The built design for the recommended method.
    model : Model
        The model all candidates were evaluated under.
    rationale : str
        Human-readable justification of the choice.
    table : pandas.DataFrame
        The evaluated alternatives with their metrics.
    caveats : list of str
        Assumptions and catalog-coverage caveats.
    scenario : dict
        Echo of the resolved scenario (goal, n_factors, budget, model_order).
    """

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
        """Serialize to a JSON-safe dict (``schema: doekit.Recommendation/1``)."""
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


def recommend_design(goal: str, factors, budget: Optional[int] = None,
                     model_order: Optional[str] = None,
                     priorities: Optional[dict] = None, constrained: bool = False,
                     effect_size=1.0, sigma: float = 1.0, seed: Optional[int] = None,
                     n_region: int = 4000) -> Recommendation:
    """Recommend the best experimental-design method for a given case.

    Parameters
    ----------
    goal : {"screening", "optimization"}
        ``"screening"`` (which factors matter?) or ``"optimization"`` (response
        surface: where is the optimum?).
    factors : int or dict or sequence of Factor
        An ``int`` (number of factors), a ``dict`` ``{name: (low, high)}`` or a
        list of :class:`~doekit.factors.Factor`.
    budget : int, optional
        Maximum affordable number of runs. Designs exceeding it are shown but do
        not win; if none fit, the smallest is recommended with a caveat.
    model_order : {"linear", "interactions", "quadratic"}, optional
        Model order; inferred from the goal if ``None`` (screening->linear,
        optimization->quadratic). **The recommendation is conditional on this
        assumption.**
    priorities : dict, optional
        Weights ``{"runs", "precision", "prediction"}`` for the multi-objective
        trade-off. Balanced by default. *"The best" depends on these weights.*
    constrained : bool, default False
        ``True`` if the region is irregular/constrained -> forces an optimal design.
    effect_size, sigma : float, default 1.0
        Assumptions for the power analysis (currently informative).
    seed : int, optional
        Seed controlling the evaluation.
    n_region : int, default 4000
        Region sample size for the efficiency evaluation.

    Returns
    -------
    Recommendation
        The chosen method, the built ``Design``, the rationale, the table of
        alternatives with their metrics and the **caveats**.

    Notes
    -----
    Coverage limited to the ``doekit`` catalog: it does **not** cover mixture nor
    split-plot *design generation*; if the case requires them, this is flagged in
    ``caveats``. Collected split-plot / blocked data can still be analysed with
    :func:`~doekit.analysis.fit_mixed_model` / ``blocks=`` in
    :func:`~doekit.analysis.fit_linear_model`.
    """
    order = model_order or ("linear" if goal == "screening" else "quadratic")
    facs = as_factors(factors)
    names = [f.name for f in facs]
    k = len(names)
    prio = {**_DEFAULT_PRIORITIES, **(priorities or {})}
    caveats = _base_caveats(order, facs, effect_size)

    model = _model_for(order, names)
    cands = _candidate_designs(goal, factors, facs, names, k, constrained, model, seed)

    # --- evaluate every candidate under the SAME model (fair comparison) ---
    rows = []
    for label, d in cands:
        try:
            eff = _efficiencies(d, model=model, n_region=n_region, seed=seed)
            evaluable = not eff["rank_deficient"]
        except Exception:
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

    winner = _rank(rows, prio, budget, caveats)
    table = pd.DataFrame([{k2: v for k2, v in r.items() if not k2.startswith("_")} for r in rows])
    rationale = _rationale(goal, k, order, budget, winner, prio)
    return Recommendation(
        method=winner["method"], design=winner["_design"], model=model,
        rationale=rationale, table=table, caveats=caveats,
        scenario={"goal": goal, "n_factors": k, "budget": budget, "model_order": order},
    )


def _rank(rows, prio, budget, caveats):
    feas = [r for r in rows if r["_feasible"]]
    if not feas:
        min_runs = min(rows, key=lambda r: r["runs"])
        caveats.insert(0, (
            f"No catalog design fits the budget ({budget}) and supports the model; "
            f"recommending the smallest viable option ({min_runs['method']}, "
            f"{min_runs['runs']} runs). Increase the budget or reduce the model."
        ))
        return min_runs
    min_runs = min(r["runs"] for r in feas)
    w = np.array([prio["runs"], prio["precision"], prio["prediction"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])
    eps = 1e-3
    for r in feas:
        s_runs = min_runs / r["runs"]
        s_prec = max(eps, r["D_eff"] / 100.0)
        s_pred = max(eps, r["G_eff"] / 100.0)
        r["_score"] = float(s_runs ** w[0] * s_prec ** w[1] * s_pred ** w[2])
    return max(feas, key=lambda r: r["_score"])


def _base_caveats(order, facs, effect_size) -> list:
    cav = [
        (f"Recommendation is conditional on model_order='{order}' "
         f"(and effect_size={effect_size} for power)."),
        ("\"Best\" is a multi-objective trade-off (runs vs precision vs prediction): "
         "adjust 'priorities' for your case."),
        ("Outside the current catalog: mixture designs (simplex) and split-plot "
         "design generation (hard-to-change factors). If that is your case, design "
         "those separately; for collected grouped/blocked data use "
         "fit_mixed_model(groups=...) or fit_linear_model(blocks=...)."),
        ("After the first experimental wave, use propose_next_runs / augment_design "
         "to choose the next batch (sequential DoE) instead of restarting from scratch."),
    ]
    if any(isinstance(f, CategoricalFactor) for f in facs):
        cav.append(
            "Categorical factors present: RSM designs (Box-Behnken/CCD) assume "
            "continuous factors; prefer full factorial or D-optimal."
        )
    return cav


def _rationale(goal, k, order, budget, winner, prio) -> str:
    obj = ("identify influential factors" if goal == "screening"
           else "model the response surface and locate the optimum")
    pres = f" with a budget of {budget} runs" if budget else ""
    met = winner["method"]
    if winner.get("D_eff") is not None:
        detalle = (f" ({winner['runs']} runs, D-efficiency {winner['D_eff']}%, "
                   f"G-efficiency {winner['G_eff']}%)")
    else:
        detalle = f" ({winner['runs']} runs)"
    return (f"To {obj} with {k} factors{pres} and a '{order}' model, the best "
            f"compromise under your priorities is {met}{detalle}.")
