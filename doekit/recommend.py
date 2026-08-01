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
  ``caveats`` rather than staying silent.

Note: the user-facing strings (method labels, table columns, caveats and
rationale) are kept in Spanish here and become language-parameterized in the
reporting layer.
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
        _try("D-optimo", lambda: _build_optimal(facs, model, seed))
        return cands

    if has_cat:
        # RSM/PB/DSD assume continuous/2-level factors; with categoricals only
        # the full factorial and D-optimal (dummy) designs are valid.
        _try("Factorial completo", lambda: full_factorial(_levels_dict(facs, 2)))
        _try("D-optimo", lambda: _build_optimal(facs, model, seed))
        return cands

    if goal == "screening":
        _try("Plackett-Burman", lambda: plackett_burman(k, names=list(names)))
        if k >= 2:
            _try("Definitive Screening", lambda: definitive_screening(factors))
        if k <= 5 and all_continuous:
            _try("Factorial completo", lambda: full_factorial(_levels_dict(facs, 2)))
    elif goal == "optimization":
        if k >= 3:
            _try("Box-Behnken", lambda: box_behnken(factors))
        if k >= 2:
            _try("Central Composite", lambda: central_composite(factors))
            _try("Definitive Screening", lambda: definitive_screening(factors))
        _try("D-optimo", lambda: _build_optimal(facs, model, seed))
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
        lines = [f"Recomendacion: {self.method}", "-" * 46, self.rationale, "",
                 "Alternativas evaluadas:", self.table.to_string(index=False)]
        if self.caveats:
            lines += ["", "Salvedades:"] + [f"  - {c}" for c in self.caveats]
        return "\n".join(lines)

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
    split-plot designs; if the case requires them, this is flagged in ``caveats``.
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
            eff, evaluable = None, False   # e.g. a model with categoricals not alignable
        fits_budget = budget is None or d.n_runs <= budget
        feasible = evaluable and fits_budget
        rows.append({
            "metodo": label, "corridas": d.n_runs,
            "D_eff": round(eff["D_efficiency"], 1) if evaluable else None,
            "G_eff": round(eff["G_efficiency"], 1) if evaluable else None,
            "SPV_medio": round(eff["spv_mean"], 2) if evaluable else None,
            "en_presupuesto": fits_budget, "soporta_modelo": evaluable,
            "_feasible": feasible, "_eff": eff, "_design": d,
        })

    winner = _rank(rows, prio, budget, caveats)
    table = pd.DataFrame([{k2: v for k2, v in r.items() if not k2.startswith("_")} for r in rows])
    rationale = _rationale(goal, k, order, budget, winner, prio)
    return Recommendation(
        method=winner["metodo"], design=winner["_design"], model=model,
        rationale=rationale, table=table, caveats=caveats,
        scenario={"goal": goal, "n_factors": k, "budget": budget, "model_order": order},
    )


def _rank(rows, prio, budget, caveats):
    feas = [r for r in rows if r["_feasible"]]
    if not feas:  # nada cabe en presupuesto / soporta el modelo
        min_runs = min(rows, key=lambda r: r["corridas"])
        caveats.insert(0, f"Ningun diseno del catalogo cabe en el presupuesto ({budget}) y "
                          f"soporta el modelo; se recomienda el menor viable ({min_runs['metodo']}, "
                          f"{min_runs['corridas']} corridas). Aumenta el presupuesto o reduce el modelo.")
        return min_runs
    min_runs = min(r["corridas"] for r in feas)
    w = np.array([prio["runs"], prio["precision"], prio["prediction"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])
    eps = 1e-3
    for r in feas:
        s_runs = min_runs / r["corridas"]                 # fewer runs -> higher
        s_prec = max(eps, r["D_eff"] / 100.0)
        s_pred = max(eps, r["G_eff"] / 100.0)
        # weighted geometric mean: a catastrophic axis (e.g. G~0) SINKS the score,
        # with no other axis compensating. Consistent with "good on ALL objectives".
        r["_score"] = float(s_runs ** w[0] * s_prec ** w[1] * s_pred ** w[2])
    return max(feas, key=lambda r: r["_score"])


def _base_caveats(order, facs, effect_size) -> list:
    cav = [
        f"Recomendacion condicional al supuesto model_order='{order}' "
        f"(y effect_size={effect_size} para la potencia).",
        "\"El mejor\" es un trade-off multiobjetivo (corridas vs precision vs prediccion): "
        "ajusta 'priorities' segun tu caso.",
        "Fuera del catalogo actual: disenos de mezcla (simplex) y split-plot (factores "
        "dificiles de cambiar); si aplica tu caso, considera esos metodos por separado.",
    ]
    if any(isinstance(f, CategoricalFactor) for f in facs):
        cav.append("Hay factores categoricos: los disenos RSM (Box-Behnken/CCD) asumen "
                   "factores continuos; para categoricos prefiere factorial o D-optimo.")
    return cav


def _rationale(goal, k, order, budget, winner, prio) -> str:
    obj = "identificar los factores influyentes" if goal == "screening" else \
          "modelar la superficie de respuesta y localizar el optimo"
    pres = f" con un presupuesto de {budget} corridas" if budget else ""
    met = winner["metodo"]
    if winner.get("D_eff") is not None:
        detalle = (f" ({winner['corridas']} corridas, D-eficiencia {winner['D_eff']}%, "
                   f"G-eficiencia {winner['G_eff']}%)")
    else:
        detalle = f" ({winner['corridas']} corridas)"
    return (f"Para {obj} con {k} factores{pres} y un modelo '{order}', el mejor compromiso "
            f"segun tus prioridades es {met}{detalle}.")
