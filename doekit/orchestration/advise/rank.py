"""Multi-objective ranking of candidate designs (testeable policy)."""

from __future__ import annotations

import numpy as np


def rank_candidates(rows, priorities: dict, budget, caveats: list) -> dict:
    """Pick the winning row via weighted geometric mean of runs/precision/prediction.

    Mutates feasible rows with ``_score`` and may prepend a caveat when nothing
    is feasible under the budget.
    """
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
    w = np.array([priorities["runs"], priorities["precision"],
                  priorities["prediction"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
    eps = 1e-3
    for r in feas:
        s_runs = min_runs / r["runs"]
        s_prec = max(eps, r["D_eff"] / 100.0)
        s_pred = max(eps, r["G_eff"] / 100.0)
        r["_score"] = float(s_runs ** w[0] * s_prec ** w[1] * s_pred ** w[2])
    return max(feas, key=lambda r: r["_score"])
