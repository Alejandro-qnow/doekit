"""Multi-objective ranking of candidate designs (testable policy)."""

from __future__ import annotations

import numpy as np


def rank_candidates(rows, priorities: dict, budget, caveats: list) -> dict:
    """Pick the winning candidate row via weighted geometric mean of objectives.

    Scores each feasible row on run economy, D-efficiency (precision), and
    G-efficiency (prediction), then returns the row with the highest weighted
    geometric mean. Mutates feasible rows with ``_score`` and may prepend a
    caveat when nothing fits the budget.

    Formulas
    --------
    For each feasible row ``r`` with minimum feasible run count ``n_min``:

    - ``s_runs = n_min / r["runs"]``
    - ``s_prec = max(eps, r["D_eff"] / 100)``
    - ``s_pred = max(eps, r["G_eff"] / 100)``
    - ``score = s_runs^w0 * s_prec^w1 * s_pred^w2``

    Weights ``w`` are normalized from ``priorities["runs"]``,
    ``priorities["precision"]``, and ``priorities["prediction"]``.

    Parameters
    ----------
    rows : list of dict
        Candidate rows from the advisor shortlist. Each must include ``"runs"``,
        ``"D_eff"``, ``"G_eff"``, and ``"_feasible"`` (bool).
    priorities : dict
        Relative weights with keys ``"runs"``, ``"precision"``, ``"prediction"``.
    budget : int or None
        Run budget; used only in the fallback caveat message.
    caveats : list of str
        Mutable list; a budget warning is prepended when no row is feasible.

    Returns
    -------
    dict
        The winning row (same structure as an element of ``rows``).

    Examples
    --------
    >>> from doekit.orchestration.advise.rank import rank_candidates
    >>> rows = [
    ...     {"method": "A", "runs": 8, "D_eff": 90, "G_eff": 85, "_feasible": True},
    ...     {"method": "B", "runs": 12, "D_eff": 95, "G_eff": 90, "_feasible": True},
    ... ]
    >>> winner = rank_candidates(rows, {"runs": 1, "precision": 1, "prediction": 1}, 20, [])
    >>> winner["method"] in ("A", "B")
    True
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
