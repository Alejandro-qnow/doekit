"""
FastMCP server exposing deterministic doekit workflows for research support.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import doekit as ed
import numpy as np
from fastmcp import FastMCP


mcp = FastMCP("doekit-research")


def _build_factor_dict(factors: Dict[str, List[float]] | Dict[str, tuple]) -> Dict[str, tuple]:
    normalized: Dict[str, tuple] = {}
    for name, bounds in factors.items():
        if isinstance(bounds, tuple):
            normalized[name] = bounds
        else:
            normalized[name] = (bounds[0], bounds[1])
    return normalized


@mcp.tool()
def recommend_design(
    goal: str,
    factors: Dict[str, List[float]],
    budget: int,
    model_order: str = "quadratic",
) -> Dict[str, Any]:
    """Recommend an experimental design from doekit with deterministic summary."""
    factor_dict = _build_factor_dict(factors)
    rec = ed.recommend_design(
        goal=goal,
        factors=factor_dict,
        budget=budget,
        model_order=model_order,
    )
    return {
        "method": rec.method,
        "n_runs": rec.design.n_runs,
        "factor_names": rec.design.factor_names,
        "rationale": rec.rationale,
        "caveats": rec.caveats,
        "scenario": rec.scenario,
    }


@mcp.tool()
def evaluate_design(
    design_type: str,
    factors: Dict[str, List[float]],
    model_order: str = "quadratic",
) -> Dict[str, Any]:
    """Build and evaluate a design for deterministic quality metrics."""
    factor_dict = _build_factor_dict(factors)

    if design_type == "central_composite":
        design = ed.central_composite(factor_dict)
    elif design_type == "box_behnken":
        design = ed.box_behnken(factor_dict)
    else:
        raise ValueError("design_type must be 'central_composite' or 'box_behnken'")

    if model_order == "quadratic":
        model = ed.Model.full_quadratic(design.factor_names)
    else:
        model = ed.Model.main_effects(design.factor_names)

    evaluation = ed.evaluate(design, model=model)
    return {
        "n_runs": design.n_runs,
        "d_efficiency": float(evaluation.d_efficiency) if hasattr(evaluation, "d_efficiency") else None,
        "mean_power": float(evaluation.power.mean()) if hasattr(evaluation, "power") else None,
        "max_vif": float(evaluation.vif.max()) if hasattr(evaluation, "vif") else None,
        "dof": int(evaluation.dof) if hasattr(evaluation, "dof") else None,
    }


@mcp.tool()
def propose_next_wave(
    factors: Dict[str, List[float]],
    model_order: str = "quadratic",
    n_add: int = 2,
    seed: int = 42,
    sigma: float = 1.0,
) -> Dict[str, Any]:
    """Simulate one sequential wave and return proposal/comparison deltas."""
    factor_dict = _build_factor_dict(factors)
    np.random.seed(seed)

    design = ed.central_composite(factor_dict)
    model = (
        ed.Model.full_quadratic(design.factor_names)
        if model_order == "quadratic"
        else ed.Model.main_effects(design.factor_names)
    )

    x = design.matrix.values
    y = np.random.randn(len(x)) * sigma

    proposal = ed.propose_next_runs(design, response=y, n_add=n_add, model=model)
    comparison = ed.compare_designs(design, proposal.combined, model=model)

    return {
        "n_added": proposal.added.n_runs,
        "worth_it": bool(comparison.worth_it),
        "delta": comparison.delta,
        "criterion": proposal.criterion,
        "sigma_hat": float(proposal.sigma_hat) if proposal.sigma_hat is not None else None,
        "rationale": proposal.rationale,
    }


if __name__ == "__main__":
    mcp.run()
