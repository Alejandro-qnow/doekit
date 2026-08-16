"""MCP adapter: expose doekit's core workflow as agent tools.

Turns the transparent doekit pipeline — recommend → evaluate → propose
(learn/optimize) → interpret → decide — into MCP tools an autonomous agent can
call. Every tool returns plain JSON-safe dicts built from doekit's own results
(``to_dict`` / ``interpret`` / ``decide``); nothing is invented.

Requires ``pip install "doekit[mcp]"`` (fastmcp). The tool *logic* lives in plain
functions (``tool_*``) that need no fastmcp, so it is unit-testable; only
:func:`build_server` imports fastmcp to register them.

    from doekit.adapters.mcp import build_server
    build_server().run()          # or: python -m doekit.adapters.mcp
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import doekit as ed


def _require_fastmcp():
    """Import fastmcp or raise a helpful ImportError."""
    try:
        from fastmcp import FastMCP  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fastmcp is required for the MCP server. Install with "
            "'pip install \"doekit[mcp]\"'."
        ) from exc
    return FastMCP


def _factor_dict(factors: dict) -> dict:
    """Normalize ``{name: [low, high]}`` / ``{name: (low, high)}`` to tuples."""
    out = {}
    for name, bounds in factors.items():
        out[name] = tuple(bounds[:2])
    return out


def _model_for(names, model_order: str):
    if model_order == "quadratic":
        return ed.Model.full_quadratic(list(names))
    if model_order == "interactions":
        return ed.Model.parse("0 ~ " + " + ".join(
            list(names) + [f"{a}:{b}" for i, a in enumerate(names)
                           for b in list(names)[i + 1:]]))
    return ed.Model.main_effects(list(names))


# ---------------------------------------------------------------------------
# tool logic (plain, fastmcp-free — unit-testable)
# ---------------------------------------------------------------------------

def tool_recommend(goal: str, factors: dict, budget: int,
                   model_order: str = "quadratic") -> dict:
    """Recommend a design and return its interpretation."""
    rec = ed.recommend_design(goal=goal, factors=_factor_dict(factors),
                              budget=budget, model_order=model_order)
    view = ed.interpret(rec)
    return {
        "method": rec.method,
        "n_runs": rec.design.n_runs,
        "factor_names": rec.design.factor_names,
        "rationale": rec.rationale,
        "caveats": list(rec.caveats),
        "interpretation": view.to_dict(),
        "context_addition": view.for_llm(),
    }


def tool_evaluate(design_type: str, factors: dict,
                  model_order: str = "quadratic") -> dict:
    """Build a named design, evaluate it, and return its interpretation."""
    fd = _factor_dict(factors)
    builders = {"central_composite": ed.central_composite,
                "box_behnken": ed.box_behnken}
    if design_type not in builders:
        raise ValueError("design_type must be 'central_composite' or 'box_behnken'")
    design = builders[design_type](fd)
    model = _model_for(design.factor_names, model_order)
    ev = ed.evaluate(design, model=model)
    view = ed.interpret(ev)
    return {
        "design_type": design_type,
        "n_runs": design.n_runs,
        "efficiencies": ev.efficiencies,
        "interpretation": view.to_dict(),
        "context_addition": view.for_llm(),
    }


def tool_propose_and_decide(design_type: str, factors: dict, response: list,
                            model_order: str = "quadratic", n_add: int = 4,
                            intent: str = "learn", acquisition: Optional[str] = None,
                            budget: Optional[int] = None, seed: int = 0) -> dict:
    """Propose the next wave (learn/optimize), interpret and decide the next action."""
    fd = _factor_dict(factors)
    builders = {"central_composite": ed.central_composite,
                "box_behnken": ed.box_behnken}
    if design_type not in builders:
        raise ValueError("design_type must be 'central_composite' or 'box_behnken'")
    design = builders[design_type](fd)
    model = _model_for(design.factor_names, model_order)
    facs = [ed.ContinuousFactor(n, *fd[n]) for n in design.factor_names if n in fd]
    if facs:
        design = design.replace(factors=facs, model=model)
    y = np.asarray(response, dtype=float)
    if y.shape[0] != design.n_runs:
        raise ValueError(f"response length ({y.shape[0]}) must match "
                         f"n_runs ({design.n_runs})")
    kwargs = {"seed": seed}
    if acquisition is not None:
        kwargs["acquisition"] = acquisition
    if intent == "optimize":
        kwargs["surrogate"] = "ols"
    proposal = ed.propose_next_runs(design, response=y, n_add=n_add, model=model,
                                    intent=intent, **kwargs)
    ctx = ed.context_from_proposal(
        proposal, budget_total=int(budget or 0), budget_spent=design.n_runs)
    decision = ed.decide_next_action(ctx)
    view = ed.interpret(proposal)
    return {
        "intent": proposal.intent,
        "n_added": proposal.added.n_runs,
        "proposed_runs": proposal.added.matrix.to_dict("records"),
        "interpretation": view.to_dict(),
        "decision": decision.to_dict(),
        "context_addition": (view.for_llm() + "\n\n" + decision.for_llm()),
    }


TOOLS = (tool_recommend, tool_evaluate, tool_propose_and_decide)


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def build_server(name: str = "doekit"):
    """Create a FastMCP server exposing the doekit tools (requires ``doekit[mcp]``)."""
    FastMCP = _require_fastmcp()
    server = FastMCP(name)
    for fn in TOOLS:
        server.tool()(fn)
    return server


def main() -> None:  # pragma: no cover - entrypoint
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
