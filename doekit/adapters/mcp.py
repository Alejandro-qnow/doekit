"""MCP adapter: expose doekit's core workflow as agent tools.

Turns the transparent doekit pipeline — recommend → evaluate → propose
(learn/optimize) → interpret → decide — into MCP tools an autonomous agent can
call. Every tool returns plain JSON-safe dicts built from doekit's own results
(``to_dict`` / ``interpret`` / ``decide``); nothing is invented.

Requires ``pip install "doekit[mcp]"`` (fastmcp). Tool logic lives in plain
``tool_*`` functions (unit-testable without fastmcp); only :func:`build_server`
imports fastmcp to register them.

Notes
-----
Run locally with ``python -m doekit.adapters.mcp`` or
``build_server().run()`` after installing ``doekit[mcp]``.
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


def _surrogate_calibration(surrogate) -> Optional[dict]:
    """Best-effort LOO calibration summary for an optimize surrogate."""
    if surrogate is None:
        return None
    try:
        cal = surrogate.calibration()
    except Exception:  # pragma: no cover - calibration is best-effort
        return None
    return {
        "kind": getattr(surrogate, "kind", type(surrogate).__name__),
        **cal,
    }


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
    """Recommend a design and return JSON-safe interpretation.

    Wraps :func:`doekit.recommend_design` and :func:`doekit.interpret` so an
    MCP client receives plain dicts (no Design objects).

    Parameters
    ----------
    goal : str
        ``"screening"`` or ``"optimization"``.
    factors : dict
        ``{name: [low, high]}`` or ``{name: (low, high)}`` factor bounds.
    budget : int
        Maximum number of runs.
    model_order : str, default ``"quadratic"``
        ``"linear"``, ``"interactions"``, or ``"quadratic"``.

    Returns
    -------
    dict
        Keys ``method``, ``n_runs``, ``factor_names``, ``rationale``,
        ``caveats``, ``interpretation`` (dict), ``context_addition`` (str).

    Notes
    -----
    Registered on the FastMCP server by :func:`build_server`. Unit-testable
    without fastmcp installed.

    Examples
    --------
    >>> from doekit.adapters.mcp import tool_recommend
    >>> out = tool_recommend("screening", {"A": [0, 1], "B": [0, 1]}, budget=8)
    >>> "method" in out and "interpretation" in out
    True
    """
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
    """Build a named RSM design, evaluate it, and return interpretation.

    Parameters
    ----------
    design_type : str
        ``"central_composite"`` or ``"box_behnken"``.
    factors : dict
        ``{name: [low, high]}`` factor bounds.
    model_order : str, default ``"quadratic"``
        Model order for evaluation (see :func:`tool_recommend`).

    Returns
    -------
    dict
        Keys ``design_type``, ``n_runs``, ``efficiencies``, ``interpretation``,
        ``context_addition``.

    Raises
    ------
    ValueError
        When ``design_type`` is not supported.
    """
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
                            budget: Optional[int] = None, seed: int = 0,
                            history: Optional[list] = None) -> dict:
    """Propose next runs (learn/optimize), interpret, and decide the next action.

    Fits on the current design plus ``response``, proposes ``n_add`` follow-up
    runs, and returns a decision with step diagnostics. Optional ``history``
    enables the convergence stop gate.

    Parameters
    ----------
    design_type : str
        ``"central_composite"`` or ``"box_behnken"``.
    factors : dict
        ``{name: [low, high]}`` factor bounds.
    response : list
        Response values aligned with design run order (length = ``n_runs``).
    model_order : str, default ``"quadratic"``
        Model order for proposal and evaluation.
    n_add : int, default 4
        Number of follow-up runs to propose.
    intent : str, default ``"learn"``
        ``"learn"`` (augment for model quality) or ``"optimize"`` (BO-style).
    acquisition : str, optional
        Acquisition function for ``intent="optimize"``.
    budget : int, optional
        Total run budget for decision context (0 when omitted).
    seed : int, default 0
        Random seed for proposal stochasticity.
    history : list, optional
        Per-iteration metric history for :func:`~doekit.check_convergence`
        (``best_so_far`` for optimize, ``delta_D_efficiency`` for learn).

    Returns
    -------
    dict
        Keys ``intent``, ``n_added``, ``proposed_runs`` (list of row dicts),
        ``interpretation``, ``decision``, ``diagnostics``, ``context_addition``;
        optional ``convergence`` and ``calibration`` when applicable.

    Raises
    ------
    ValueError
        When ``design_type`` is unsupported or ``response`` length mismatches
        ``n_runs``.
    """
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
        # GP surrogate when doekit[bo] (scikit-learn) is installed, else OLS.
        kwargs["surrogate"] = "auto"
    proposal = ed.propose_next_runs(design, response=y, n_add=n_add, model=model,
                                    intent=intent, **kwargs)
    ctx = ed.context_from_proposal(
        proposal, budget_total=int(budget or 0), budget_spent=design.n_runs)
    # Convergence stop gate (optional): needs a per-generation history.
    convergence = None
    if history is not None:
        metric_key = "best_so_far" if intent == "optimize" else "delta_D_efficiency"
        convergence = ed.check_convergence(list(history), metric_key=metric_key)
    decision = ed.decide_next_action(ctx, convergence=convergence)
    # Per-step diagnostics ride along so the agent sees the same monitoring
    # signals the core exposes (power, G-eff, budget, uncertainty, convergence).
    diagnostics = ed.diagnose_step(
        ctx.metrics, budget_remaining=ctx.budget_remaining,
        uncertainty=ctx.uncertainty, convergence=convergence)
    view = ed.interpret(proposal)
    out = {
        "intent": proposal.intent,
        "n_added": proposal.added.n_runs,
        "proposed_runs": proposal.added.matrix.to_dict("records"),
        "interpretation": view.to_dict(),
        "decision": decision.to_dict(),
        "diagnostics": diagnostics.to_dict(),
        "context_addition": (view.for_llm() + "\n\n" + decision.for_llm()),
    }
    if convergence is not None:
        out["convergence"] = convergence.to_dict()
    calibration = _surrogate_calibration(getattr(proposal, "surrogate", None))
    if calibration is not None:
        out["calibration"] = calibration
    return out


# Agent-facing tool name -> implementation (clean names for the LLM).
TOOLS = {
    "recommend": tool_recommend,
    "evaluate": tool_evaluate,
    "propose_and_decide": tool_propose_and_decide,
}


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def build_server(name: str = "doekit"):
    """Create a FastMCP server exposing the doekit tools.

    Registers :data:`TOOLS` (``recommend``, ``evaluate``, ``propose_and_decide``)
    with fastmcp. Requires ``pip install "doekit[mcp]"``.

    Parameters
    ----------
    name : str, default ``"doekit"``
        MCP server name passed to FastMCP.

    Returns
    -------
    FastMCP
        Configured server; call ``.run()`` to start stdio transport.

    Raises
    ------
    ImportError
        When ``fastmcp`` is not installed.

    Examples
    --------
    Install ``doekit[mcp]``, then from Python::

        from doekit.adapters.mcp import build_server
        build_server().run()

    Or: ``python -m doekit.adapters.mcp``.
    """
    FastMCP = _require_fastmcp()
    server = FastMCP(name)
    for tool_name, fn in TOOLS.items():
        server.tool(name=tool_name)(fn)
    return server


def main() -> None:  # pragma: no cover - entrypoint
    """Entry point for ``python -m doekit.adapters.mcp``."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
