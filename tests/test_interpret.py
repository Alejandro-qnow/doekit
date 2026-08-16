"""Uniform interpretation of doekit results (presentation.narrative.interpret)."""

import json

import numpy as np
import pytest

import doekit as ed
from doekit import interpret, Interpretation


def _rsm():
    cols = list(ed.central_composite(2).matrix.columns)
    d = ed.central_composite(2)
    facs = [ed.ContinuousFactor(cols[0], -1, 1), ed.ContinuousFactor(cols[1], -1, 1)]
    d = ed.Design(matrix=d.matrix, factors=facs, model=ed.Model.full_quadratic(cols))
    X = d.matrix[cols].to_numpy(dtype=float)
    y = 5 - 3 * ((X[:, 0] - 0.4) ** 2 + (X[:, 1] + 0.3) ** 2) + 0.02 * X[:, 0]
    return d, cols, y


def _assert_valid(view, kind):
    assert isinstance(view, Interpretation)
    assert view.kind == kind
    assert view.headline and view.reasoning
    text = view.for_llm()
    assert kind in text and view.headline in text
    d = view.to_dict()
    assert d["schema"] == "doekit.Interpretation/1"
    json.dumps(d)  # must be JSON-safe


def test_interpret_recommendation():
    rec = ed.recommend_design(goal="optimization", factors=3, budget=20)
    view = interpret(rec)
    _assert_valid(view, "recommendation")
    assert view.facts["method"] == rec.method


def test_interpret_evaluation():
    ev = ed.evaluate(ed.central_composite(2), n_region=500, seed=0)
    view = interpret(ev)
    _assert_valid(view, "evaluation")
    assert "D_eff" in view.facts


def test_interpret_fit():
    d, cols, y = _rsm()
    fit = ed.fit_linear_model(d, y)
    view = interpret(fit)
    _assert_valid(view, "fit")
    assert view.facts["r_squared"] == pytest.approx(round(float(fit.r_squared), 4))


def test_interpret_proposal_learn():
    d, cols, y = _rsm()
    prop = ed.propose_next_runs(d, response=y, n_add=3, seed=1)
    view = interpret(prop)
    _assert_valid(view, "proposal")
    assert view.facts["intent"] == "learn"


def test_interpret_proposal_optimize_reads_native_fields():
    d, cols, y = _rsm()
    prop = ed.propose_next_runs(d, response=y, n_add=3, intent="optimize",
                                surrogate="ols", seed=1)
    view = interpret(prop)
    _assert_valid(view, "proposal")
    assert view.facts["intent"] == "optimize"
    assert view.facts["acquisition"] == "ei"
    assert view.facts["mode"] in {"exploring", "exploiting", "balanced", "unknown"}
    # the optimize interpretation nudges the agent to audit calibration
    assert any("calibration" in r.lower() for r in view.recommendations)


def test_interpret_comparison():
    a = ed.plackett_burman(5)
    b = ed.fold(a)
    cmp = ed.compare_designs(a, b, n_region=500, seed=0)
    view = interpret(cmp)
    _assert_valid(view, "comparison")
    assert "delta_D_efficiency" in view.facts


def test_interpret_unknown_type_raises():
    with pytest.raises(TypeError):
        interpret({"not": "a result"})


def test_facts_do_not_invent_numbers():
    # every fact must trace back to the source object's own to_dict()
    d, cols, y = _rsm()
    prop = ed.propose_next_runs(d, response=y, n_add=2, seed=0)
    view = interpret(prop)
    assert view.facts["n_add"] == prop.added.n_runs
