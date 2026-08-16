"""Decision engine: scoring, policies, gate delegation, Experiment.decide_next."""

import json

import numpy as np
import pandas as pd
import pytest

import doekit as ed
from doekit.orchestration.decide import (
    Decision, DecisionContext, DecisionScore, ContinuationScorer,
    ThresholdPolicy, RiskAdaptivePolicy, BudgetAwarePolicy,
    decide_next_action, context_from_proposal,
)


def _rsm(seed=0):
    cols = list(ed.central_composite(2).matrix.columns)
    d = ed.central_composite(2)
    facs = [ed.ContinuousFactor(cols[0], -1, 1), ed.ContinuousFactor(cols[1], -1, 1)]
    d = ed.Design(matrix=d.matrix, factors=facs, model=ed.Model.full_quadratic(cols))
    X = d.matrix[cols].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    y = 5 - 3 * ((X[:, 0] - 0.4) ** 2 + (X[:, 1] + 0.3) ** 2) + 0.02 * rng.standard_normal(len(X))
    return d, cols, y


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def test_learn_benefit_from_efficiency_gains():
    ctx = DecisionContext(intent="learn", budget_total=40, budget_spent=10,
                          metrics={"delta_D_efficiency": 12.0,
                                   "delta_mean_power": 0.08, "n_add": 4})
    score = ContinuationScorer().score(ctx)
    assert score.benefit > 0
    assert "d_eff_gain" in score.components


def test_optimize_benefit_ignores_d_efficiency():
    # optimize is scored by predicted improvement, NOT by D-efficiency (which
    # may fall while the outcome improves) — the paper's key tension.
    ctx = DecisionContext(intent="optimize", budget_total=40, budget_spent=10,
                          metrics={"delta_D_efficiency": -11.0,
                                   "predicted_improvement": 0.9, "n_add": 4})
    score = ContinuationScorer().score(ctx)
    assert score.benefit == 1.0  # positive predicted improvement
    assert "d_eff_gain" not in score.components
    assert "predicted_improvement" in score.components


def test_optimize_no_improvement_zero_benefit():
    ctx = DecisionContext(intent="optimize", metrics={"predicted_improvement": 0.0,
                                                      "n_add": 4})
    assert ContinuationScorer().score(ctx).benefit == 0.0


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------

def test_threshold_policy_actions():
    hi = DecisionScore(0.5, 0.6, 0.1, 0.0, 0.0)
    mid = DecisionScore(0.0, 0.1, 0.1, 0.0, 0.0)
    lo = DecisionScore(-0.5, 0.0, 0.4, 0.1, 0.1)
    p = ThresholdPolicy()
    assert p.decide(DecisionContext(), hi).action == "augment"
    assert p.decide(DecisionContext(), mid).action == "refine"
    assert p.decide(DecisionContext(), lo).action == "stop"


def test_risk_adaptive_is_more_conservative_when_low():
    score = DecisionScore(0.18, 0.3, 0.1, 0.0, 0.0)  # just above base continue
    low = RiskAdaptivePolicy().decide(DecisionContext(risk_tolerance="low"), score)
    high = RiskAdaptivePolicy().decide(DecisionContext(risk_tolerance="high"), score)
    # low tolerance raises the bar -> not augment; high tolerance -> augment
    assert low.action != "augment"
    assert high.action == "augment"


def test_budget_aware_stops_when_exhausted():
    ctx = DecisionContext(budget_total=12, budget_spent=12)
    score = DecisionScore(0.9, 0.9, 0.0, 0.0, 0.0)
    assert BudgetAwarePolicy().decide(ctx, score).action == "stop"


# --------------------------------------------------------------------------
# decide_next_action: hard gates + serialization
# --------------------------------------------------------------------------

def test_hard_gate_rank_deficient_redesign():
    ctx = DecisionContext(quality="rank_deficient")
    d = decide_next_action(ctx)
    assert d.action == "redesign" and d.gate_status == "redesign"


def test_hard_gate_budget_exhausted_stop():
    ctx = DecisionContext(budget_total=8, budget_spent=8,
                          metrics={"delta_D_efficiency": 30.0, "n_add": 4})
    assert decide_next_action(ctx).action == "stop"


def test_gate_mode_without_metrics_uses_worth_it():
    assert decide_next_action(DecisionContext(worth_it=True)).action == "augment"
    assert decide_next_action(DecisionContext(inference="no_response")).action == "stop"
    assert decide_next_action(
        DecisionContext(inference="saturated_no_test")).action == "augment"


def test_convergence_overrides_to_stop():
    class Conv:
        should_stop = True
        reason = "no marginal gain"
    ctx = DecisionContext(metrics={"delta_D_efficiency": 30.0, "n_add": 4})
    assert decide_next_action(ctx, convergence=Conv()).action == "stop"


def test_decision_serialization_and_gate_status():
    d = decide_next_action(DecisionContext(worth_it=True))
    dd = d.to_dict()
    assert dd["schema"] == "doekit.Decision/1"
    assert dd["action"] in {"augment", "stop", "refine", "redesign"}
    assert d.gate_status in {"augment", "stop", "redesign"}
    json.dumps(dd)
    assert d.action in d.for_llm()


# --------------------------------------------------------------------------
# context_from_proposal + Experiment.decide_next
# --------------------------------------------------------------------------

def test_context_from_proposal_learn():
    d, cols, y = _rsm()
    prop = ed.propose_next_runs(d, response=y, n_add=3, seed=1)
    ctx = context_from_proposal(prop, budget_total=40, budget_spent=d.n_runs)
    assert ctx.intent == "learn"
    assert "delta_D_efficiency" in ctx.metrics
    assert ctx.metrics["n_add"] == 3


def test_context_from_proposal_optimize_reads_native_fields():
    d, cols, y = _rsm()
    prop = ed.propose_next_runs(d, response=y, n_add=3, intent="optimize",
                                surrogate="ols", seed=1)
    ctx = context_from_proposal(prop, budget_total=40, budget_spent=d.n_runs)
    assert ctx.intent == "optimize"
    assert "predicted_improvement" in ctx.metrics
    assert 0.0 <= ctx.uncertainty <= 1.0


def test_experiment_decide_next_end_to_end():
    d, cols, y = _rsm()
    exp = ed.experiment(design=d, model=d.model, responses=["y"])
    exp.ingest(y)
    decision = exp.decide_next(n_add=3, budget=d.n_runs + 6)
    assert isinstance(decision, Decision)
    assert decision.action in {"augment", "refine", "stop", "redesign"}


def test_experiment_decide_next_optimize_intent():
    d, cols, y = _rsm()
    exp = ed.experiment(design=d, model=d.model, responses=["y"])
    exp.ingest(y)
    decision = exp.decide_next(n_add=3, intent="optimize", surrogate="ols",
                               budget=d.n_runs + 6)
    assert decision.action in {"augment", "refine", "stop", "redesign"}


# --------------------------------------------------------------------------
# gate_board still honors the engine (no contract break)
# --------------------------------------------------------------------------

def test_build_conclusions_process_status_contract():
    d, cols, y = _rsm()
    conclusions = ed.build_conclusions(d, response=y)
    status = conclusions["gate_board"]["process"]["status"]
    assert status in {"stop", "augment", "redesign"}
