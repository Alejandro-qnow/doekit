"""Sequential monitoring: convergence detection and step diagnostics (M3)."""

import json

import numpy as np

import doekit as ed
from doekit.orchestration.decide import (
    check_convergence, ConvergenceResult, diagnose_step, DiagnosticsReport,
    decide_next_action, DecisionContext,
)


# --------------------------------------------------------------------------
# convergence
# --------------------------------------------------------------------------

def test_convergence_detected_on_flat_tail():
    # best-so-far plateaus at the end -> converged
    history = [2.0, 3.5, 4.2, 4.25, 4.28]
    res = check_convergence(history, metric_key="best_so_far",
                            marginal_threshold=0.1, consecutive_required=2)
    assert isinstance(res, ConvergenceResult)
    assert res.converged and res.should_stop
    assert res.consecutive_hits >= 2


def test_convergence_not_detected_while_improving():
    history = [2.0, 3.0, 4.0, 5.0, 6.0]
    res = check_convergence(history, metric_key="best_so_far",
                            marginal_threshold=0.1, consecutive_required=2)
    assert not res.converged and not res.should_stop


def test_convergence_insufficient_history():
    res = check_convergence([1.0, 2.0], min_points=3)
    assert not res.should_stop
    assert "Insufficient" in res.reason


def test_convergence_reads_dicts_and_metrics_key():
    history = [
        {"metrics": {"delta_D_efficiency": 10.0}},
        {"metrics": {"delta_D_efficiency": 10.2}},
        {"metrics": {"delta_D_efficiency": 10.3}},
    ]
    res = check_convergence(history, metric_key="delta_D_efficiency",
                            marginal_threshold=0.5, consecutive_required=2)
    assert res.converged
    d = res.to_dict()
    assert d["schema"] == "doekit.ConvergenceResult/1"
    json.dumps(d)


def test_convergence_feeds_decision_engine():
    history = [4.0, 4.1, 4.15, 4.16]
    conv = check_convergence(history, metric_key="best_so_far",
                             marginal_threshold=0.1, consecutive_required=2)
    ctx = DecisionContext(intent="optimize",
                          metrics={"predicted_improvement": 0.9, "n_add": 3})
    # without convergence this would augment; convergence forces stop
    assert decide_next_action(ctx, convergence=conv).action == "stop"


# --------------------------------------------------------------------------
# step diagnostics
# --------------------------------------------------------------------------

def test_diagnose_flags_prediction_degradation_and_power():
    rep = diagnose_step({"delta_mean_power": 0.0, "delta_G_efficiency": -5.0,
                         "n_add": 4})
    assert isinstance(rep, DiagnosticsReport)
    codes = {i.code for i in rep.issues}
    assert "LOW_POWER_GAIN" in codes
    assert "PREDICTION_DEGRADATION" in codes
    assert not rep.has_blockers


def test_diagnose_budget_overflow_is_blocker():
    rep = diagnose_step({"n_add": 8, "delta_mean_power": 0.2}, budget_remaining=3)
    assert rep.has_blockers
    assert any(i.code == "BUDGET_OVERFLOW" and i.severity == "error"
               for i in rep.issues)


def test_diagnose_high_uncertainty_warning():
    rep = diagnose_step({"delta_mean_power": 0.2}, uncertainty=0.8)
    assert any(i.code == "HIGH_UNCERTAINTY" for i in rep.issues)


def test_diagnose_clean_step_no_issues():
    rep = diagnose_step({"delta_mean_power": 0.2, "delta_G_efficiency": 1.0,
                         "n_add": 2}, budget_remaining=10, uncertainty=0.1)
    assert not rep.has_issues
    assert "No relevant" in rep.summary
    json.dumps(rep.to_dict())


def test_diagnose_convergence_info():
    conv = check_convergence([4.0, 4.05, 4.06], marginal_threshold=0.1,
                             consecutive_required=2)
    rep = diagnose_step({"delta_mean_power": 0.2}, convergence=conv)
    assert any(i.code == "CONVERGENCE_REACHED" and i.severity == "info"
               for i in rep.issues)


# --------------------------------------------------------------------------
# Experiment.decide_next with history
# --------------------------------------------------------------------------

def test_experiment_decide_next_history_forces_stop():
    cols = list(ed.central_composite(2).matrix.columns)
    d = ed.central_composite(2)
    facs = [ed.ContinuousFactor(cols[0], -1, 1), ed.ContinuousFactor(cols[1], -1, 1)]
    d = ed.Design(matrix=d.matrix, factors=facs, model=ed.Model.full_quadratic(cols))
    X = d.matrix[cols].to_numpy(dtype=float)
    y = 5 - 3 * ((X[:, 0] - 0.4) ** 2 + (X[:, 1] + 0.3) ** 2)
    exp = ed.experiment(design=d, model=d.model, responses=["y"])
    exp.ingest(y)
    flat = [4.0, 4.05, 4.06, 4.07]  # plateaued best-so-far
    decision = exp.decide_next(n_add=3, intent="optimize", surrogate="ols",
                               budget=d.n_runs + 6, history=flat)
    assert decision.action == "stop"
