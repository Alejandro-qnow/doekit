"""
Tests para Fase 2 (decision) usando datos reales de doekit cuando aplica.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from decision import (
    build_context,
    decide_next_action,
    ContinuationScorer,
    MultiObjectiveScorer,
)
from decision.policies import ThresholdPolicy, RiskAdaptivePolicy, BudgetAwarePolicy


class TestDecisionContextAndScoring:
    def test_context_budget_properties(self):
        ctx = build_context(budget_total=50, budget_spent=12)
        assert ctx.budget_remaining == 38
        assert 0.23 < ctx.budget_usage_ratio < 0.25

    def test_continuation_scorer_positive_signal(self):
        ctx = build_context(
            budget_total=40,
            budget_spent=10,
            metrics={
                "delta_D_efficiency": 12.0,
                "delta_mean_power": 0.08,
                "delta_G_efficiency": 1.5,
                "n_add": 4,
            },
            uncertainty=0.1,
        )

        score = ContinuationScorer().score(ctx)
        assert score.benefit > 0
        assert score.cost >= 0
        assert isinstance(score.composite, float)
        assert len(score.rationale) >= 3

    def test_multi_objective_scorer(self):
        ctx = build_context(
            budget_total=30,
            budget_spent=5,
            metrics={"objectives": {"precision": 0.7, "prediction": 0.6, "cost": 0.2}},
            uncertainty=0.2,
        )

        score = MultiObjectiveScorer().score(ctx)
        assert score.components["precision"] == 0.7
        assert score.components["prediction"] == 0.6
        assert score.components["cost"] == 0.2


class TestPolicies:
    def test_threshold_policy_continue(self):
        ctx = build_context(
            budget_total=30,
            budget_spent=10,
            metrics={"delta_D_efficiency": 15.0, "delta_mean_power": 0.1, "n_add": 2},
        )

        score = ContinuationScorer().score(ctx)
        decision = ThresholdPolicy().decide(ctx, score)
        assert decision.action in ["continue", "refine_model", "stop"]
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.context_addition) > 30

    def test_risk_adaptive_policy_low_vs_high(self):
        base_metrics = {
            "delta_D_efficiency": 5.0,
            "delta_mean_power": 0.03,
            "delta_G_efficiency": -2.0,
            "n_add": 3,
        }

        low_ctx = build_context(40, 15, risk_tolerance="low", metrics=base_metrics)
        high_ctx = build_context(40, 15, risk_tolerance="high", metrics=base_metrics)
        score = ContinuationScorer().score(low_ctx)

        low_decision = RiskAdaptivePolicy().decide(low_ctx, score)
        high_decision = RiskAdaptivePolicy().decide(high_ctx, score)

        assert low_decision.action in ["continue", "refine_model", "stop"]
        assert high_decision.action in ["continue", "refine_model", "stop"]

    def test_budget_aware_policy_exhausted(self):
        ctx = build_context(
            budget_total=12,
            budget_spent=12,
            metrics={"delta_D_efficiency": 20.0, "delta_mean_power": 0.2, "n_add": 3},
        )

        score = ContinuationScorer().score(ctx)
        decision = BudgetAwarePolicy().decide(ctx, score)

        assert decision.action == "stop"
        assert decision.confidence >= 0.9


class TestEndToEndWithDoekit:
    def test_decide_next_action_with_real_comparison_metrics(self):
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)

        np.random.seed(42)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comparison = ed.compare_designs(design, proposal.combined, model=model)

        delta = comparison.delta
        metrics = {
            "delta_D_efficiency": float(delta.get("D_efficiency", 0.0)),
            "delta_mean_power": float(delta.get("mean_power", 0.0)),
            "delta_G_efficiency": float(delta.get("G_efficiency", 0.0)),
            "n_add": float(delta.get("n_runs", 0.0)),
        }

        ctx = build_context(
            budget_total=40,
            budget_spent=18,
            risk_tolerance="moderate",
            metrics=metrics,
            uncertainty=0.15,
            proposal=proposal,
            comparison=comparison,
        )

        decision = decide_next_action(ctx)

        assert decision.action in ["continue", "refine_model", "stop"]
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.score is not None
        assert hasattr(decision.score, "composite")
        assert len(decision.reasoning) > 20
        assert len(decision.context_addition) > 60


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
