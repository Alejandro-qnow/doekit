"""
Tests para pipeline opcional y configurable.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from decision import (
    build_context,
    DecisionPipelineConfig,
    run_decision_pipeline,
)


class TestDecisionPipeline:
    def test_pipeline_executes_all_enabled_stages(self):
        np.random.seed(42)
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comparison = ed.compare_designs(design, proposal.combined, model=model)

        history = [
            {"wave": 1, "metrics": {"delta_D_efficiency": 10.0}},
            {"wave": 2, "metrics": {"delta_D_efficiency": 10.3}},
            {"wave": 3, "metrics": {"delta_D_efficiency": 10.5}},
        ]

        ctx = build_context(
            budget_total=40,
            budget_spent=18,
            metrics={
                "delta_D_efficiency": float(comparison.delta.get("D_efficiency", 0.0)),
                "delta_mean_power": float(comparison.delta.get("mean_power", 0.0)),
                "delta_G_efficiency": float(comparison.delta.get("G_efficiency", 0.0)),
                "n_add": float(comparison.delta.get("n_runs", 0.0)),
            },
            proposal=proposal,
            comparison=comparison,
        )

        cfg = DecisionPipelineConfig(
            enable_uncertainty=True,
            enable_convergence=True,
            enable_diagnostics=True,
            enable_events=True,
            convergence_marginal_threshold=0.5,
            convergence_consecutive_required=2,
            convergence_min_points=3,
        )

        out = run_decision_pipeline(ctx, history=history, config=cfg)

        assert out.uncertainty_estimate is not None
        assert out.convergence_result is not None
        assert out.diagnostics_report is not None
        assert out.event_bus is not None
        assert out.convergence_result.should_stop is True
        assert out.decision.action == "stop"

        assert "uncertainty" in out.executed_stages
        assert "convergence" in out.executed_stages
        assert "diagnostics" in out.executed_stages
        assert "decision" in out.executed_stages

    def test_pipeline_allows_stage_disabling(self):
        ctx = build_context(
            budget_total=30,
            budget_spent=10,
            metrics={
                "delta_D_efficiency": 5.0,
                "delta_mean_power": 0.03,
                "delta_G_efficiency": -0.2,
                "n_add": 2.0,
            },
        )

        cfg = DecisionPipelineConfig(
            enable_uncertainty=False,
            enable_convergence=False,
            enable_diagnostics=False,
            enable_events=False,
        )

        out = run_decision_pipeline(ctx, history=None, config=cfg)

        assert out.uncertainty_estimate is None
        assert out.convergence_result is None
        assert out.diagnostics_report is None
        assert out.event_bus is None
        assert out.executed_stages == ["decision"]
        assert out.decision.action in ["continue", "refine_model", "stop"]
