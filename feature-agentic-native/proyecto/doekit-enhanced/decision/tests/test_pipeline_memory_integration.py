"""
Test de integración pipeline + memory/integrations (smoke).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from decision import build_context, run_decision_pipeline
from memory import ExperimentStore, ExperimentRecord, PriorLearner
from integrations import propose_with_bayesian_opt


class TestPipelineMemoryIntegration:
    def test_pipeline_can_work_with_memory_prior_and_bo_proposal(self):
        store = ExperimentStore()
        store.add(
            ExperimentRecord(
                experiment_id="exp-prev",
                objective="optimization",
                factor_names=["X1", "X2"],
                metrics={"delta_D_efficiency": 8.0, "delta_mean_power": 0.05, "uncertainty": 0.25},
            )
        )

        prior = PriorLearner(store).learn("optimization", ["X1", "X2"], top_k=3)
        assert prior.n_sources >= 1

        candidates = [{"X1": -0.5, "X2": 0.1}, {"X1": 0.4, "X2": -0.2}]
        bo = propose_with_bayesian_opt(
            candidate_pool=candidates,
            objective_values=[0.4, 0.45],
            uncertainty_values=[0.08, 0.12],
        )
        assert bo.selected_index in [0, 1]

        np.random.seed(10)
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        y = np.random.randn(design.n_runs)
        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comparison = ed.compare_designs(design, proposal.combined, model=model)

        metrics = {
            "delta_D_efficiency": float(comparison.delta.get("D_efficiency", 0.0)),
            "delta_mean_power": float(comparison.delta.get("mean_power", 0.0)),
            "delta_G_efficiency": float(comparison.delta.get("G_efficiency", 0.0)),
            "n_add": float(comparison.delta.get("n_runs", 0.0)),
        }

        ctx = build_context(
            budget_total=40,
            budget_spent=15,
            metrics=metrics,
            proposal=proposal,
            comparison=comparison,
            metadata={"prior_expected_d_eff": prior.expected_delta_d_efficiency, "bo_candidate": bo.selected_candidate},
        )

        out = run_decision_pipeline(ctx, history=None)
        assert out.decision.action in ["continue", "refine_model", "stop"]
