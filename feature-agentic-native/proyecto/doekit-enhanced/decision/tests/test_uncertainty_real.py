"""
Tests de Fase 3 (uncertainty) con validacion numerica y flujo real de doekit.
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
    estimate_uncertainty,
    estimate_uncertainty_from_proposal,
)
from decision.scoring import ContinuationScorer


class TestUncertaintyQuantification:
    def test_estimate_uncertainty_basic_ranges(self):
        est = estimate_uncertainty(expected_gain=0.3, sigma_hat=0.1, threshold=0.0)

        assert est.sigma_hat > 0
        assert est.ci_low < est.ci_high
        assert 0.0 <= est.probability_of_improvement <= 1.0
        assert est.expected_improvement >= 0.0
        assert 0.0 <= est.normalized_uncertainty <= 1.0

    def test_higher_sigma_reduces_confidence_in_context(self):
        low = estimate_uncertainty(expected_gain=0.2, sigma_hat=0.05)
        high = estimate_uncertainty(expected_gain=0.2, sigma_hat=0.30)

        assert low.normalized_uncertainty < high.normalized_uncertainty


class TestUncertaintyWithDecisionEngine:
    def test_uncertainty_propagates_to_decision_score(self):
        metrics = {
            "delta_D_efficiency": 8.0,
            "delta_mean_power": 0.06,
            "delta_G_efficiency": -0.5,
            "n_add": 3,
        }

        base_ctx = build_context(40, 12, metrics=metrics)
        base_score = ContinuationScorer().score(base_ctx)

        uncertain_ctx = build_context(40, 12, metrics=metrics)
        est = estimate_uncertainty(expected_gain=0.2, sigma_hat=0.25)
        uncertain_decision = decide_next_action(uncertain_ctx, uncertainty_estimate=est)

        assert uncertain_decision.score.composite <= base_score.composite
        assert "uncertainty" in uncertain_decision.metadata

    def test_uncertainty_from_real_proposal(self):
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)

        np.random.seed(21)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comparison = ed.compare_designs(design, proposal.combined, model=model)

        est = estimate_uncertainty_from_proposal(proposal, comparison=comparison)

        assert est.sigma_hat > 0
        assert 0.0 <= est.probability_of_improvement <= 1.0
        assert est.metadata.get("source") == "proposal"

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
            metrics=metrics,
            proposal=proposal,
            comparison=comparison,
        )

        decision = decide_next_action(ctx, uncertainty_estimate=est)
        assert decision.action in ["continue", "refine_model", "stop"]
        assert 0.0 <= decision.confidence <= 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
