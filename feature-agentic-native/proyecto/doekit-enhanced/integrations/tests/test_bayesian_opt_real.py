"""
Tests para integrations.bayesian_opt (Fase 6).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from integrations import BayesianOptAdapter, propose_with_bayesian_opt


class TestBayesianOptAdapter:
    def test_propose_with_fallback_ei(self):
        candidates = [
            {"x1": -1.0, "x2": 0.0},
            {"x1": 0.2, "x2": 0.3},
            {"x1": 0.8, "x2": 0.7},
        ]
        objective_values = [0.4, 0.45, 0.5]
        uncertainty_values = [0.05, 0.1, 0.2]

        proposal = propose_with_bayesian_opt(
            candidate_pool=candidates,
            objective_values=objective_values,
            uncertainty_values=uncertainty_values,
            exploration_weight=0.2,
        )

        # With this synthetic setup, candidate 2 should dominate EI+exploration.
        assert proposal.selected_index == 2
        assert isinstance(proposal.selected_candidate, dict)
        assert len(proposal.scores) == 3
        assert proposal.method in ["fallback_ei", "botorch"]
        assert proposal.scores[0].score >= proposal.scores[1].score >= proposal.scores[2].score
        assert proposal.scores[0].details["ei"] >= 0.0

    def test_adapter_rejects_invalid_lengths(self):
        adapter = BayesianOptAdapter()
        candidates = [{"x": 1.0}, {"x": 2.0}]

        with pytest.raises(ValueError, match="igual longitud"):
            adapter.propose(candidates, objective_values=[0.1], uncertainty_values=[0.2, 0.3])
