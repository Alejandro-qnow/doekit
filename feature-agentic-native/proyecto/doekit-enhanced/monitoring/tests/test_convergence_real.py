"""
Tests para monitoring.convergence con criterios de mejora marginal
y con historico real de waves en doekit.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from monitoring import DefaultConvergenceChecker


class TestConvergenceMarginalCriteria:
    def test_converges_with_small_marginal_improvements(self):
        history = [
            {"wave": 1, "metrics": {"delta_D_efficiency": 10.0}},
            {"wave": 2, "metrics": {"delta_D_efficiency": 10.4}},
            {"wave": 3, "metrics": {"delta_D_efficiency": 10.7}},
            {"wave": 4, "metrics": {"delta_D_efficiency": 10.9}},
        ]

        checker = DefaultConvergenceChecker(
            metric_key="delta_D_efficiency",
            marginal_threshold=0.5,
            consecutive_required=2,
            min_points=3,
        )
        result = checker.check(history)

        assert result.converged is True
        assert result.should_stop is True
        assert result.consecutive_hits >= 2

    def test_not_converged_with_large_last_jump(self):
        history = [
            {"wave": 1, "metrics": {"delta_D_efficiency": 10.0}},
            {"wave": 2, "metrics": {"delta_D_efficiency": 10.2}},
            {"wave": 3, "metrics": {"delta_D_efficiency": 10.3}},
            {"wave": 4, "metrics": {"delta_D_efficiency": 12.0}},
        ]

        checker = DefaultConvergenceChecker(
            metric_key="delta_D_efficiency",
            marginal_threshold=0.5,
            consecutive_required=2,
            min_points=3,
        )
        result = checker.check(history)

        assert result.converged is False
        assert result.should_stop is False


class TestConvergenceWithRealWaveHistory:
    def test_checker_handles_real_doekit_wave_history(self):
        np.random.seed(42)

        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)

        history = []

        current_design = design
        for wave in range(3):
            X = current_design.matrix.values
            y = 20 + 3 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(len(X)) * 0.8

            proposal = ed.propose_next_runs(
                current_design,
                response=y,
                n_add=2,
                model=model,
            )
            comparison = ed.compare_designs(current_design, proposal.combined, model=model)

            delta = comparison.delta
            history.append(
                {
                    "wave": wave + 1,
                    "metrics": {
                        "delta_D_efficiency": float(delta.get("D_efficiency", 0.0)),
                        "delta_mean_power": float(delta.get("mean_power", 0.0)),
                        "delta_G_efficiency": float(delta.get("G_efficiency", 0.0)),
                    },
                }
            )

            current_design = proposal.combined

        checker = DefaultConvergenceChecker(
            metric_key="delta_D_efficiency",
            marginal_threshold=2.0,
            consecutive_required=2,
            min_points=3,
        )
        result = checker.check(history)

        assert result.observed_points == 3
        assert isinstance(result.converged, bool)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 10
        assert "values" in result.metadata
        assert len(result.metadata["values"]) == 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
