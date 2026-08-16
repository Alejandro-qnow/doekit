"""
Tests para monitoring.diagnostics con reglas automáticas y datos reales de doekit.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from monitoring import DefaultDiagnosticsAnalyzer, DefaultConvergenceChecker


class TestDiagnosticsRules:
    def test_detects_low_power_and_high_uncertainty(self):
        analyzer = DefaultDiagnosticsAnalyzer()

        report = analyzer.analyze(
            metrics={
                "delta_mean_power": 0.001,
                "delta_G_efficiency": -2.5,
                "n_add": 6,
            },
            budget_remaining=4,
            uncertainty=0.85,
        )

        assert report.has_issues is True
        assert report.has_blockers is True
        codes = [issue.code for issue in report.issues]
        assert "LOW_POWER_GAIN" in codes
        assert "PREDICTION_DEGRADATION" in codes
        assert "BUDGET_OVERFLOW" in codes
        assert "HIGH_UNCERTAINTY" in codes


class TestDiagnosticsWithRealConvergence:
    def test_analyze_with_real_convergence_signal(self):
        np.random.seed(7)
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)

        history = []
        current = design

        for wave in range(3):
            y = np.random.randn(current.n_runs)
            proposal = ed.propose_next_runs(current, response=y, n_add=2, model=model)
            comp = ed.compare_designs(current, proposal.combined, model=model)

            history.append(
                {"wave": wave + 1, "metrics": {"delta_D_efficiency": float(comp.delta.get("D_efficiency", 0.0))}}
            )
            current = proposal.combined

        conv = DefaultConvergenceChecker(
            metric_key="delta_D_efficiency",
            marginal_threshold=3.0,
            consecutive_required=2,
            min_points=3,
        ).check(history)

        analyzer = DefaultDiagnosticsAnalyzer()
        report = analyzer.analyze(
            metrics={"delta_mean_power": 0.02, "delta_G_efficiency": -0.2, "n_add": 2},
            budget_remaining=10,
            uncertainty=0.2,
            convergence_result=conv,
        )

        codes = [issue.code for issue in report.issues]
        if conv.should_stop:
            assert "CONVERGENCE_REACHED" in codes
