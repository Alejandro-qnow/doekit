"""
Tests de integración decision + monitoring.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from decision import build_context, decide_next_action
from monitoring import (
    DefaultConvergenceChecker,
    DefaultDiagnosticsAnalyzer,
    EventBus,
)


class TestDecisionMonitoringIntegration:
    def test_convergence_can_override_action_to_stop(self):
        history = [
            {"metrics": {"delta_D_efficiency": 10.0}},
            {"metrics": {"delta_D_efficiency": 10.2}},
            {"metrics": {"delta_D_efficiency": 10.3}},
        ]
        conv = DefaultConvergenceChecker(
            metric_key="delta_D_efficiency",
            marginal_threshold=0.5,
            consecutive_required=2,
            min_points=3,
        ).check(history)

        ctx = build_context(
            budget_total=40,
            budget_spent=10,
            metrics={
                "delta_D_efficiency": 8.0,
                "delta_mean_power": 0.04,
                "delta_G_efficiency": 0.5,
                "n_add": 2,
            },
        )

        decision = decide_next_action(ctx, convergence_result=conv)
        assert decision.action == "stop"
        assert "Override por monitoring" in decision.reasoning

    def test_diagnostics_blocker_can_override_action_to_stop_and_emit_events(self):
        np.random.seed(42)
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comp = ed.compare_designs(design, proposal.combined, model=model)

        ctx = build_context(
            budget_total=10,
            budget_spent=9,
            metrics={
                "delta_D_efficiency": float(comp.delta.get("D_efficiency", 0.0)),
                "delta_mean_power": 0.0,
                "delta_G_efficiency": -3.0,
                "n_add": 5.0,
            },
            uncertainty=0.9,
            proposal=proposal,
            comparison=comp,
        )

        diagnostics = DefaultDiagnosticsAnalyzer().analyze(
            metrics=ctx.metrics,
            budget_remaining=ctx.budget_remaining,
            uncertainty=ctx.uncertainty,
        )
        assert diagnostics.has_blockers is True

        bus = EventBus()
        decision = decide_next_action(
            ctx,
            diagnostics_report=diagnostics,
            event_bus=bus,
        )

        assert decision.action == "stop"
        assert "Override por diagnostico bloqueante" in decision.reasoning

        event_types = [event.event_type for event in bus.get_events()]
        assert "decision.scored" in event_types
        assert "diagnostics.generated" in event_types
        assert "decision.finalized" in event_types
