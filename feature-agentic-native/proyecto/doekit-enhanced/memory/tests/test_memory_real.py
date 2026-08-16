"""
Tests para memory (Fase 5).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import (
    ExperimentRecord,
    ExperimentStore,
    PriorLearner,
    HistoricalRecommender,
)


class TestExperimentStore:
    def test_add_get_and_similarity(self):
        store = ExperimentStore()

        store.add(
            ExperimentRecord(
                experiment_id="exp-1",
                objective="optimization",
                factor_names=["X1", "X2", "X3"],
                metrics={"delta_D_efficiency": 8.0, "delta_mean_power": 0.06, "uncertainty": 0.2},
            )
        )
        store.add(
            ExperimentRecord(
                experiment_id="exp-2",
                objective="screening",
                factor_names=["X1", "X4"],
                metrics={"delta_D_efficiency": 2.0, "delta_mean_power": 0.01, "uncertainty": 0.6},
            )
        )

        assert store.size() == 2
        assert store.get("exp-1") is not None

        similar = store.find_similar("optimization", ["X1", "X2"], top_k=2)
        assert len(similar) >= 1
        assert similar[0].experiment_id == "exp-1"


class TestPriorLearner:
    def test_learn_prior_from_history(self):
        store = ExperimentStore()
        store.add(
            ExperimentRecord(
                experiment_id="exp-1",
                objective="optimization",
                factor_names=["X1", "X2"],
                metrics={"delta_D_efficiency": 10.0, "delta_mean_power": 0.08, "uncertainty": 0.2},
            )
        )
        store.add(
            ExperimentRecord(
                experiment_id="exp-2",
                objective="optimization",
                factor_names=["X1", "X3"],
                metrics={"delta_D_efficiency": 6.0, "delta_mean_power": 0.04, "uncertainty": 0.3},
            )
        )

        prior = PriorLearner(store).learn("optimization", ["X1", "X2"], top_k=5)
        assert prior.n_sources >= 1
        assert prior.expected_delta_d_efficiency > 0
        assert 0.0 <= prior.expected_uncertainty <= 1.0


class TestHistoricalRecommender:
    def test_recommender_returns_actions(self):
        store = ExperimentStore()
        store.add(
            ExperimentRecord(
                experiment_id="exp-1",
                objective="optimization",
                factor_names=["X1", "X2"],
                metrics={"delta_D_efficiency": 7.0},
            )
        )

        rec = HistoricalRecommender(store).suggest("optimization", ["X1", "X2"], top_k=3)
        assert len(rec.actions) >= 1
        assert len(rec.title) > 0
        assert len(rec.rationale) > 0
