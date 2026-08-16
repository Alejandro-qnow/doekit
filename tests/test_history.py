"""Meta-learning from past experiments: priors + historical advice (M4)."""

import json

import numpy as np

import doekit as ed
from doekit.orchestration.advise import (
    ExperimentRecord, ExperimentHistory, PriorEstimate, HistoricalRecommendation,
    learn_priors, historical_recommendation,
)


def _rec(eid, objective, factors, d_gain):
    return ExperimentRecord(eid, objective, factors,
                            metrics={"delta_D_efficiency": d_gain})


def test_find_similar_ranks_by_objective_and_overlap():
    hist = ExperimentHistory([
        _rec("a", "optimization", ["T", "pH", "cat"], 8.0),
        _rec("b", "optimization", ["T", "pH"], 6.0),
        _rec("c", "screening", ["X1", "X2"], 1.0),
    ])
    similar = hist.find_similar("optimization", ["T", "pH", "cat"])
    assert similar[0].experiment_id == "a"       # objective + full overlap
    assert "c" not in [r.experiment_id for r in similar]  # different objective, no overlap


def test_learn_priors_averages_similar():
    hist = ExperimentHistory([
        _rec("a", "optimization", ["T", "pH"], 10.0),
        _rec("b", "optimization", ["T", "pH"], 6.0),
    ])
    prior = learn_priors(hist, "optimization", ["T", "pH"])
    assert isinstance(prior, PriorEstimate)
    assert prior.n_sources == 2
    assert prior.expected_delta_d_efficiency == 8.0
    json.dumps(prior.to_dict())


def test_learn_priors_fallback_without_history():
    prior = learn_priors(ExperimentHistory(), "optimization", ["T"])
    assert prior.n_sources == 0
    assert prior.metadata.get("fallback") is True


def test_historical_recommendation_expansion_vs_caution():
    favorable = ExperimentHistory([_rec("a", "optimization", ["T"], 9.0)])
    cautious = ExperimentHistory([_rec("b", "optimization", ["T"], 1.0)])
    r1 = historical_recommendation(favorable, "optimization", ["T"])
    r2 = historical_recommendation(cautious, "optimization", ["T"])
    assert isinstance(r1, HistoricalRecommendation)
    assert "expansion" in r1.title.lower()
    assert "caution" in r2.title.lower()


def test_historical_recommendation_no_history():
    r = historical_recommendation(ExperimentHistory(), "optimization", ["T"])
    assert "No comparable" in r.title
    json.dumps(r.to_dict())


def test_record_roundtrip():
    rec = _rec("a", "optimization", ["T", "pH"], 5.0)
    rt = ExperimentRecord.from_dict(rec.to_dict())
    assert rt.experiment_id == "a"
    assert rt.factor_names == ["T", "pH"]
    assert rt.metrics["delta_D_efficiency"] == 5.0


def test_history_from_project_reads_waves(tmp_path):
    # build a real traceable project with one wave, then read it back as history
    bb = ed.box_behnken({"a": (0, 1), "b": (0, 1), "c": (0, 1)}, center=3)
    exp = ed.experiment(design=bb, model=ed.Model.full_quadratic(["a", "b", "c"]),
                        responses=["y"])
    exp.evaluate()
    rng = np.random.default_rng(0)
    exp.ingest(rng.normal(size=bb.n_runs))
    proj = ed.project("hist study", root=str(tmp_path))
    exp.save(proj)

    hist = ExperimentHistory.from_project(proj)
    assert len(hist) >= 1
    rec = hist.all()[0]
    assert set(["a", "b", "c"]).issubset(set(rec.factor_names))
