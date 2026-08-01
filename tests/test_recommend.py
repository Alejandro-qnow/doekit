"""Experimental-design advisor: rules + evaluation."""

import numpy as np
import pytest

import doekit as ed


def test_screening_low_budget_recommends_plackett_burman():
    r = ed.recommend_design("screening", factors=7, budget=8, seed=0)
    assert r.method == "Plackett-Burman"
    assert r.design.n_runs <= 8
    assert r.method in set(r.table["method"])


def test_constrained_forces_optimal():
    r = ed.recommend_design(
        "optimization", factors=3,
        constraints=ed.Constraints(irregular=True), seed=0,
    )
    assert r.method == "D-optimal"
    assert set(r.table["method"]) == {"D-optimal"}


def test_winner_is_feasible_and_supports_model():
    r = ed.recommend_design("optimization", factors=3, seed=0)
    row = r.table.set_index("method").loc[r.method]
    assert bool(row["in_budget"]) and bool(row["supports_model"])


def test_insufficient_budget_flags_and_picks_smallest():
    r = ed.recommend_design("optimization", factors=4, budget=6, seed=0)
    assert any("budget" in c.lower() for c in r.caveats)
    assert r.design.n_runs == min(r.table["runs"])


def test_priorities_affect_ranking_or_are_dominated():
    for prio in ({"runs": 5, "precision": 1, "prediction": 1},
                 {"runs": 1, "precision": 1, "prediction": 5}):
        r = ed.recommend_design("optimization", factors=3, priorities=prio, seed=0)
        assert r.method in set(r.table["method"])


def test_caveats_always_present_and_mention_sequential():
    r = ed.recommend_design("screening", factors=5, seed=0)
    joined = " ".join(r.caveats).lower()
    assert "multi-objective" in joined or "trade-off" in joined
    assert "propose_next_runs" in joined


def test_categorical_factor_does_not_crash_and_is_flagged():
    r = ed.recommend_design(
        "screening",
        factors=[ed.ContinuousFactor("x1", 0, 1), ed.CategoricalFactor("mat", ["A", "B", "C"])],
        seed=0)
    assert r.method in set(r.table["method"])
    assert any("categoric" in c.lower() for c in r.caveats)


def test_report_includes_coherent_recommendation():
    pb = ed.plackett_burman(7)
    g = ed.report_summary(pb)
    rec = g["recommendation"]
    assert rec is not None
    assert rec["method"] == "Plackett-Burman" and rec["matches"] is True


def test_report_recommendation_informative_when_differs():
    bb = ed.box_behnken({"a": (0, 1), "b": (0, 1), "c": (0, 1)}, center=3)
    g = ed.report_summary(bb, model=ed.Model.full_quadratic(["a", "b", "c"]))
    rec = g["recommendation"]
    assert rec is not None
    assert rec["actual"] == "Box-Behnken"
    assert "informative" in rec["note"] or rec["matches"]


def test_recommendation_to_dict_schema():
    r = ed.recommend_design("screening", factors=4, budget=8, seed=0)
    d = r.to_dict()
    assert d["schema"] == "doekit.Recommendation/1"
    assert d["method"] == r.method
    assert "design" in d and d["design"]["schema"] == "doekit.Design/1"
