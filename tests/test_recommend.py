"""Asesor de diseno experimental: reglas + evaluacion."""

import numpy as np
import pytest

import doekit as ed


def test_screening_low_budget_recommends_plackett_burman():
    r = ed.recommend_design("screening", factors=7, budget=8, seed=0)
    assert r.method == "Plackett-Burman"
    assert r.design.n_runs <= 8
    # la tabla lista alternativas y el metodo elegido esta en ella
    assert r.method in set(r.table["metodo"])


def test_constrained_forces_optimal():
    r = ed.recommend_design("optimization", factors=3, constrained=True, seed=0)
    assert r.method == "D-optimo"
    assert set(r.table["metodo"]) == {"D-optimo"}


def test_winner_is_feasible_and_supports_model():
    r = ed.recommend_design("optimization", factors=3, seed=0)
    row = r.table.set_index("metodo").loc[r.method]
    assert bool(row["en_presupuesto"]) and bool(row["soporta_modelo"])


def test_insufficient_budget_flags_and_picks_smallest():
    r = ed.recommend_design("optimization", factors=4, budget=6, seed=0)
    # ningun diseno cabe en 6 corridas -> salvedad al frente y se elige el menor
    assert any("presupuesto" in c.lower() for c in r.caveats)
    assert r.design.n_runs == min(r.table["corridas"])


def test_priorities_affect_ranking_or_are_dominated():
    # Con pesos muy distintos el resultado es valido (puede o no cambiar si un
    # diseno domina). Verificamos que corre y devuelve un metodo de la tabla.
    for prio in ({"runs": 5, "precision": 1, "prediction": 1},
                 {"runs": 1, "precision": 1, "prediction": 5}):
        r = ed.recommend_design("optimization", factors=3, priorities=prio, seed=0)
        assert r.method in set(r.table["metodo"])


def test_caveats_always_present_and_mention_catalog_gaps():
    r = ed.recommend_design("screening", factors=5, seed=0)
    joined = " ".join(r.caveats).lower()
    assert "mezcla" in joined and "split-plot" in joined     # huecos del catalogo
    assert "multiobjetivo" in joined or "trade-off" in joined


def test_categorical_factor_does_not_crash_and_is_flagged():
    r = ed.recommend_design(
        "screening",
        factors=[ed.ContinuousFactor("x1", 0, 1), ed.CategoricalFactor("mat", ["A", "B", "C"])],
        seed=0)
    assert r.method in set(r.table["metodo"])
    assert any("categoric" in c.lower() for c in r.caveats)


def test_report_includes_coherent_recommendation():
    # PB ejecutado -> el reporte recomienda PB (coincide)
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
    # ejecutado Box-Behnken; el asesor puede sugerir otro, siempre framing informativo
    assert rec["actual"] == "Box-Behnken"
    assert "informativo" in rec["note"] or rec["matches"]
