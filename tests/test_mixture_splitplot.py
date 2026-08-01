"""Mixture (simplex) and split-plot generation + advisor wiring (0.6)."""

import json

import numpy as np
import pytest

import doekit as ed


def test_simplex_lattice_rows_sum_to_one():
    d = ed.simplex_lattice(3, degree=2)
    assert d.metadata["kind"] == "SimplexLattice"
    assert d.metadata["region"] == "simplex"
    s = d.matrix.to_numpy().sum(axis=1)
    assert np.allclose(s, 1.0)
    assert d.n_runs == 6  # {3,2} lattice


def test_simplex_centroid_count():
    d = ed.simplex_centroid(3)
    # 2^3 - 1 = 7
    assert d.n_runs == 7
    assert np.allclose(d.matrix.to_numpy().sum(axis=1), 1.0)


def test_scheffe_model_no_intercept():
    m = ed.Model.scheffe_quadratic(["a", "b", "c"])
    assert not any(t.label() == "(Intercept)" for t in m.terms)
    d = ed.simplex_lattice([ed.MixtureFactor("a"), ed.MixtureFactor("b"),
                            ed.MixtureFactor("c")], degree=2)
    X = m.matrix(d.matrix)
    assert X.shape[1] == 6  # 3 linear + 3 interactions


def test_mixture_evaluate_uses_simplex_region():
    d = ed.simplex_lattice(3, degree=2)
    ev = ed.evaluate(d, n_region=500, seed=0)
    assert not ev.efficiencies["rank_deficient"]
    assert np.isfinite(ev.d_efficiency)
    # region samples should sum to 1
    from doekit.domain.region import region_from_design
    reg = region_from_design(d, d.model)
    assert type(reg).__name__ == "SimplexRegion"
    sample = reg.sample(20, rng=np.random.default_rng(0))
    assert np.allclose(sample.to_numpy().sum(axis=1), 1.0, atol=1e-6)


def test_mixture_factor_roundtrip():
    f = ed.MixtureFactor("oil", 0.1, 0.8)
    f2 = ed.factor_from_dict(f.to_dict())
    assert isinstance(f2, ed.MixtureFactor)
    assert f2.lower == 0.1 and f2.upper == 0.8


def test_recommend_mixture_shortlist():
    facs = [ed.MixtureFactor(f"x{i}") for i in range(3)]
    r = ed.recommend_design("optimization", facs, seed=0, n_region=800)
    assert r.scenario["mixture"] is True
    assert set(r.table["method"]) <= {"Simplex lattice", "Simplex centroid"}
    assert r.method in set(r.table["method"])
    joined = " ".join(r.caveats).lower()
    assert "mixture" in joined or "scheffé" in joined or "scheffe" in joined
    # no longer "outside catalog" for mixture
    assert "outside the current catalog: mixture" not in joined


def test_split_plot_structure():
    d = ed.split_plot_design(
        whole_plot=[ed.ContinuousFactor("temp", 20, 80)],
        subplot=[ed.ContinuousFactor("time", 1, 5),
                 ed.ContinuousFactor("cat", 0, 1)],
        whole_plot_reps=2,
    )
    assert d.metadata["kind"] == "SplitPlot"
    assert "whole_plot_id" in d.matrix.columns
    assert d.metadata["n_whole_plots"] == 4  # 2 levels * 2 reps
    # each plot has full subplot factorial (2^2 = 4)
    assert d.n_runs == 4 * 4


def test_split_plot_mixed_analysis():
    d = ed.split_plot_design(
        whole_plot={"batch": (-1, 1)},
        subplot={"x": (-1, 1)},
        whole_plot_reps=3,
        seed=0,
    )
    rng = np.random.default_rng(1)
    # response with WP random effect
    wp = d.matrix["whole_plot_id"].to_numpy()
    re = rng.normal(0, 1.0, int(wp.max()) + 1)
    y = (1.0 + 0.5 * d.matrix["x"].to_numpy() + re[wp]
         + rng.normal(0, 0.2, d.n_runs))
    fit = ed.fit_mixed_model(d, y, groups="whole_plot_id",
                             model=ed.Model.parse("0 ~ x"))
    assert fit.n_groups == d.metadata["n_whole_plots"]
    assert fit.converged


def test_recommend_split_plot():
    r = ed.recommend_design(
        "optimization",
        factors=[ed.ContinuousFactor("oven", 100, 200),
                 ed.ContinuousFactor("time", 1, 10),
                 ed.ContinuousFactor("speed", 0, 1)],
        hard_to_change=["oven"],
        seed=0,
        n_region=600,
    )
    assert r.scenario["split_plot"] is True
    assert "Split-plot" in set(r.table["method"])
    assert any("split-plot" in c.lower() for c in r.caveats)


def test_constraints_irregular_forces_optimal():
    r = ed.recommend_design(
        "optimization", factors=3,
        constraints=ed.Constraints(irregular=True),
        seed=0,
    )
    assert set(r.table["method"]) == {"D-optimal"}


def test_constrained_bool_deprecated():
    with pytest.warns(DeprecationWarning, match="constrained"):
        r = ed.recommend_design("optimization", factors=3, constrained=True, seed=0)
    assert r.method == "D-optimal"


def test_constraints_to_dict():
    c = ed.Constraints(mixture=True, hard_to_change=["a"], run_cost=2.0)
    d = c.to_dict()
    assert d["mixture"] is True and d["hard_to_change"] == ["a"]
    json.dumps(d)
