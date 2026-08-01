"""Capa de evaluacion / benchmarking: eficiencias, FDS, power, alias, VIF."""

import numpy as np
import pytest

import doekit as ed


def test_orthogonal_design_is_100_percent_efficient():
    # Un factorial completo 2^3 (modelo de efectos principales) es ortogonal:
    # D- y A-eficiencia ~ 100%.
    ff = ed.full_factorial(3)
    ff.model = ed.Model.main_effects(["factor1", "factor2", "factor3"])
    eff = ed.efficiencies(ff, seed=0)
    assert eff["D_efficiency"] == pytest.approx(100.0, abs=1e-6)
    assert eff["A_efficiency"] == pytest.approx(100.0, abs=1e-6)
    assert eff["dof"] == 8 - 4


def test_plackett_burman_main_effects_efficient():
    pb = ed.plackett_burman(6)
    pb.model = ed.Model.main_effects(pb.factor_names)
    eff = ed.efficiencies(pb, seed=1)
    assert eff["D_efficiency"] == pytest.approx(100.0, abs=1e-6)


def test_efficiency_penalizes_correlated_design():
    # Diseno con dos factores casi colineales -> D-eficiencia baja y VIF alto.
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-1, 1, 20)
    x2 = x1 + rng.normal(0, 0.02, 20)   # casi identico a x1
    import pandas as pd
    df = pd.DataFrame({"x1": x1, "x2": x2})
    d = ed.Design(matrix=df, model=ed.Model.parse("0 ~ x1 + x2"))
    eff = ed.efficiencies(d, seed=0)
    assert eff["D_efficiency"] < 60.0
    v = ed.vif(d)
    assert v.max() > 10.0


def test_g_efficiency_bounded_and_spv_ge_p():
    # Teorema de equivalencia general: max SPV >= p, luego G-eff <= 100%.
    cc = ed.central_composite(2)
    eff = ed.efficiencies(cc, n_region=5000, seed=2)
    assert 0.0 < eff["G_efficiency"] <= 100.0 + 1e-6
    assert eff["spv_max"] >= eff["n_params"] - 1e-6


def test_power_increases_with_replication():
    base = ed.full_factorial(2)
    base.model = ed.Model.parse("y ~ x1 + x2")  # ojo: usa nombres reales
    # renombrar a factor1/factor2
    m = ed.Model.main_effects(["factor1", "factor2"])
    p_small = ed.power_analysis(base, model=m, effect_size=1.0, sigma=1.0)
    # replicar el diseno 4x baja el error estandar -> sube la potencia
    import pandas as pd
    big = ed.Design(matrix=pd.concat([base.matrix] * 4, ignore_index=True), model=m)
    p_big = ed.power_analysis(big, model=m, effect_size=1.0, sigma=1.0)
    assert (p_big.to_numpy() >= p_small.to_numpy() - 1e-9).all()
    assert p_big["factor1"] > p_small["factor1"]


def test_alias_matrix_captures_confounding_in_resolution_iii():
    # En 2^(3-1) con C=AB, el efecto principal C esta aliasado con AB.
    fr = ed.fractional_factorial(3, generators=["C=AB"])
    fr.model = ed.Model.main_effects(["factor1", "factor2", "factor3"], intercept=True)
    A = ed.alias_matrix(fr)
    # factor3 debe estar perfectamente aliasado (|alias|=1) con factor1:factor2
    assert abs(A.loc["factor3", "factor1:factor2"]) == pytest.approx(1.0, abs=1e-9)


def test_fds_data_monotone_and_fraction_range():
    cc = ed.central_composite(3)
    fds = ed.fds_data(cc, n_region=3000, seed=3)
    assert fds["spv"].is_monotonic_increasing
    assert fds["fraction"].min() > 0 and fds["fraction"].max() < 1


def test_evaluate_report_smoke():
    bb = ed.box_behnken(3, center=3)
    rep = ed.evaluate(bb, n_region=3000, seed=4)
    assert rep.n_runs == bb.n_runs
    assert 0 <= rep.d_efficiency <= 100 + 1e-6
    assert "D-efficiency" in rep.summary()
    assert len(rep.power) == rep.n_params
