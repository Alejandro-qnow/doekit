"""Regresiones de fallos silenciosos en power / VIF / FDS."""

import numpy as np
import pandas as pd
import pytest

from doekit import (
    Design,
    Main,
    Model,
    Interaction,
    box_behnken,
    fds_data,
    power_analysis,
    vif,
)


def _orthogonal_2_3(replicas: int = 8) -> Design:
    base = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    return Design(matrix=pd.DataFrame(base * replicas, columns=["x", "y", "z"]))


@pytest.mark.parametrize("sigma", [0.8, 0.7, 0.6, 0.4, 0.3])
def test_power_no_nan_on_large_orthogonal_design(sigma):
    """nct lower tail used to return NaN erratically for large non-centrality."""
    d = _orthogonal_2_3()
    mod = Model([Main("x"), Main("y"), Main("z")])
    pw = power_analysis(d, mod, effect_size=1.0, sigma=sigma)
    assert not pw.isna().any()
    assert pw.min() > 0.999


def test_power_is_one_when_standard_error_vanishes():
    d = _orthogonal_2_3()
    mod = Model([Main("x"), Main("y"), Main("z")])
    pw = power_analysis(d, mod, effect_size=1.0, sigma=1e-300)
    assert not pw.isna().any()
    assert pw.max() == pytest.approx(1.0)


def test_power_null_effect_equals_alpha():
    d = _orthogonal_2_3()
    mod = Model([Main("x"), Main("y"), Main("z")])
    pw = power_analysis(d, mod, effect_size=0.0, sigma=1.0, alpha=0.05)
    assert pw.min() == pytest.approx(0.05, abs=1e-9)


def test_power_monotone_nonincreasing_in_sigma():
    d = _orthogonal_2_3()
    mod = Model([Main("x"), Main("y"), Main("z")])
    sigmas = np.arange(0.05, 3.01, 0.05)
    mins = [float(power_analysis(d, mod, effect_size=1.0, sigma=float(s)).min())
            for s in sigmas]
    assert np.all(np.isfinite(mins))
    assert np.all(np.diff(mins) <= 1e-9)


def test_power_box_behnken_no_nan_at_high_ncp():
    bb = box_behnken(3, center=3)
    for sigma in (0.3, 0.2):
        pw = power_analysis(bb, effect_size=1.0, sigma=sigma)
        assert not pw.isna().any(), f"NaN at sigma={sigma}: {pw.to_dict()}"
        assert ((pw >= 0.0) & (pw <= 1.0)).all()


def test_vif_perfect_collinearity_is_infinite():
    """pinv(corr) used to report VIF=0.25 — looks orthogonal, is catastrophic."""
    mat = pd.DataFrame(
        {"a": [-1, 1, -1, 1] * 2, "b": [-1, 1, -1, 1] * 2, "c": [-1, -1, 1, 1] * 2}
    )
    d = Design(matrix=mat)
    mod = Model([Main("a"), Main("b"), Main("c")])
    v = vif(d, mod)
    assert np.isinf(v["a"])
    assert np.isinf(v["b"])
    assert v["c"] == pytest.approx(1.0, abs=1e-9)


def test_vif_orthogonal_is_one():
    base = [(a, b) for a in (-1, 1) for b in (-1, 1)]
    d = Design(matrix=pd.DataFrame(base * 2, columns=["x", "y"]))
    v = vif(d, Model([Main("x"), Main("y")]))
    assert v.min() == pytest.approx(1.0, abs=1e-9)
    assert v.max() == pytest.approx(1.0, abs=1e-9)


def test_fds_undefined_when_rank_deficient():
    base = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    d = Design(matrix=pd.DataFrame(base[:5], columns=["x0", "x1", "x2"]))
    mod = Model(
        [
            Main("x0"),
            Main("x1"),
            Main("x2"),
            Interaction(("x0", "x1")),
            Interaction(("x0", "x2")),
            Interaction(("x1", "x2")),
        ]
    )
    fds = fds_data(d, mod, n_region=200, seed=0)
    assert fds["spv"].isna().all()
