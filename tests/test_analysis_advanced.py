"""Blocked OLS, robust SE, lack-of-fit, mixed models and FitResult serialization."""

import json

import numpy as np
import pytest

import doekit as ed


def _blocked_factorial():
    """2^2 factorial in 2 blocks with a clear block shift."""
    base = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
    # two replicates of the factorial, one per block
    import pandas as pd
    m1 = base.matrix.copy()
    m2 = base.matrix.copy()
    mat = pd.concat([m1, m2], ignore_index=True)
    d = ed.Design(matrix=mat, factors=list(base.factors),
                  model=ed.Model.main_effects(["A", "B"]),
                  metadata={"kind": "FullFactorial"})
    blocks = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return ed.attach_blocks(d, blocks, name="block")


def test_attach_blocks_sets_metadata():
    d = _blocked_factorial()
    assert "block" in d.matrix.columns
    assert d.metadata["blocking"]["column"] == "block"
    assert d.metadata["blocking"]["n_blocks"] == 2


def test_fit_with_blocks_recovers_effects():
    d = _blocked_factorial()
    rng = np.random.default_rng(0)
    # true: intercept 10, A=2, B=-1.5, block1 shift +5
    y = (10
         + 2 * d.matrix["A"].to_numpy()
         - 1.5 * d.matrix["B"].to_numpy()
         + 5 * (d.matrix["block"].to_numpy() == 1).astype(float)
         + rng.normal(0, 0.05, d.n_runs))
    fit_blk = ed.fit_linear_model(d, y, blocks="block")
    fit_pool = ed.fit_linear_model(d, y, blocks=False)  # opt out of metadata blocking

    coef = dict(zip(fit_blk.names, fit_blk.coef))
    assert coef["A"] == pytest.approx(2.0, abs=0.2)
    assert coef["B"] == pytest.approx(-1.5, abs=0.2)
    # blocking should reduce residual variance vs pooled when block effect is large
    assert fit_blk.sigma2 < fit_pool.sigma2
    assert fit_blk.blocks == "block"
    assert fit_pool.blocks is None
    assert any(n.startswith("block[") for n in fit_blk.names)


def test_hc3_changes_standard_errors():
    rng = np.random.default_rng(1)
    pb = ed.plackett_burman(5)
    # heteroscedastic noise: larger at +1 of factor1
    scale = 0.05 + 0.4 * (pb.matrix["factor1"].to_numpy() > 0)
    y = (1.0
         + 2.0 * pb.matrix["factor1"].to_numpy()
         + rng.normal(0, 1, pb.n_runs) * scale)
    fit_ols = ed.fit_linear_model(pb, y, cov_type="nonrobust")
    fit_hc = ed.fit_linear_model(pb, y, cov_type="HC3")
    assert fit_hc.cov_type == "HC3"
    # SE vectors should differ under heteroscedasticity
    assert not np.allclose(fit_ols.se, fit_hc.se, rtol=1e-3, atol=1e-4)


def test_invalid_cov_type_raises():
    pb = ed.plackett_burman(4)
    y = np.zeros(pb.n_runs)
    with pytest.raises(ValueError, match="cov_type"):
        ed.fit_linear_model(pb, y, cov_type="HC9")


def test_anova_table_has_terms():
    d = _blocked_factorial()
    y = (1
         + 2 * d.matrix["A"].to_numpy()
         + d.matrix["B"].to_numpy()
         + 3 * (d.matrix["block"].to_numpy() == 1).astype(float))
    fit = ed.fit_linear_model(d, y, blocks="block")
    tab = ed.anova_table(fit)
    assert "term" in tab.columns and "F" in tab.columns
    terms = set(tab["term"])
    assert "A" in terms and "Residual" in terms


def test_lack_of_fit_with_replicates():
    # Replicated 2^2 under a main-effects model -> pure error + LOF (interaction)
    import pandas as pd
    fac = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
    mat = pd.concat([fac.matrix, fac.matrix], ignore_index=True)
    d = ed.Design(matrix=mat, factors=list(fac.factors),
                  model=ed.Model.main_effects(["A", "B"]),
                  metadata={"kind": "FullFactorial"})
    rng = np.random.default_rng(2)
    y = (5
         + 1.5 * d.matrix["A"].to_numpy()
         - 0.8 * d.matrix["B"].to_numpy()
         + 1.2 * d.matrix["A"].to_numpy() * d.matrix["B"].to_numpy()  # LOF source
         + rng.normal(0, 0.05, d.n_runs))
    lof = ed.lack_of_fit(d, y)
    sources = set(lof["source"])
    assert "lack_of_fit" in sources and "pure_error" in sources
    pe = lof.set_index("source").loc["pure_error"]
    assert pe["df"] >= 1


def test_lack_of_fit_without_replicates_raises():
    pb = ed.plackett_burman(4)
    y = np.arange(pb.n_runs, dtype=float)
    with pytest.raises(ValueError, match="replicate"):
        ed.lack_of_fit(pb, y)


def test_fit_mixed_model_random_intercept():
    rng = np.random.default_rng(3)
    # 4 groups x 6 runs, continuous covariate
    n_g, n_per = 4, 6
    n = n_g * n_per
    group = np.repeat(np.arange(n_g), n_per)
    x = rng.uniform(-1, 1, n)
    # random intercepts + fixed slope
    re = rng.normal(0, 1.5, n_g)
    y = 2.0 + 1.2 * x + re[group] + rng.normal(0, 0.3, n)
    import pandas as pd
    d = ed.Design(
        matrix=pd.DataFrame({"x": x, "batch": group}),
        factors=[ed.ContinuousFactor("x", -1, 1)],
        model=ed.Model.parse("0 ~ x"),
        metadata={"kind": "RandomDesign"},
    )
    fit = ed.fit_mixed_model(d, y, groups="batch")
    assert fit.n_groups == n_g
    assert fit.method == "reml"
    assert fit.groups == "batch"
    coef = dict(zip(fit.names, fit.coef))
    # slope is the stable estimand; intercept absorbs RE mean
    assert coef["x"] == pytest.approx(1.2, abs=0.4)
    assert fit.converged
    assert "Intercept" in fit.re_var or len(fit.re_var) >= 1
    assert np.isfinite(fit.sigma2) and fit.sigma2 > 0


def test_fit_mixed_model_rejects_single_group():
    pb = ed.plackett_burman(4)
    y = np.ones(pb.n_runs)
    with pytest.raises(ValueError, match="at least 2"):
        ed.fit_mixed_model(pb, y, groups=np.zeros(pb.n_runs))


def test_fit_result_to_dict_roundtrip():
    pb = ed.plackett_burman(5)
    rng = np.random.default_rng(4)
    y = rng.normal(size=pb.n_runs)
    fit = ed.fit_linear_model(pb, y, cov_type="HC1")
    d = fit.to_dict()
    assert d["schema"] == "doekit.FitResult/1"
    assert d["kind"] == "ols"
    assert d["cov_type"] == "HC1"
    # JSON serializable
    json.dumps(d)
    fit2 = ed.FitResult.from_dict(d)
    assert fit2.names == fit.names
    assert np.allclose(fit2.coef, fit.coef)
    assert fit2.cov_type == "HC1"


def test_mixed_fit_to_dict_roundtrip():
    rng = np.random.default_rng(5)
    n_g, n_per = 3, 5
    n = n_g * n_per
    group = np.repeat(np.arange(n_g), n_per)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(0, 0.5, n_g)[group] + rng.normal(0, 0.2, n)
    import pandas as pd
    d = ed.Design(
        matrix=pd.DataFrame({"x": x, "g": group}),
        factors=[ed.ContinuousFactor("x", float(x.min()), float(x.max()))],
        model=ed.Model.parse("0 ~ x"),
    )
    fit = ed.fit_mixed_model(d, y, groups="g")
    payload = fit.to_dict()
    assert payload["schema"] == "doekit.MixedFitResult/1"
    json.dumps(payload)
    fit2 = ed.MixedFitResult.from_dict(payload)
    assert fit2.n_groups == fit.n_groups
    assert np.allclose(fit2.coef, fit.coef)


def test_design_evaluation_to_dict():
    bb = ed.box_behnken({"a": (0, 1), "b": (0, 1), "c": (0, 1)}, center=3)
    ev = ed.evaluate(bb, n_region=500, seed=0)
    d = ev.to_dict()
    assert d["schema"] == "doekit.DesignEvaluation/1"
    assert "efficiencies" in d and "D_efficiency" in d["efficiencies"]
    json.dumps(d)


def test_report_summary_includes_fit_and_anova():
    d = _blocked_factorial()
    y = (1 + d.matrix["A"].to_numpy() + d.matrix["B"].to_numpy()
         + (d.matrix["block"].to_numpy() == 1).astype(float))
    g = ed.report_summary(d, response=y, blocks="block")
    assert g["fit"] is not None and g["fit"]["schema"] == "doekit.FitResult/1"
    assert g["anova"] is not None and len(g["anova"]) >= 1
    assert g["fit"]["blocks"] == "block"


def test_metadata_blocking_auto_used():
    d = _blocked_factorial()
    y = np.arange(d.n_runs, dtype=float)
    fit = ed.fit_linear_model(d, y)  # blocks=None but metadata has blocking
    assert fit.blocks == "block"
