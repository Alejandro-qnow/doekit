"""Surrogate models: predict (mean, std), sigma growth, LOO calibration."""

import numpy as np
import pytest

import doekit as ed
from doekit.assessment.surrogate import (fit_surrogate, OLSSurrogate,
                                         encode_features, loo_calibration)
from doekit.assessment.surrogate.base import _sklearn_available


def _rsm_design():
    cols = list(ed.central_composite(2).matrix.columns)
    d = ed.central_composite(2)
    facs = [ed.ContinuousFactor(cols[0], -1, 1),
            ed.ContinuousFactor(cols[1], -1, 1)]
    d = ed.Design(matrix=d.matrix, factors=facs,
                  model=ed.Model.full_quadratic(cols))
    return d, cols


def _wiggly_response(d, cols, seed=0):
    rng = np.random.default_rng(seed)
    X = d.matrix[cols].to_numpy(dtype=float)
    return np.sin(1.5 * X[:, 0]) + np.cos(1.2 * X[:, 1]) + 0.03 * rng.standard_normal(len(X))


def test_ols_surrogate_predict_shapes_and_no_sklearn_needed():
    d, cols = _rsm_design()
    y = _wiggly_response(d, cols)
    sur = OLSSurrogate.fit(d, y)
    mean, std = sur.predict(d)
    assert mean.shape == (d.n_runs,)
    assert std.shape == (d.n_runs,)
    assert np.all(std >= 0)
    # array input with factor-order columns also works
    m2, s2 = sur.predict(np.zeros((1, 2)))
    assert m2.shape == (1,) and s2.shape == (1,)


def test_ols_std_grows_away_from_data():
    d, cols = _rsm_design()
    # underspecified model (main effects only) leaves leverage structure
    d = d.replace(model=ed.Model.main_effects(cols))
    y = _wiggly_response(d, cols)
    sur = OLSSurrogate.fit(d, y)
    _, s_center = sur.predict(np.array([[0.0, 0.0]]))
    _, s_far = sur.predict(np.array([[5.0, 5.0]]))
    assert s_far[0] > s_center[0]


def test_fit_surrogate_auto_and_ols_kind():
    d, cols = _rsm_design()
    y = _wiggly_response(d, cols)
    sur = fit_surrogate(d, y, kind="ols")
    assert isinstance(sur, OLSSurrogate)
    auto = fit_surrogate(d, y)  # gp if sklearn present, else ols
    expected = "GPSurrogate" if _sklearn_available() else "OLSSurrogate"
    assert type(auto).__name__ == expected


def test_encode_features_one_hot_categorical():
    import pandas as pd
    facs = [ed.ContinuousFactor("x", -1, 1),
            ed.CategoricalFactor("g", ["a", "b", "c"])]
    frame = pd.DataFrame({"x": [0.0, 1.0], "g": ["a", "c"]})
    feats = encode_features(facs, frame)
    # 1 continuous + 3 one-hot columns
    assert feats.shape == (2, 4)
    assert set(np.unique(feats[:, 1:])) <= {0.0, 1.0}


def test_loo_calibration_reports_coverage():
    d, cols = _rsm_design()
    y = _wiggly_response(d, cols)
    sur = OLSSurrogate.fit(d, y)
    cal = sur.calibration()
    assert set(cal["coverage"]) == {0.5, 0.8, 0.95}
    assert cal["n"] == d.n_runs
    for lvl, cov in cal["coverage"].items():
        assert 0.0 <= cov <= 1.0


def test_loo_calibration_tiny_sample_is_nan():
    import pandas as pd
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    out = loo_calibration(lambda f, y: OLSSurrogate.fit(f, y), frame,
                          np.array([1.0, 2.0, 3.0]))
    assert np.isnan(out["coverage"][0.5])


@pytest.mark.skipif(not _sklearn_available(), reason="scikit-learn not installed")
def test_gp_surrogate_std_grows_and_predict_shapes():
    d, cols = _rsm_design()
    y = _wiggly_response(d, cols)
    sur = fit_surrogate(d, y, kind="gp", seed=1)
    mean, std = sur.predict(d)
    assert mean.shape == (d.n_runs,) and std.shape == (d.n_runs,)
    _, s_center = sur.predict(np.array([[0.0, 0.0]]))
    _, s_far = sur.predict(np.array([[6.0, 6.0]]))
    assert s_far[0] > s_center[0]


@pytest.mark.skipif(not _sklearn_available(), reason="scikit-learn not installed")
def test_gp_prior_mean_is_ols_surface():
    # With a perfectly quadratic truth, the OLS prior already fits; the GP
    # residual should be tiny, so predictions stay close to the OLS surface.
    d, cols = _rsm_design()
    X = d.matrix[cols].to_numpy(dtype=float)
    y = 2.0 - (X[:, 0] - 0.2) ** 2 - (X[:, 1] + 0.1) ** 2
    ols = OLSSurrogate.fit(d, y)
    gp = fit_surrogate(d, y, kind="gp", seed=0)
    grid = np.array([[0.2, -0.1], [0.5, 0.5], [-0.7, 0.3]])
    m_ols, _ = ols.predict(grid)
    m_gp, _ = gp.predict(grid)
    assert np.allclose(m_ols, m_gp, atol=0.1)
