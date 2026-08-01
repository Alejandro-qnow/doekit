"""Nucleo: factores (codificacion), modelo (matriz) y criterios."""

import numpy as np
import pandas as pd
import pytest

import doekit as ed
from doekit import criteria


def test_continuous_factor_encode_decode_roundtrip():
    f = ed.ContinuousFactor("temp", 20.0, 80.0)
    assert f.encode(20.0) == pytest.approx(-1.0)
    assert f.encode(80.0) == pytest.approx(1.0)
    assert f.encode(50.0) == pytest.approx(0.0)
    vals = np.array([20.0, 50.0, 80.0])
    assert np.allclose(f.decode(f.encode(vals)), vals)


def test_discrete_factor_snaps_to_nearest_level():
    f = ed.DiscreteFactor("n", [1, 2, 4])
    # -1 -> 1, +1 -> 4; el centro decodifica al nivel valido mas cercano
    assert f.decode(-1.0) == 1
    assert f.decode(1.0) == 4
    assert f.decode(0.0) in (2,)


def test_categorical_factor_encode_decode():
    f = ed.CategoricalFactor("mat", ["a", "b", "c"])
    assert f.encode("b") == 1
    assert f.decode(2) == "c"


def test_model_parse_intercept_and_terms():
    m = ed.Model.parse("y ~ x1 + x2 + x1:x2 + x3^2")
    df = pd.DataFrame({"x1": [1.0, -1.0], "x2": [1.0, 1.0], "x3": [2.0, 3.0]})
    X = m.matrix(df)
    names = m.column_names(df)
    assert names[0] == "(Intercept)"
    assert "x1:x2" in names
    assert "x3^2" in names
    # columna de interaccion = producto
    inter = X[:, names.index("x1:x2")]
    assert np.allclose(inter, df["x1"] * df["x2"])
    # potencia
    pw = X[:, names.index("x3^2")]
    assert np.allclose(pw, df["x3"] ** 2)


def test_model_no_intercept_when_suppressed():
    m = ed.Model.parse("0 ~ -1 + x1 + x2")
    df = pd.DataFrame({"x1": [1.0], "x2": [1.0]})
    assert "(Intercept)" not in m.column_names(df)


def test_d_criterion_orthogonal_design_is_one():
    # matriz +/-1 ortogonal: det((X'X)/N)^(1/p) = 1
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    assert criteria.d_criterion(X) == pytest.approx(1.0, abs=1e-6)


def test_criteria_all_positive_for_full_rank():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    for name, fn in criteria.CRITERIA.items():
        assert fn(X) > 0, name
