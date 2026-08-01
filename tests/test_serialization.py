"""Serializacion to_dict/from_dict (base para el MCP)."""

import json

import numpy as np
import pytest

import doekit as ed
from doekit.domain.design import Design


@pytest.mark.parametrize("build", [
    lambda: ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3),
    lambda: ed.plackett_burman(4),
    lambda: ed.definitive_screening(5),
    lambda: ed.fractional_factorial(3, generators=["C=AB"]),
    lambda: ed.central_composite(2),
])
def test_design_roundtrip(build):
    d = build()
    d2 = Design.from_json(d.to_json())
    assert list(d2.matrix.columns) == list(d.matrix.columns)
    num = d.matrix.select_dtypes("number").to_numpy()
    assert np.allclose(d2.matrix.select_dtypes("number").to_numpy(), num)
    assert len(d2.factors) == len(d.factors or [])
    assert (d2.model is None) == (d.model is None)
    if d.model is not None:
        assert repr(d2.model) == repr(d.model)


def test_to_dict_is_json_serializable():
    # metadata con tipos numpy (criteria del optimal_design) debe ser JSON-safe
    import pandas as pd
    cand = ed.Design(matrix=pd.DataFrame(np.random.default_rng(0).uniform(-1, 1, (60, 2)),
                                         columns=["x1", "x2"]),
                     model=ed.Model.parse("0 ~ x1 + x2 + x1:x2"))
    opt = ed.optimal_design(cand, n_runs=8, seed=1)
    s = json.dumps(opt.to_dict())          # no debe lanzar
    assert "criteria" in s


def test_factor_roundtrip_all_types():
    facs = [ed.ContinuousFactor("a", 0, 10),
            ed.DiscreteFactor("b", [1, 2, 4]),
            ed.CategoricalFactor("c", ["x", "y", "z"])]
    for f in facs:
        f2 = ed.factor_from_dict(f.to_dict())
        assert type(f2) is type(f)
        assert f2.name == f.name


def test_model_roundtrip():
    m = ed.Model.parse("y ~ x1 + x2 + x1:x2 + x3^2")
    m2 = ed.Model.from_dict(m.to_dict())
    assert repr(m2) == repr(m)
    assert m2.response == "y"
