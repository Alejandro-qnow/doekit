"""MCP adapter tool logic (fastmcp-free smoke tests) — M5."""

import json

import numpy as np
import pytest

import doekit as ed
from doekit.adapters import mcp


FACTORS = {"temp": [20, 80], "ph": [3, 9]}


def _ccd_response(seed=0):
    d = ed.central_composite(_ccd_factors())
    rng = np.random.default_rng(seed)
    X = d.matrix.to_numpy(dtype=float)
    return list(5 - (X[:, 0] ** 2 + X[:, 1] ** 2) + 0.02 * rng.standard_normal(len(X)))


def _ccd_factors():
    return {k: tuple(v) for k, v in FACTORS.items()}


def test_tool_recommend_returns_interpretation():
    out = mcp.tool_recommend("optimization", FACTORS, budget=20)
    assert out["method"]
    assert out["interpretation"]["schema"] == "doekit.Interpretation/1"
    assert out["context_addition"]
    json.dumps(out)


def test_tool_evaluate_ccd():
    out = mcp.tool_evaluate("central_composite", FACTORS)
    assert out["n_runs"] > 0
    assert "D_efficiency" in out["efficiencies"]
    json.dumps(out)


def test_tool_evaluate_rejects_unknown_design():
    with pytest.raises(ValueError):
        mcp.tool_evaluate("latin_hypercube", FACTORS)


def test_tool_propose_and_decide_learn():
    y = _ccd_response()
    out = mcp.tool_propose_and_decide("central_composite", FACTORS, y, n_add=3,
                                      budget=len(y) + 6)
    assert out["intent"] == "learn"
    assert out["n_added"] == 3
    assert out["decision"]["schema"] == "doekit.Decision/1"
    assert out["decision"]["action"] in {"augment", "refine", "stop", "redesign"}
    assert "calibration" not in out  # learn carries no surrogate to audit
    json.dumps(out)


def test_tool_propose_and_decide_optimize():
    y = _ccd_response(seed=1)
    out = mcp.tool_propose_and_decide("central_composite", FACTORS, y, n_add=2,
                                      intent="optimize", acquisition="ei",
                                      budget=len(y) + 6)
    assert out["intent"] == "optimize"
    assert out["interpretation"]["facts"]["intent"] == "optimize"
    assert "decision" in out
    # surrogate="auto" → GP when doekit[bo] is installed, else OLS; either way the
    # calibration audit rides along so an agent can vet sigma(x) before best_so_far.
    cal = out["calibration"]
    assert cal["kind"] in {"GPSurrogate", "OLSSurrogate"}
    assert "coverage" in cal and "rmse_standardized" in cal
    json.dumps(out)


def test_tool_propose_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mcp.tool_propose_and_decide("central_composite", FACTORS, [1.0, 2.0])


def test_build_server_requires_fastmcp():
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="doekit\\[mcp\\]"):
            mcp.build_server()
    else:  # pragma: no cover - only when fastmcp is installed
        server = mcp.build_server()
        assert server is not None
