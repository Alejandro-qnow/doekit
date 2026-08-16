"""
Tests reales para el servidor FastMCP de doekit.
"""

import sys
import os
import importlib.util
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))


def _load_server_module():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    server_path = os.path.join(root, 'mcp', 'doekit_fastmcp_server.py')
    spec = importlib.util.spec_from_file_location('doekit_fastmcp_server', server_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestFastMCPServer:
    def test_recommend_design_tool(self):
        server = _load_server_module()
        out = server.recommend_design(
            goal='optimization',
            factors={'X1': [-1, 1], 'X2': [-1, 1], 'X3': [-1, 1]},
            budget=20,
            model_order='quadratic',
        )

        assert out['n_runs'] > 0
        assert isinstance(out['method'], str)
        assert isinstance(out['factor_names'], list)
        assert 'rationale' in out

    def test_evaluate_design_tool(self):
        server = _load_server_module()
        out = server.evaluate_design(
            design_type='central_composite',
            factors={'X1': [-1, 1], 'X2': [-1, 1]},
            model_order='quadratic',
        )

        assert out['n_runs'] > 0
        assert out['d_efficiency'] is None or out['d_efficiency'] >= 0.0
        assert out['mean_power'] is None or 0.0 <= out['mean_power'] <= 1.0

    def test_propose_next_wave_tool(self):
        server = _load_server_module()
        out = server.propose_next_wave(
            factors={'X1': [-1, 1], 'X2': [-1, 1]},
            model_order='quadratic',
            n_add=2,
            seed=42,
            sigma=1.0,
        )

        assert out['n_added'] == 2
        assert isinstance(out['worth_it'], bool)
        assert isinstance(out['delta'], dict)
        assert 'D_efficiency' in out['delta']

    def test_invalid_design_type_raises(self):
        server = _load_server_module()
        with pytest.raises(ValueError, match="design_type"):
            server.evaluate_design(
                design_type='invalid_design_type',
                factors={'X1': [-1, 1], 'X2': [-1, 1]},
                model_order='quadratic',
            )
