"""Definitive Screening Designs y matrices de conferencia de Paley."""

import numpy as np
import pytest

import doekit as ed
from doekit.designs.definitive import _paley_conference, _next_conference_order


@pytest.mark.parametrize("order", [4, 6, 8, 12, 14, 18, 20])
def test_paley_conference_property(order):
    C = _paley_conference(order)
    assert C is not None
    assert np.allclose(np.diag(C), 0)                       # diagonal cero
    off = C[~np.eye(order, dtype=bool)]
    assert np.all(np.isin(off, (-1.0, 1.0)))                # +/-1 fuera de diag
    assert np.allclose(C.T @ C, (order - 1) * np.eye(order))  # C^T C = (m-1) I


def test_dsd_run_count_and_levels():
    dsd = ed.definitive_screening(6)          # orden de conferencia 6
    assert dsd.metadata["conference_order"] == 6
    assert dsd.n_runs == 2 * 6 + 1            # 13 corridas
    assert set(np.unique(dsd.matrix.to_numpy())) <= {-1.0, 0.0, 1.0}


def test_dsd_main_effects_orthogonal_to_second_order():
    # Propiedad definitoria del DSD: efectos principales ortogonales entre si
    # y ortogonales a cuadraticos e interacciones de dos factores.
    dsd = ed.definitive_screening(6)
    X = dsd.matrix.to_numpy().astype(float)   # ya en +/-1/0 (codificado)
    n, m = X.shape
    # main effects mutuamente ortogonales
    G = X.T @ X
    assert np.allclose(G - np.diag(np.diag(G)), 0)
    # main effect_i ortogonal a cuadratico_j (columna^2) y a interaccion_j*k
    quad = X ** 2
    assert np.allclose(X.T @ quad, 0, atol=1e-9)
    for j in range(m):
        for k in range(j + 1, m):
            inter = X[:, j] * X[:, k]
            assert np.allclose(X.T @ inter, 0, atol=1e-9)


def test_dsd_decodes_to_natural_units():
    dsd = ed.definitive_screening({"temp": (20, 80), "ph": (3, 9),
                                   "conc": (0.1, 0.5), "t": (10, 30)})
    # 4 factores -> conferencia de Paley orden 4 (q=3) -> DSD minimo, 9 corridas
    assert dsd.metadata["conference_order"] == 4
    assert dsd.metadata["phantom_factors"] == 0
    assert dsd.n_runs == 9
    assert dsd.matrix["temp"].min() == pytest.approx(20)
    assert dsd.matrix["temp"].max() == pytest.approx(80)
    # el nivel central (0 codificado) mapea al punto medio natural
    assert 50.0 in set(np.round(dsd.matrix["temp"].unique(), 6))


def test_dsd_requires_two_factors():
    with pytest.raises(ValueError):
        ed.definitive_screening(1)


def test_next_conference_order_minimal():
    assert _next_conference_order(5)[0] == 6
    assert _next_conference_order(7)[0] == 8
    assert _next_conference_order(10)[0] == 12
