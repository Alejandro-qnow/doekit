"""Disenos: screening, factoriales, superficie de respuesta y aleatorios.

Valida propiedades teóricas conocidas de cada constructor.
"""

import numpy as np
import pytest

import doekit as ed
from doekit.generation.catalog.screening import _hadamard, _is_hadamard


# --- Screening / Plackett-Burman -------------------------------------------

@pytest.mark.parametrize("k", [2, 4, 16])
def test_plackett_burman_is_orthogonal(k):
    pb = ed.plackett_burman(k)
    assert ed.is_plackett_burman(pb)
    D = pb.matrix.to_numpy().astype(float)
    n = D.shape[0]
    assert np.allclose(D.T @ D, n * np.eye(D.shape[1]))  # columnas ortogonales
    assert np.allclose(D.sum(axis=0), 0)                  # cada columna suma cero


def test_is_plackett_burman_rejects_non_orthogonal_balanced():
    # Matriz +/-1 con cada columna sumando cero pero NO ortogonal (columnas
    # iguales): un chequeo solo agregado la aceptaria; el fuerte no.
    D = np.array([[1, 1], [1, 1], [-1, -1], [-1, -1]])
    assert D.sum() == 0  # suma total cero no basta
    assert not ed.is_plackett_burman(D)  # no es ortogonal -> rechazado
    # y un PB real si pasa
    assert ed.is_plackett_burman(ed.plackett_burman(3))


def test_is_plackett_burman_rejects_non_two_level():
    assert not ed.is_plackett_burman(np.array([[2, -2], [-2, 2]]))


@pytest.mark.parametrize("order", [4, 8, 12, 16, 20, 24, 28, 32])
def test_hadamard_orders_constructible(order):
    H = _hadamard(order)
    assert H is not None and _is_hadamard(H)


def test_plackett_burman_uses_minimal_order():
    # 11 factores -> Hadamard de orden 12 (Paley I), sin dummies.
    pb = ed.plackett_burman(11)
    assert pb.metadata["order"] == 12
    assert len(pb.metadata["dummy_factors"]) == 0


def test_fold_doubles_and_mirrors():
    pb = ed.plackett_burman(4)
    folded = ed.fold(pb)
    assert folded.n_runs == 2 * pb.n_runs
    n = pb.n_runs
    assert np.allclose(folded.matrix.to_numpy()[n:], -pb.matrix.to_numpy())


# --- Factoriales -----------------------------------------------------------

def test_full_factorial_row_count():
    ff = ed.full_factorial(3)  # 3 factores a 2 niveles
    assert ff.n_runs == 8
    ff2 = ed.full_factorial({"A": [1, 2, 4], "B": ["a", "b"], "C": [1.0, -1.0]})
    assert ff2.n_runs == 12


def test_full_factorial_lazy_matches_explicit():
    lazy = list(ed.full_factorial(2, lazy=True))
    assert len(lazy) == 4
    assert lazy[0] == {"factor1": -1, "factor2": -1}


def test_fractional_factorial_is_real_fraction():
    fr = ed.fractional_factorial(3, generators=["C=AB"])
    assert fr.n_runs == 4  # 2^(3-1), NO 2^3
    assert fr.metadata["resolution"] == "III"
    assert fr.metadata["defining_relation"] == "I = ABC"
    # C = A*B en cada corrida
    m = fr.matrix
    assert np.allclose(m["factor3"], m["factor1"] * m["factor2"])


def test_fractional_factorial_alias_structure():
    fr = ed.fractional_factorial(3, generators=["C=AB"])
    aliases = fr.metadata["aliases"]
    # cada efecto principal esta aliasado con una interaccion de 2 factores
    main_classes = [c for c in aliases if any(":" not in x for x in c)]
    assert ["factor1", "factor2:factor3"] in aliases
    assert len(main_classes) == 3


def test_fractional_factorial_quarter_fraction():
    fr = ed.fractional_factorial(5, generators=["D=AB", "E=AC"])
    assert fr.n_runs == 8  # 2^(5-2)


# --- Superficie de respuesta -----------------------------------------------

def test_box_behnken_row_count():
    bb = ed.box_behnken(3, center=3)
    assert bb.n_runs == 15  # 12 aristas + 3 centro


def test_box_behnken_requires_three_factors():
    with pytest.raises(ValueError):
        ed.box_behnken(2)


def test_box_behnken_decodes_to_natural_units():
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3)
    assert bb.matrix["temp"].min() == pytest.approx(20)
    assert bb.matrix["temp"].max() == pytest.approx(80)
    assert set(np.round(bb.matrix["ph"].unique(), 6)) <= {3.0, 6.0, 9.0}


def test_central_composite_row_count_and_alpha():
    cc = ed.central_composite(3)
    assert cc.n_runs == 22
    assert cc.metadata["alpha_value"] == pytest.approx(1.82574, abs=1e-4)


def test_central_composite_rotatable_alpha():
    cc = ed.central_composite(2, alpha="rotatable")
    # alpha rotatable = (2^n)^(1/4); n=2 -> 4^0.25 = sqrt(2)
    assert cc.metadata["alpha_value"] == pytest.approx(np.sqrt(2), abs=1e-6)


# --- Aleatorios ------------------------------------------------------------

def test_latin_hypercube_covers_range():
    lhs = ed.latin_hypercube([ed.ContinuousFactor("x", 0, 10)], n=20, seed=1)
    col = lhs.matrix["x"].to_numpy()
    assert col.min() >= 0 and col.max() <= 10
    assert len(np.unique(np.floor(col))) >= 8  # buena cobertura


def test_random_design_from_scipy_dist():
    from scipy import stats
    rd = ed.random_design({"a": stats.uniform(2, 1)}, n=50, seed=0)
    assert rd.n_runs == 50
    assert rd.matrix["a"].between(2, 3).all()
