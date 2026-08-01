"""Diseno optimo (KL/Fedorov) y capa de analisis."""

import numpy as np
import pytest

import doekit as ed


def test_kl_exchange_d_optimal_known_case():
    # Candidate set 3^5, modelo con un cuadratico, 11 corridas: D ~ 0.7305
    cand = ed.full_factorial({f"factor{i+1}": [-1, 0, 1] for i in range(5)})
    cand.model = ed.Model.parse(
        "0 ~ factor1 + factor2 + factor3 + factor4 + factor5 + factor3^2"
    )
    opt = ed.optimal_design(cand, n_runs=11, criterion="D", n_starts=5, seed=1)
    assert opt.n_runs == 11
    assert opt.metadata["criteria"]["D"] == pytest.approx(0.7305, abs=5e-3)


def test_optimal_design_beats_random_on_d():
    rng = np.random.default_rng(0)
    cand = ed.random_design(
        [ed.ContinuousFactor("x1", -1, 1), ed.ContinuousFactor("x2", -1, 1)],
        n=200, seed=0,
    )
    model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")
    cand.model = model
    opt = ed.optimal_design(cand, n_runs=10, criterion="D", n_starts=3, seed=1)

    X_full = model.matrix(cand.matrix)
    random_ds = [ed.d_criterion(X_full[rng.choice(200, 10, replace=False)])
                 for _ in range(50)]
    assert opt.metadata["criteria"]["D"] > np.mean(random_ds)


def test_optimal_design_criterion_a_uses_fedorov():
    cand = ed.random_design(
        [ed.ContinuousFactor("x1", -1, 1), ed.ContinuousFactor("x2", -1, 1)],
        n=80, seed=7,
    )
    cand.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")
    opt = ed.optimal_design(cand, n_runs=8, criterion="A", n_starts=2, seed=2)
    assert opt.metadata["algorithm"] == "fedorov"
    assert opt.metadata["criterion"] == "A"
    assert opt.metadata["criteria"]["A"] > 0


def test_fit_linear_model_recovers_coefficients():
    rng = np.random.default_rng(42)
    pb = ed.plackett_burman(6)
    true = {"factor1": 2.3, "factor2": -3.4, "factor3": 7.12,
            "factor4": -0.03, "factor5": 1.1, "factor6": -0.5}
    y = 1.2 + sum(true[c] * pb.matrix[c] for c in true) + rng.normal(0, 0.01, pb.n_runs)

    me = ed.main_effects(pb, y)
    for c, coef in true.items():
        assert me[c] == pytest.approx(coef, abs=0.05)


def test_half_normal_data_orders_by_magnitude():
    effects = [0.1, -5.0, 0.3, 2.0]
    labels = ["a", "b", "c", "d"]
    hnd = ed.half_normal_data(effects, labels)
    # el ultimo (mayor |efecto|) debe ser 'b'
    assert hnd.iloc[-1]["label"] == "b"
    assert hnd["abs_effect"].is_monotonic_increasing


def test_main_effects_scale_effect_is_twice_coefficient():
    # Para un diseno ortogonal en +/-1, efecto clasico = 2 * coeficiente.
    rng = np.random.default_rng(3)
    pb = ed.plackett_burman(5)
    true = {"factor1": 1.5, "factor2": -2.0, "factor3": 0.4,
            "factor4": 3.1, "factor5": -0.2}
    y = 0.5 + sum(true[c] * pb.matrix[c] for c in true) + rng.normal(0, 0.01, pb.n_runs)

    beta = ed.main_effects(pb, y, scale="coefficient")
    effect = ed.main_effects(pb, y, scale="effect")
    assert np.allclose(effect.to_numpy(), 2.0 * beta.to_numpy())
    assert effect.name == "effect"
    with pytest.raises(ValueError):
        ed.main_effects(pb, y, scale="invalid")


def test_optimal_design_with_categorical_factor():
    # Un factor categorico debe fluir por el modelo (dummy) y optimal_design.
    cand = ed.random_design(
        [ed.ContinuousFactor("x1", -1, 1),
         ed.CategoricalFactor("mat", ["A", "B", "C"])],
        n=120, seed=11,
    )
    cand.model = ed.Model.parse("0 ~ x1 + mat")
    opt = ed.optimal_design(cand, n_runs=9, criterion="D", n_starts=2, seed=4)
    assert opt.n_runs == 9
    assert opt.metadata["criteria"]["D"] > 0
    # los 3 niveles categoricos deben quedar representados
    assert set(opt.matrix["mat"].unique()) == {"A", "B", "C"}
