"""Optimize intent: acquisitions, Pareto, convergence, batch, serialization."""

import json

import numpy as np
import pandas as pd
import pytest

import doekit as ed
from doekit.orchestration.optimize import (
    expected_improvement, probability_of_improvement, upper_confidence_bound,
    expected_hypervolume_improvement, get_acquisition,
    pareto_mask, pareto_front, dominates, hypervolume,
)
from doekit.assessment.surrogate.base import _sklearn_available


# --------------------------------------------------------------------------
# acquisitions
# --------------------------------------------------------------------------

def test_expected_improvement_non_negative_and_zero_when_certain():
    mean = np.array([1.0, 2.0, 0.5])
    std = np.array([0.1, 1.0, 0.0])
    ei = expected_improvement(mean, std, best=1.5)
    assert np.all(ei >= 0)
    # certain (std=0) and below best -> no improvement
    assert expected_improvement(np.array([0.5]), np.array([0.0]), best=1.5)[0] == 0.0


def test_probability_of_improvement_in_unit_interval():
    mean = np.array([1.0, 2.0, 0.5])
    std = np.array([0.1, 1.0, 0.3])
    pi = probability_of_improvement(mean, std, best=1.5)
    assert np.all((pi >= 0) & (pi <= 1))


def test_ucb_prefers_uncertainty():
    a = upper_confidence_bound(np.array([1.0]), np.array([0.1]), kappa=2.0)
    b = upper_confidence_bound(np.array([1.0]), np.array([2.0]), kappa=2.0)
    assert b[0] > a[0]


def test_min_goal_flips_direction():
    # for minimization, a small mean is an improvement over best
    ei = expected_improvement(np.array([0.5, 2.0]), np.array([0.1, 0.1]),
                              best=1.0, goal="min")
    assert ei[0] > ei[1]


def test_get_acquisition_lookup():
    assert get_acquisition("ei") is expected_improvement
    assert get_acquisition("EHVI") is expected_hypervolume_improvement
    with pytest.raises(ValueError):
        get_acquisition("nope")


# --------------------------------------------------------------------------
# pareto
# --------------------------------------------------------------------------

def test_dominance_and_mask():
    assert dominates([2, 2], [1, 1])
    assert not dominates([2, 1], [1, 2])
    Y = np.array([[3, 1], [2, 2], [1, 3], [0.5, 0.5]])
    mask = pareto_mask(Y)
    assert mask.tolist() == [True, True, True, False]


def test_hypervolume_exact_2d_and_monotone():
    Y = np.array([[2, 1], [1, 2]])
    ref = np.array([0, 0])
    assert hypervolume(Y, ref) == pytest.approx(3.0)
    grown = hypervolume(np.vstack([Y, [1.5, 1.5]]), ref)
    assert grown > 3.0
    # a dominated point does not change the hypervolume
    assert hypervolume(np.vstack([Y, [0.5, 0.5]]), ref) == pytest.approx(3.0)


def test_ehvi_non_negative_and_prefers_non_dominated():
    front = np.array([[2.0, 1.0], [1.0, 2.0]])
    means = np.array([[1.5, 1.5], [0.1, 0.1]])
    stds = np.array([[0.2, 0.2], [0.2, 0.2]])
    ehvi = expected_hypervolume_improvement(means, stds, front, seed=0,
                                            n_samples=256)
    assert np.all(ehvi >= 0)
    assert ehvi[0] > ehvi[1]


def test_pareto_front_respects_goals():
    Y = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    # minimize b: (1,10) dominates (2,20) only if a also minimized -> both max on a
    front = pareto_front(Y.to_numpy(), goals={"a": "max", "b": "min"},
                         columns=["a", "b"])
    # (1,10): better b, worse a; (2,20): better a, worse b -> both non-dominated
    assert len(front) == 2


# --------------------------------------------------------------------------
# propose_next_runs: learn regression + optimize behaviour
# --------------------------------------------------------------------------

def _rsm(seed=0, noise=0.03):
    cols = list(ed.central_composite(2).matrix.columns)
    d = ed.central_composite(2)
    facs = [ed.ContinuousFactor(cols[0], -1, 1),
            ed.ContinuousFactor(cols[1], -1, 1)]
    d = ed.Design(matrix=d.matrix, factors=facs,
                  model=ed.Model.full_quadratic(cols))
    rng = np.random.default_rng(seed)
    X = d.matrix[cols].to_numpy(dtype=float)
    y = 5 - 3 * ((X[:, 0] - 0.4) ** 2 + (X[:, 1] + 0.3) ** 2)
    y = y + noise * rng.standard_normal(len(X))
    return d, cols, y


def test_learn_intent_is_unchanged():
    d, cols, y = _rsm()
    p_default = ed.propose_next_runs(d, response=y, n_add=3, seed=7)
    p_learn = ed.propose_next_runs(d, response=y, n_add=3, seed=7, intent="learn")
    assert p_learn.intent == "learn"
    pd.testing.assert_frame_equal(p_default.added.matrix, p_learn.added.matrix)


def test_optimize_batch_no_duplicates_and_fields():
    d, cols, y = _rsm()
    p = ed.propose_next_runs(d, response=y, n_add=4, intent="optimize",
                             surrogate="ols", seed=1)
    assert p.intent == "optimize"
    assert p.acquisition == "ei"
    assert p.added.n_runs == 4
    # constant-liar diversifies: no repeated rows
    assert p.added.matrix.drop_duplicates().shape[0] == 4
    assert p.best_so_far is not None
    assert p.explore_exploit["mode"] in {"exploring", "exploiting", "balanced", "unknown"}


def test_optimize_serialization_roundtrip():
    d, cols, y = _rsm()
    p = ed.propose_next_runs(d, response=y, n_add=2, intent="optimize",
                             surrogate="ols", seed=1)
    dd = p.to_dict()
    assert dd["schema"] == "doekit.NextRunsProposal/1"
    assert dd["intent"] == "optimize"
    assert dd["acquisition"] == "ei"
    assert dd["surrogate"]["kind"] == "OLSSurrogate"
    json.dumps(dd)  # must be JSON-safe


def test_optimize_multiobjective_pareto_front():
    d, cols, _ = _rsm()
    X = d.matrix[cols].to_numpy(dtype=float)
    y1 = 5 - (X[:, 0] ** 2 + X[:, 1] ** 2)
    y2 = -(X[:, 0] - 0.5) ** 2
    Y = np.column_stack([y1, y2])
    p = ed.propose_next_runs(d, response=Y, n_add=2, intent="optimize",
                             objectives=["yield", "purity"],
                             goals={"yield": "max", "purity": "max"},
                             surrogate="ols", seed=2)
    assert p.acquisition == "ehvi"
    assert isinstance(p.pareto_front, list) and len(p.pareto_front) >= 1
    json.dumps(p.to_dict())


def test_optimize_simplex_candidates_are_feasible():
    mix = ed.simplex_lattice(3, degree=2)
    cols = list(mix.matrix.columns)
    X = mix.matrix.to_numpy(dtype=float)
    y = 2 * X[:, 0] + X[:, 1] + 3 * X[:, 0] * X[:, 2]
    p = ed.propose_next_runs(mix, response=y, n_add=3, intent="optimize",
                             surrogate="ols", seed=0)
    add = p.added.matrix.to_numpy(dtype=float)
    assert np.allclose(add.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(add >= -1e-9)


def _loop_best(mode, seed, generations=6, n_add=2):
    rng = np.random.default_rng(seed)
    opt = np.array([0.6, -0.5])

    def truth(M):
        M = np.atleast_2d(np.asarray(M, dtype=float))
        return 5.0 - 3 * ((M[:, 0] - opt[0]) ** 2 + (M[:, 1] + 0.5) ** 2)

    facs = [ed.ContinuousFactor("x1", -1, 1), ed.ContinuousFactor("x2", -1, 1)]
    mdl = ed.Model.full_quadratic(["x1", "x2"])
    base = ed.full_factorial({"x1": [-1, 1], "x2": [-1, 1]})
    frame = base.matrix.copy()
    y = truth(frame.to_numpy()) + 0.03 * rng.standard_normal(len(frame))
    for gen in range(generations):
        d = ed.Design(matrix=frame.copy(), factors=facs, model=mdl)
        if mode == "optimize":
            p = ed.propose_next_runs(d, response=y, n_add=n_add,
                                     intent="optimize", surrogate="ols",
                                     seed=seed + gen)
            newX = p.added.matrix
        else:
            newX = pd.DataFrame({"x1": rng.uniform(-1, 1, n_add),
                                 "x2": rng.uniform(-1, 1, n_add)})
        ynew = truth(newX.to_numpy()) + 0.03 * rng.standard_normal(len(newX))
        frame = pd.concat([frame, newX], ignore_index=True)
        y = np.concatenate([y, ynew])
    return float(np.max(y))


def test_optimize_converges_and_beats_random():
    opt_scores = [_loop_best("optimize", s) for s in range(4)]
    rnd_scores = [_loop_best("random", s) for s in range(4)]
    # approaches the known optimum (~5.0) and beats random at equal budget
    assert np.mean(opt_scores) > 4.5
    assert np.mean(opt_scores) > np.mean(rnd_scores)


# --------------------------------------------------------------------------
# plots (smoke) — skipped without matplotlib
# --------------------------------------------------------------------------

def test_surrogate_plots_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from doekit.presentation.render import figures_mpl as P
    from doekit.assessment.surrogate import fit_surrogate

    d, cols, y = _rsm()
    sur = fit_surrogate(d, y, kind="ols")
    p = ed.propose_next_runs(d, response=y, n_add=2, intent="optimize",
                             surrogate="ols", seed=1)
    assert isinstance(P.surrogate_surface(sur, measured=d.matrix,
                                          proposed=p.added.matrix), Axes)
    assert isinstance(P.acquisition_plot(sur, proposed=p.added.matrix), Axes)
    assert isinstance(P.convergence_plot([2, 3, 4, 4.8], optimum=5.0), Axes)
    assert isinstance(P.parity_plot(sur), Axes)
    assert isinstance(P.calibration_plot(sur), Axes)
    assert isinstance(P.slice_plot(sur, factor=cols[0]), Axes)
    assert isinstance(P.fds_plot(d, surrogate=sur), Axes)
    Y = np.column_stack([y, -y])
    assert isinstance(P.pareto_plot(Y, columns=["a", "b"]), Axes)
