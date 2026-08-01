"""Sequential DoE: augment, propose_next_runs, compare_designs."""

import json

import numpy as np
import pytest

import doekit as ed


def test_augment_design_appends_runs_and_improves_d():
    # Start from a tiny random design; augment should grow n_runs and not worsen D
    facs = [ed.ContinuousFactor("x1", -1, 1), ed.ContinuousFactor("x2", -1, 1)]
    base = ed.random_design(facs, n=6, seed=0)
    base.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")
    cand = ed.random_design(facs, n=120, seed=1)
    cand.model = base.model

    d0 = ed.d_criterion(base.model.matrix(base.matrix))
    aug = ed.augment_design(base, n_add=4, candidates=cand, criterion="D",
                            n_starts=4, seed=2)
    assert aug.n_runs == base.n_runs + 4
    assert aug.metadata["kind"] == "AugmentedDesign"
    assert aug.metadata["n_original"] == 6
    d1 = ed.d_criterion(aug.model.matrix(aug.matrix))
    # Combined design should have higher D-score than the tiny base
    assert d1 >= d0 - 1e-9


def test_augment_factorial_golden_grows():
    ff = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
    ff.model = ed.Model.parse("0 ~ A + B + A:B")
    # drop to half the factorial as "current"
    import pandas as pd
    cur = ed.Design(matrix=ff.matrix.iloc[:2].copy(), factors=ff.factors,
                    model=ff.model, metadata={"kind": "Partial"})
    aug = ed.augment_design(cur, n_add=2, candidates=ff, criterion="D",
                            n_starts=3, seed=0)
    assert aug.n_runs == 4
    # should be able to estimate the 4-parameter model
    X = aug.model.matrix(aug.matrix)
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_compare_designs_delta_table():
    a = ed.plackett_burman(5)
    b = ed.fold(a)
    cmp_ = ed.compare_designs(a, b, n_region=800, seed=0)
    assert cmp_.delta["n_runs"] == b.n_runs - a.n_runs
    assert "D_efficiency" in cmp_.delta
    assert not cmp_.table.empty
    d = cmp_.to_dict()
    assert d["schema"] == "doekit.DesignComparison/1"
    json.dumps(d)


def test_propose_next_runs_without_response():
    bb = ed.box_behnken({"a": (0, 1), "b": (0, 1), "c": (0, 1)}, center=3)
    prop = ed.propose_next_runs(bb, n_add=3, criterion="D", n_region=600,
                                n_candidates=80, n_starts=3, seed=0)
    assert prop.added.n_runs == 3
    assert prop.combined.n_runs == bb.n_runs + 3
    assert prop.comparison.b["n_runs"] == bb.n_runs + 3
    # Augmented should not be rank-deficient for the quadratic model
    assert not prop.comparison.b.get("rank_deficient", True)
    payload = prop.to_dict()
    assert payload["schema"] == "doekit.NextRunsProposal/1"
    json.dumps(payload)


def test_propose_next_runs_with_response_sets_sigma():
    rng = np.random.default_rng(0)
    pb = ed.plackett_burman(6)
    true = {c: rng.normal(0, 1) for c in pb.matrix.columns}
    y = 1.0 + sum(true[c] * pb.matrix[c] for c in true) + rng.normal(0, 0.2, pb.n_runs)
    # Fold to have residual df for sigma_hat
    d = ed.fold(pb)
    y2 = np.concatenate([y, y + rng.normal(0, 0.2, len(y))])
    prop = ed.propose_next_runs(d, response=y2, n_add=2, n_candidates=60,
                                n_starts=2, n_region=400, seed=1)
    assert prop.sigma_hat is not None and prop.sigma_hat > 0
    assert "propose_next_runs" in " ".join(prop.caveats) or len(prop.caveats) >= 1


def test_propose_respects_budget():
    pb = ed.plackett_burman(4)
    prop = ed.propose_next_runs(pb, n_add=10, budget=pb.n_runs + 3,
                                n_candidates=40, n_starts=2, seed=0)
    assert prop.added.n_runs == 3
    assert prop.combined.n_runs == pb.n_runs + 3


def test_propose_budget_exhausted_raises():
    pb = ed.plackett_burman(4)
    with pytest.raises(ValueError, match="exhausted"):
        ed.propose_next_runs(pb, n_add=2, budget=pb.n_runs)


def test_candidates_from_bounds():
    cand = ed.candidates_from_bounds(
        [("temp", 20, 80), ("ph", 3, 9)], n=50, seed=0
    )
    assert cand.n_runs == 50
    assert set(cand.matrix.columns) == {"temp", "ph"}
    assert cand.metadata["source"] == "bounds"


def test_augment_improves_or_holds_spv_mean_via_propose():
    # Weak starting design -> propose should lower mean SPV or raise D-eff
    facs = [ed.ContinuousFactor("x1", -1, 1), ed.ContinuousFactor("x2", -1, 1)]
    weak = ed.random_design(facs, n=5, seed=3)
    weak.model = ed.Model.parse("0 ~ x1 + x2")
    prop = ed.propose_next_runs(weak, n_add=5, criterion="I", n_candidates=100,
                                n_starts=4, n_region=1000, seed=4)
    dlt_d = prop.comparison.delta["D_efficiency"]
    dlt_spv = prop.comparison.delta["spv_mean"]
    # At least one quality axis should move in the useful direction
    assert (np.isfinite(dlt_d) and dlt_d >= -1e-6) or (
        np.isfinite(dlt_spv) and dlt_spv <= 1e-6
    )
