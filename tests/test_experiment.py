"""Experiment aggregate, multi-y, export, CLI (0.7)."""

import json

import numpy as np
import pandas as pd

import doekit as ed


def test_experiment_from_goal_plan_and_evaluate():
    exp = ed.experiment(goal="screening", factors=5, budget=8, seed=0)
    assert exp.design.n_runs <= 8
    assert exp.recommendation is not None
    plan = exp.plan
    assert "run_id" in plan.columns and "y" in plan.columns
    ev = exp.evaluate(n_region=500, seed=0)
    assert ev.n_runs == exp.design.n_runs


def test_experiment_ingest_and_next():
    d = ed.fold(ed.plackett_burman(4))
    exp = ed.Experiment.from_design(d)
    rng = np.random.default_rng(0)
    y = rng.normal(size=d.n_runs)
    exp.ingest(y)
    assert exp.fit is not None
    nxt = exp.next(n_add=2, n_candidates=40, n_starts=2, seed=1)
    assert nxt.added.n_runs == 2


def test_multi_response_summary():
    base = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
    # replicate so OLS has residual df
    matrix = pd.concat([base.matrix, base.matrix], ignore_index=True)
    d = ed.Design(
        matrix=matrix,
        factors=base.factors,
        model=ed.Model.main_effects(["A", "B"]),
    )
    rng = np.random.default_rng(2)
    y1 = 1 + d.matrix["A"] + rng.normal(0, 0.05, d.n_runs)
    y2 = 0.1 * d.matrix["B"] + rng.normal(0, 0.5, d.n_runs)
    exp = ed.experiment(design=d, responses=["yield", "impurity"])
    exp.ingest({"yield": y1, "impurity": y2})
    summary = exp.multi_response_summary(goals={"impurity": "min"})
    assert "yield" in summary["per_response"] and "impurity" in summary["per_response"]
    assert "Stronger" in summary["note"] or "Single" in summary["note"]
    assert summary["desirability"] is not None
    payload = exp.to_dict()
    assert payload["schema"] == "doekit.Experiment/1"
    assert "multi_response" in payload
    json.dumps(payload)


def test_export_csv(tmp_path):
    exp = ed.experiment(goal="screening", factors=4, seed=0)
    path = exp.export_csv(tmp_path / "runs.csv")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "run_id" in text


def test_compare_without_response():
    exp = ed.Experiment.from_design(
        ed.random_design([ed.ContinuousFactor("x", -1, 1)], n=6, seed=0)
    )
    exp.design = exp.design.replace(model=ed.Model.parse("0 ~ x"))
    cmp_ = exp.compare(n_add=3, n_starts=2, seed=1)
    assert cmp_.delta["n_runs"] == 3


def test_cli_recommend(tmp_path):
    from doekit.cli import main
    out = tmp_path / "sheet.csv"
    rc = main(["recommend", "--factors", "4", "--budget", "8",
               "--export", str(out), "--seed", "0"])
    assert rc == 0
    assert out.exists()


def test_cli_experiment_json(capsys):
    from doekit.cli import main
    rc = main(["experiment", "--factors", "temp:20:80,ph:3:9",
               "--goal", "optimization", "--json", "--seed", "0"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "doekit.Experiment/1"
