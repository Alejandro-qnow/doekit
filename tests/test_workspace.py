"""Traceable experiment workspace (project → waves → conclusions)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import doekit as ed


def test_project_wave_sync_ingest_conclude(tmp_path):
    proj = ed.ExperimentProject.create("Demo Screen", root=tmp_path / "experiments")
    assert (proj.path / "PROJECT.json").exists()
    assert proj.slug == "demo-screen"

    exp = ed.experiment(goal="screening", factors=4, budget=8, seed=0)
    exp.evaluate(n_region=400, seed=0)
    wave = proj.new_wave(exp, seed=0)

    assert wave.wave_id == "wave_001"
    assert (wave.path / "doe-configuration" / "experiment.json").exists()
    assert (wave.path / "data" / "run_sheet.csv").exists()
    assert (wave.path / "results" / "evaluation.json").exists()
    assert wave.manifest["status"] == "awaiting_response"
    assert wave.manifest["schema"] == "doekit.WaveManifest/1"

    # fill responses and re-sync
    y = np.random.default_rng(1).normal(size=exp.design.n_runs)
    exp.ingest(y)
    wave.sync(exp)
    assert (wave.path / "data" / "responses.csv").exists()
    assert (wave.path / "results" / "fit.json").exists()
    assert wave.manifest["status"] == "analyzed"

    conclusions = wave.conclude(exp, lang="en")
    assert conclusions["schema"] == "doekit.AutomaticConclusions/1"
    assert "gate_board" in conclusions
    assert conclusions["gate_board"]["process"]["status"] in {
        "stop", "augment", "redesign",
    }
    assert (wave.path / "automatic-conclusions" / "conclusions.json").exists()
    assert (wave.path / "automatic-conclusions" / "conclusions.md").exists()
    assert wave.manifest["status"] == "concluded"

    # round-trip load
    loaded = ed.Experiment.load(wave.path)
    assert loaded.design.n_runs == exp.design.n_runs
    assert loaded.response is not None
    assert loaded.evaluation is not None
    assert loaded.fit is not None


def test_experiment_save_to_project_and_from_dict(tmp_path):
    exp = ed.Experiment.from_design(ed.fold(ed.plackett_burman(4)))
    exp.evaluate(n_region=300, seed=0)
    y = np.linspace(0, 1, exp.design.n_runs)
    exp.ingest(y)

    snap = exp.to_dict()
    json.dumps(snap)
    back = ed.Experiment.from_dict(snap)
    assert back.design.n_runs == exp.design.n_runs
    assert np.allclose(back.response, exp.response)
    assert back.fit is not None

    proj = ed.project("Roundtrip", root=tmp_path)
    wave = exp.save(proj)
    assert wave.wave_id == "wave_001"
    again = ed.Experiment.load(wave.path / "doe-configuration" / "experiment.json")
    assert again.fit is not None


def test_ingest_from_csv(tmp_path):
    proj = ed.project("Ingest CSV", root=tmp_path)
    d = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
    matrix = pd.concat([d.matrix, d.matrix], ignore_index=True)
    design = ed.Design(
        matrix=matrix, factors=d.factors,
        model=ed.Model.main_effects(["A", "B"]),
    )
    exp = ed.Experiment.from_design(design)
    exp.evaluate(n_region=200, seed=0)
    wave = proj.new_wave(exp)

    sheet = pd.read_csv(wave.path / "data" / "run_sheet.csv")
    rng = np.random.default_rng(3)
    sheet["y"] = 1.0 + sheet["A"] + rng.normal(0, 0.1, len(sheet))
    filled = tmp_path / "filled.csv"
    sheet.to_csv(filled, index=False)

    exp2 = wave.ingest_from(filled)
    assert exp2.fit is not None
    assert wave.manifest["status"] in {"analyzed", "concluded"}
    conclusions = exp2.conclude(wave)
    assert "rules" in conclusions


def test_cli_project_init_sync_conclude(tmp_path):
    from doekit.cli import main

    root = tmp_path / "experiments"
    rc = main(["project", "init", "--name", "CLI Demo", "--root", str(root)])
    assert rc == 0
    proj_path = root / "experiment_project_cli-demo"
    assert (proj_path / "PROJECT.json").exists()

    rc = main([
        "project", "sync",
        "--path", str(proj_path),
        "--factors", "4",
        "--budget", "8",
        "--seed", "0",
        "--n-region", "300",
    ])
    assert rc == 0
    wave_path = proj_path / "waves" / "wave_001"
    assert (wave_path / "manifest.json").exists()

    # attach synthetic y then conclude
    exp = ed.Experiment.load(wave_path)
    exp.ingest(np.random.default_rng(0).normal(size=exp.design.n_runs))
    ed.Wave(wave_path).sync(exp)

    rc = main(["project", "conclude", "--path", str(wave_path)])
    assert rc == 0
    assert (wave_path / "automatic-conclusions" / "conclusions.json").exists()


def test_design_evaluation_from_dict_roundtrip():
    d = ed.plackett_burman(4)
    ev = ed.evaluate(d, n_region=200, seed=0)
    ev2 = ed.DesignEvaluation.from_dict(ev.to_dict())
    assert ev2.n_runs == ev.n_runs
    assert ev2.efficiencies["D_efficiency"] == ev.efficiencies["D_efficiency"]
