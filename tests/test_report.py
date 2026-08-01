"""HTML report module (folder + self-contained modes, i18n) and anomaly detection."""

import numpy as np
import pytest

import doekit as ed


def test_report_folder_quality_without_response(tmp_path):
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3)
    path = ed.report(bb, output_dir=str(tmp_path), seed=0)          # folder mode (default)
    assert path.name == "index.html" and path.exists()
    # separate assets: css, images, data
    assert (tmp_path / "report.css").exists()
    assert list((tmp_path / "images").glob("*.png"))                # at least one plot saved
    assert (tmp_path / "data" / "design_matrix.csv").exists()
    assert (tmp_path / "data" / "efficiencies.csv").exists()
    assert (tmp_path / "data" / "fds.csv").exists()

    html = path.read_text(encoding="utf-8")
    assert "<link rel='stylesheet' href='report.css'>" in html      # linked, not inlined
    assert "src='images/" in html                                   # linked images, not base64
    for sec in ("Executive summary", "Methodology", "Design quality",
                "Conclusions and recommendations"):
        assert sec in html
    assert "Analysis results" not in html                           # no response -> no analysis


def test_report_folder_full_with_response(tmp_path):
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3)
    rng = np.random.default_rng(0)
    X = bb.matrix
    y = 80 + 5 * (X["temp"] - 50) / 30 - 3 * ((X["ph"] - 6) / 3) ** 2 \
        + rng.normal(0, 1.0, len(X))
    path = ed.report(bb, response=y.to_numpy(),
                     model=ed.Model.full_quadratic(["temp", "ph", "conc"]),
                     output_dir=str(tmp_path))
    html = path.read_text(encoding="utf-8")
    assert "Analysis results" in html
    assert "Anomalous values" in html
    # analysis data exported
    assert (tmp_path / "data" / "coefficients.csv").exists()
    assert (tmp_path / "data" / "anomalies.csv").exists()


def test_report_self_contained_single_file(tmp_path):
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3)
    path = ed.report(bb, output_dir=str(tmp_path), filename="q.html",
                     self_contained=True, seed=0)
    assert path.name == "q.html" and path.exists()
    html = path.read_text(encoding="utf-8")
    assert "data:image/png;base64" in html          # plots embedded as base64
    assert "<style>" in html                          # CSS inlined
    assert not (tmp_path / "images").exists()         # no separate assets


def test_report_spanish(tmp_path):
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3)
    path = ed.report(bb, output_dir=str(tmp_path), lang="es", seed=0)
    html = path.read_text(encoding="utf-8")
    assert "<html lang='es'>" in html
    for sec in ("Resumen ejecutivo", "Metodologia", "Calidad del diseno"):
        assert sec in html


def test_report_arg_in_fit_and_evaluate(tmp_path):
    bb = ed.box_behnken(3, center=3)
    y = np.random.default_rng(1).normal(50, 3, bb.n_runs)
    fit = ed.fit_linear_model(bb, y, model=ed.Model.main_effects(bb.factor_names),
                              report=str(tmp_path / "fitrep"))
    assert fit.report_path is not None and fit.report_path.exists()

    ev = ed.evaluate(bb, report={"output_dir": str(tmp_path / "evrep")})
    assert ev.report_path.exists()


def test_anomaly_detection_flags_outlier_not_everything():
    bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)}, center=3)
    rng = np.random.default_rng(0)
    X = bb.matrix
    y = (80 + 5 * (X["temp"] - 50) / 30 - 3 * ((X["ph"] - 6) / 3) ** 2
         + rng.normal(0, 1.0, len(X))).to_numpy().copy()
    y[4] += 25.0   # deliberate outlier
    fit = ed.fit_linear_model(bb, y, model=ed.Model.full_quadratic(["temp", "ph", "conc"]))
    an = fit.anomalies()
    # run 4 must show up as an outlier
    assert 4 in an["run"].values
    assert "outlier" in an.set_index("run").loc[4, "reason"]
    # the Cook's D>1 cutoff must not flag almost the whole design
    assert len(an) <= 4


def test_fit_result_has_diagnostics():
    bb = ed.box_behnken(3, center=3)
    y = np.random.default_rng(2).normal(50, 3, bb.n_runs)
    fit = ed.fit_linear_model(bb, y, model=ed.Model.full_quadratic(bb.factor_names))
    assert fit.leverage is not None and len(fit.leverage) == bb.n_runs
    assert fit.cooks_distance is not None
    assert np.isfinite(fit.r_squared_adj)
    # mean leverage ~ p/N
    assert fit.leverage.mean() == pytest.approx((bb.n_runs - fit.dof) / bb.n_runs, abs=1e-6)
