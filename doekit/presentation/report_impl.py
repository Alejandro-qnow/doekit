"""HTML report of a DoE: methodology, quality, results, anomalies and recommendations.

Two output modes:

- **folder** (default, ``self_contained=False``): writes ``<output_dir>/index.html``
  plus ``report.css``, ``images/*.png`` and ``data/*.csv`` — portable assets a
  researcher can reuse.
- **self-contained** (``self_contained=True``): a single ``.html`` with inlined CSS
  and base64 plots — trivially emailable.

The report is **bilingual** (``lang="en"`` default, ``lang="es"``). The narrative is
**rule-based** (deterministic, no LLM), so it is reproducible and offline.

Usage::

    import doekit as ed
    ed.report(design, response=y)                          # folder ./report/
    ed.report(design, response=y, self_contained=True)     # single .html
    ed.report(design, response=y, lang="es")               # Spanish
    ed.fit_linear_model(design, y, report="report/")       # as an argument
    ed.evaluate(design, report=True)                       # True -> ./report/

Requires ``matplotlib`` (extra ``[report]``).
"""

from __future__ import annotations

import base64
import io
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .._version import __version__
from ..assessment.analysis import fit_linear_model
from ..assessment.evaluation import evaluate as _evaluate
from ..domain.model import Model, Interaction, Power
from ..domain.design import Design

_DEFAULT_THRESHOLDS = {"d_excellent": 90.0, "d_ok": 70.0, "power_target": 0.8, "vif_warn": 5.0}

_ASSETS = Path(__file__).resolve().parents[1] / "assets"


from .narrative.i18n import _STRINGS, _t, _KIND_TO_LABEL


def _infer_scenario(design: Design, model: Optional[Model]):
    """Infer (goal, factors, model_order, budget) from the executed design."""
    kind = design.metadata.get("kind", "")
    terms = model.terms if model is not None else []
    order = ("quadratic" if any(isinstance(t, Power) for t in terms)
             else "interactions" if any(isinstance(t, Interaction) for t in terms)
             else "linear")
    rsm = {"BoxBehnken", "CentralComposite"}
    goal = "optimization" if (kind in rsm or order == "quadratic") else "screening"
    spec = design.factors if design.factors else len(design.matrix.columns)
    return goal, spec, order, design.n_runs


def _recommendation_block(design: Design, model: Optional[Model]) -> Optional[dict]:
    """Run the advisor for the inferred scenario and compare it with what was run.

    Returns a data dict (``method``, ``actual``, ``matches``, ``note``,
    ``rationale``, ``table``, ``caveats``). Strings from :func:`recommend_design`
    are English; the HTML renderer may localize its own note.
    """
    if model is None:
        return None
    from ..orchestration.advise import recommend_design  # noqa: PLC0415
    goal, spec, order, budget = _infer_scenario(design, model)
    try:
        rec = recommend_design(goal, spec, budget=budget, model_order=order, n_region=3000)
    except Exception:
        return None
    actual = _KIND_TO_LABEL.get(design.metadata.get("kind", ""), design.metadata.get("kind", ""))
    matches = rec.method == actual
    if matches:
        note = (f"The executed design ({actual}) matches the recommended method "
                "for this case: good choice.")
    else:
        note = (f"For this case ({goal}, {rec.scenario['n_factors']} factors, "
                f"model '{order}', budget {budget}), the more efficient design would be "
                f"{rec.method}; you ran {actual}. This is informative, not an error: "
                "review the alternatives table and choose by your priorities.")
    return {"method": rec.method, "actual": actual, "matches": matches, "note": note,
            "rationale": rec.rationale, "table": rec.table, "caveats": rec.caveats}


def report_summary(design: Design, response=None, model: Optional[Model] = None,
                   effect_size=1.0, sigma: float = 1.0, alpha: float = 0.05,
                   seed: Optional[int] = None, thresholds: Optional[dict] = None,
                   lang: str = "en", blocks=None, cov_type: str = "nonrobust",
                   groups=None) -> dict:
    """Return the experiment's **semantic guide** without writing HTML.

    Same narrative content as the report (methodology, executive summary,
    recommendations, quality and anomalies) but as a data structure: useful to
    show inline in a notebook, or for an agent (MCP) to consume.

    Parameters
    ----------
    lang : {"en", "es"}, default "en"
        Language of the methodology/summary/recommendations prose.
    blocks : str or array-like, optional
        Fixed blocks for the OLS fit (see :func:`~doekit.assessment.analysis.fit_linear_model`).
    cov_type : str, default "nonrobust"
        Covariance type for the OLS fit.
    groups : str or array-like, optional
        If set, also fit a mixed model and include ``mixed_fit`` in the result.

    Returns
    -------
    dict
        Keys ``methodology``, ``quality``, ``executive_summary``,
        ``recommendations``, ``anomalies``, ``recommendation``, and when a
        response is provided also ``fit``, ``anova``; ``mixed_fit`` when
        ``groups`` is set.
    """
    from ..assessment.analysis import anova_table, fit_mixed_model  # noqa: PLC0415

    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    model = model or design.model
    ev = _evaluate(design, model=model, effect_size=effect_size, sigma=sigma,
                   alpha=alpha, seed=seed)
    fit = None
    anova = None
    mixed = None
    if response is not None:
        fit = fit_linear_model(design, response, model=model,
                               blocks=blocks, cov_type=cov_type)
        try:
            anova = anova_table(fit)
        except Exception:
            anova = None
        if groups is not None:
            try:
                mixed = fit_mixed_model(design, response, groups=groups, model=model)
            except Exception:
                mixed = None
    out = {
        "methodology": _methodology_prose(design, model, lang),
        "quality": ev.efficiencies,
        "executive_summary": _executive_summary(design, ev.efficiencies, fit, thr, lang),
        "recommendations": _recommendations(design, ev.efficiencies, fit, thr, lang),
        "anomalies": fit.anomalies() if fit is not None else None,
        "recommendation": _recommendation_block(design, model),
        "fit": fit.to_dict() if fit is not None else None,
        "anova": anova.to_dict("records") if anova is not None else None,
        "mixed_fit": mixed.to_dict() if mixed is not None else None,
    }
    return out


def run_report_arg(design, response=None, model=None, report=None, **extra):
    """Normalize the functions' ``report=`` argument and generate the HTML.

    ``report`` accepts ``None``/``False`` (do nothing), ``True`` (default folder
    ``report/``), a folder path (``str``/``Path``) or an options ``dict`` for
    :func:`report`. Returns the written ``Path`` or ``None``.
    """
    if report is None or report is False:
        return None
    if report is True:
        opts = {}
    elif isinstance(report, dict):
        opts = dict(report)
    else:
        opts = {"output_dir": report}
    opts.setdefault("model", model)
    return report_html(design, response=response, **{**extra, **opts})


def _require_mpl():
    """Import matplotlib (Agg backend) or raise a helpful ImportError."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for reports. "
                          "Install with 'pip install doekit[report]'.") from exc
    return plt


def _load_css() -> str:
    """Read the report stylesheet shipped in ``doekit/assets/report.css``."""
    return (_ASSETS / "report.css").read_text(encoding="utf-8")


def _fig_b64(fig) -> str:
    """Render a figure to a base64 data URI (self-contained mode) and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _fig_save(fig, path: Path) -> None:
    """Save a figure to a PNG file (folder mode) and close it."""
    fig.savefig(path, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    plt.close(fig)


def _stars(p: float) -> str:
    """Return the significance-star string for a p-value."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def _semaphore(value: float, good: float, ok: float) -> str:
    """Map a value to a semaphore color name (green/amber/red/gray)."""
    if not np.isfinite(value):
        return "gray"
    return "green" if value >= good else "amber" if value >= ok else "red"


# --- rule-based narrative ---------------------------------------------------

def _methodology_prose(design: Design, model: Model, lang: str = "en") -> str:
    """Build the localized methodology paragraph from the design metadata."""
    kind = design.metadata.get("kind", "Design")
    m = design.metadata
    prose = _t(lang, "kind_prose")
    parts = [prose.get(kind, _t(lang, "meth_default").format(kind=kind))]
    if m.get("resolution"):
        parts.append(_t(lang, "meth_resolution").format(res=m["resolution"],
                                                         rel=m.get("defining_relation", "")))
    if m.get("alpha_value"):
        parts.append(_t(lang, "meth_alpha").format(alpha=m["alpha_value"], kind=m.get("alpha", "")))
    if m.get("center") is not None:
        parts.append(_t(lang, "meth_center").format(center=m["center"]))
    if m.get("conference_order"):
        s = _t(lang, "meth_conf").format(order=m["conference_order"])
        s += (_t(lang, "meth_phantom").format(n=m["phantom_factors"])
              if m.get("phantom_factors") else ".")
        parts.append(s)
    if m.get("criterion"):
        parts.append(_t(lang, "meth_criterion").format(crit=m["criterion"],
                                                       algo=m.get("algorithm", "")))
    return " ".join(parts)


#: Standard template designs (their efficiency is what it is by construction).
from .narrative.templates import (  # noqa: E402
    STANDARD_TEMPLATES as _STANDARD_TEMPLATES,
    RSM_TEMPLATES as _RSM_TEMPLATES,
    SCREENING_KINDS as _SCREENING,
)


def _sig_terms(fit) -> list[str]:
    """Return the names of the significant (p<0.05) non-intercept terms."""
    return [n for n, p in zip(fit.names, fit.pvalues)
            if n != "(Intercept)" and np.isfinite(p) and p < 0.05]


def _recommendations(design, eff, fit, thr, lang: str = "en") -> list[str]:
    """Build the localized list of actionable recommendation bullets."""
    recs = []
    kind = design.metadata.get("kind", "")
    if eff.get("rank_deficient"):
        recs.append(_t(lang, "rec_saturated"))
        return recs
    # D-efficiency: only actionable for NON-standard designs (optimal/custom).
    d = eff["D_efficiency"]
    if np.isfinite(d) and d < thr["d_ok"]:
        if kind in _RSM_TEMPLATES:
            recs.append(_t(lang, "rec_d_rsm").format(d=d))
        elif kind not in _STANDARD_TEMPLATES:
            recs.append(_t(lang, "rec_d_nonstd").format(d=d, ok=thr["d_ok"]))
    if fit is not None:
        if fit.dof <= 0:
            msg = _t(lang, "rec_fit_saturated")
            msg += (_t(lang, "rec_screen_suffix") if kind in _SCREENING
                    else _t(lang, "rec_addruns_suffix"))
            recs.append(msg)
        else:
            sig = _sig_terms(fit)
            if sig:
                recs.append(_t(lang, "rec_sig").format(terms=", ".join(sig)))
            else:
                recs.append(_t(lang, "rec_nosig"))
            if kind in _SCREENING and sig:
                recs.append(_t(lang, "rec_nextstep"))
        an = fit.anomalies()
        if len(an):
            recs.append(_t(lang, "rec_anom").format(n=len(an)))
    return recs


def _executive_summary(design, eff, fit, thr, lang: str = "en") -> list[str]:
    """Build the localized executive-summary bullets (quality + fit highlights)."""
    bullets = []
    if eff.get("rank_deficient"):
        bullets.append(_t(lang, "exec_saturated"))
        return bullets
    d = eff["D_efficiency"]
    kind = design.metadata.get("kind", "")
    if d >= thr["d_excellent"]:
        level = _t(lang, "lvl_excellent")
    elif d >= thr["d_ok"]:
        level = _t(lang, "lvl_acceptable")
    elif kind in _RSM_TEMPLATES:
        level = _t(lang, "lvl_rsm")
    else:
        level = _t(lang, "lvl_low")
    bullets.append(_t(lang, "exec_quality").format(d=d, level=level, g=eff["G_efficiency"]))
    if fit is not None:
        if fit.dof <= 0:
            bullets.append(_t(lang, "exec_fit_saturated").format(r2=fit.r_squared))
        else:
            sig = _sig_terms(fit)
            head = (_t(lang, "exec_sig").format(terms=", ".join(sig)) if sig
                    else _t(lang, "exec_nosig"))
            bullets.append(head + _t(lang, "exec_r2").format(r2=fit.r_squared,
                                                             r2adj=fit.r_squared_adj))
    return bullets


# --- figures ----------------------------------------------------------------

def _build_figures(plt, plotting, design, model, ev, eff, fit, lang):
    """Build the report figures. Returns ``(figures, failed)``.

    ``figures`` maps a key to a matplotlib figure for each plot that was built;
    ``failed`` maps a key to an error message for each plot that was attempted but
    raised (so the report can show a visible placeholder instead of dropping it).
    """
    figures, failed = {}, {}

    def attempt(key, builder):
        try:
            figures[key] = builder()
        except Exception as exc:  # pragma: no cover - defensive
            failed[key] = str(exc)

    if not eff.get("rank_deficient"):
        def _fds():
            _, ax = plt.subplots(figsize=(6.5, 3.8))
            plotting.fds_plot(ev.fds, ax=ax)
            return ax.figure
        attempt("fds", _fds)
    if len(ev.power):
        def _power():
            _, ax = plt.subplots(figsize=(6, 3.4))
            plotting.power_plot(ev.power, ax=ax)
            return ax.figure
        attempt("power", _power)

    def _alias():
        from ..assessment.evaluation import alias_matrix  # noqa: PLC0415
        A = alias_matrix(design, model=model)
        if A.shape[1] == 0:
            return None
        _, ax = plt.subplots(figsize=(6, 4))
        plotting.alias_heatmap(A, ax=ax)
        return ax.figure
    A_fig = None
    try:
        A_fig = _alias()
    except Exception as exc:  # pragma: no cover - defensive
        failed["alias"] = str(exc)
    if A_fig is not None:
        figures["alias"] = A_fig

    if fit is not None:
        def _halfnormal():
            eff_s = fit.summary_frame()
            mask = eff_s["term"] != "(Intercept)"
            _, ax = plt.subplots(figsize=(6, 3.6))
            plotting.half_normal_plot(eff_s["estimate"][mask].to_numpy(),
                                      eff_s["term"][mask].tolist(), ax=ax)
            return ax.figure
        attempt("halfnormal", _halfnormal)
        if fit.fitted is not None and fit.dof > 0:
            def _resid():
                _, ax = plt.subplots(figsize=(6, 3.4))
                ax.scatter(fit.fitted, fit.studentized_resid, color="tab:blue")
                ax.axhline(0, color="k", lw=.8)
                ax.axhline(3, color="r", ls="--", lw=.8)
                ax.axhline(-3, color="r", ls="--", lw=.8)
                ax.set_xlabel(_t(lang, "resid_x"))
                ax.set_ylabel(_t(lang, "resid_y"))
                ax.set_title(_t(lang, "resid_title"))
                ax.grid(True, alpha=.3)
                return ax.figure
            attempt("resid", _resid)
    return figures, failed


def _write_data_csvs(data_dir: Path, design, ev, fit) -> None:
    """Export the report's underlying data as CSV files (folder mode)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    design.matrix.to_csv(data_dir / "design_matrix.csv", index=False)
    pd.Series(ev.efficiencies, name="value").to_csv(data_dir / "efficiencies.csv")
    if len(ev.power):
        ev.power.to_csv(data_dir / "power.csv")
    if len(ev.vif):
        ev.vif.to_csv(data_dir / "vif.csv")
    ev.fds.to_csv(data_dir / "fds.csv", index=False)
    if fit is not None:
        fit.summary_frame().to_csv(data_dir / "coefficients.csv", index=False)
        fit.anomalies().to_csv(data_dir / "anomalies.csv", index=False)


# --- HTML construction -------------------------------------------------------

def _card(label, value, color):
    """Render a semaphore metric card."""
    v = f"{value:.0f}%" if np.isfinite(value) else "n/d"
    return (f'<div class="card"><div class="val {color}">{v}</div>'
            f'<div class="lbl">{label}</div></div>')


def report_html(design: Design, response=None, model: Optional[Model] = None,
                output_dir="report", filename: Optional[str] = None,
                title: Optional[str] = None, effect_size=1.0, sigma: float = 1.0,
                alpha: float = 0.05, thresholds: Optional[dict] = None,
                seed: Optional[int] = None, self_contained: bool = False,
                lang: str = "en", open_browser: bool = False) -> Path:
    """Generate an HTML report of the design (and its analysis if responses are given).

    Without ``response`` it produces a *design-quality* report; with ``response``,
    a *complete* one (analysis + anomalies).

    Parameters
    ----------
    design : Design
        The design (and, with ``response``, the executed experiment) to report.
    response : array-like, optional
        Measured response per run; enables the analysis and anomaly sections.
    model : Model, optional
        Model to evaluate/fit; resolved from the design if omitted.
    output_dir : str or Path, default "report"
        In folder mode, the report folder itself (holds ``index.html``,
        ``report.css``, ``images/`` and ``data/``). In self-contained mode, the
        directory the single ``.html`` is written to.
    filename : str, optional
        Name of the main HTML file (default ``index.html`` in folder mode, a
        timestamped name in self-contained mode).
    title : str, optional
        Report title; a default is built from the design kind.
    effect_size, sigma, alpha : float
        Passed to the evaluation (power analysis).
    thresholds : dict, optional
        Overrides for the quality thresholds (``d_excellent``, ``d_ok``,
        ``power_target``, ``vif_warn``).
    seed : int, optional
        Seed for the region sampling of the evaluation.
    self_contained : bool, default False
        If ``True``, write a single ``.html`` with inlined CSS and base64 plots.
        If ``False`` (default), write a folder with separate assets and data.
    lang : {"en", "es"}, default "en"
        Report language.
    open_browser : bool, default False
        Open the written report in the web browser.

    Returns
    -------
    pathlib.Path
        Path to the written HTML file (the ``index.html`` in folder mode).
    """
    plt = _require_mpl()
    from ..presentation.render import figures_mpl as plotting  # noqa: PLC0415
    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    model = model or design.model
    kind = design.metadata.get("kind", "Design")
    title = title or _t(lang, "title_default").format(kind=kind)

    ev = _evaluate(design, model=model, effect_size=effect_size, sigma=sigma,
                   alpha=alpha, seed=seed)
    eff = ev.efficiencies
    fit = fit_linear_model(design, response, model=model) if response is not None else None

    figures, failed = _build_figures(plt, plotting, design, model, ev, eff, fit, lang)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if self_contained:
        imgs = {k: _fig_b64(f) for k, f in figures.items()}
        head = f"<style>{_load_css()}</style>"
        if filename is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reporte_{kind.lower()}_{stamp}.html"
    else:
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        imgs = {}
        for k, f in figures.items():
            _fig_save(f, img_dir / f"{k}.png")
            imgs[k] = f"images/{k}.png"
        _write_data_csvs(out / "data", design, ev, fit)
        (out / "report.css").write_text(_load_css(), encoding="utf-8")
        head = "<link rel='stylesheet' href='report.css'>"
        filename = filename or "index.html"

    html = _render_html(design, model, ev, eff, fit, imgs, failed, thr, title, head, lang)

    path = out / filename
    path.write_text(html, encoding="utf-8")
    if open_browser:  # pragma: no cover
        webbrowser.open(path.resolve().as_uri())
    return path


def _img_or_placeholder(S, imgs, failed, key, lang, alt=""):
    """Append an <img> tag, or a visible placeholder if the plot failed."""
    if key in imgs:
        S.append(f"<img src='{imgs[key]}' alt='{alt or key}'>")
    elif key in failed:
        S.append(f"<p class='warn'>{_t(lang, 'plot_fail').format(key=key, err=failed[key])}</p>")


def _render_html(design, model, ev, eff, fit, imgs, failed, thr, title, head, lang) -> str:
    """Assemble the full HTML document string from the computed pieces."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    d_col = _semaphore(eff["D_efficiency"], thr["d_excellent"], thr["d_ok"])
    a_col = _semaphore(eff["A_efficiency"], thr["d_excellent"], thr["d_ok"])
    g_col = _semaphore(eff["G_efficiency"], thr["d_excellent"], thr["d_ok"])
    verdict = "gray" if eff.get("rank_deficient") else d_col

    S = [f"<!doctype html><html lang='{lang}'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         f"<title>{title}</title>{head}</head><body>"]
    S.append(f"<div class='hero {verdict}'><h1>{title}</h1>")
    S.append("<div class='sub'>" + _t(lang, "sub").format(
        now=now, ver=__version__, runs=ev.n_runs, params=ev.n_params, dof=ev.dof) + "</div></div>")

    # 1. executive summary
    S.append(f"<h2>{_t(lang, 'h_exec')}</h2><ul>")
    for b in _executive_summary(design, eff, fit, thr, lang):
        S.append(f"<li>{b}</li>")
    for r in _recommendations(design, eff, fit, thr, lang)[:1]:
        S.append(f"<li><b>{_t(lang, 'recommendation')}:</b> {r}</li>")
    S.append("</ul>")

    # 2. methodology
    S.append(f"<h2>{_t(lang, 'h_method')}</h2>")
    S.append(f"<p>{_methodology_prose(design, model, lang)}</p>")
    S.append(f"<h3>{_t(lang, 'h_factors')}</h3><table><thead><tr>"
             f"<th>{_t(lang, 'th_factor')}</th><th>{_t(lang, 'th_type')}</th>"
             f"<th>{_t(lang, 'th_range')}</th></tr></thead><tbody>")
    if design.factors:
        for f in design.factors:
            d = f.to_dict()
            rng = (f"[{d['low']}, {d['high']}]" if d["type"] == "continuous"
                   else str(d["levels"]))
            S.append(f"<tr><td>{d['name']}</td><td>{d['type']}</td><td>{rng}</td></tr>")
    else:
        for c in design.matrix.columns:
            vals = sorted(map(str, pd.unique(design.matrix[c])))[:6]
            S.append(f"<tr><td>{c}</td><td>{_t(lang, 'coded')}</td><td>{', '.join(vals)}</td></tr>")
    S.append("</tbody></table>")
    if model is not None:
        S.append(f"<p class='gloss'>{_t(lang, 'model')}: <code>{model!r}</code></p>")

    # 3. design matrix (collapsible)
    S.append(f"<details><summary>{_t(lang, 'see_matrix').format(runs=design.n_runs)}</summary>")
    S.append(design.matrix.round(4).to_html(border=0))
    S.append("</details>")

    # 4. design quality
    S.append(f"<h2>{_t(lang, 'h_quality')}</h2>")
    if eff.get("rank_deficient"):
        S.append(f"<p class='red'>{_t(lang, 'rank_deficient_q')}</p>")
    else:
        S.append("<div class='cards'>")
        S.append(_card(_t(lang, "card_d"), eff["D_efficiency"], d_col))
        S.append(_card(_t(lang, "card_a"), eff["A_efficiency"], a_col))
        S.append(_card(_t(lang, "card_g"), eff["G_efficiency"], g_col))
        S.append("</div>")
        S.append(f"<p class='gloss'>{_t(lang, 'gloss_eff')}</p>")
        if "fds" in imgs or "fds" in failed:
            _img_or_placeholder(S, imgs, failed, "fds", lang, "FDS plot")
            if "fds" in imgs:
                S.append(f"<p class='gloss'>{_t(lang, 'gloss_fds')}</p>")
    if "power" in imgs or "power" in failed:
        S.append(f"<h3>{_t(lang, 'h_power')}</h3>")
        _img_or_placeholder(S, imgs, failed, "power", lang, "power")
        if "power" in imgs:
            S.append(f"<p class='gloss'>{_t(lang, 'gloss_power').format(target=thr['power_target'])}</p>")
    if len(ev.vif):
        vmax = float(np.nanmax(ev.vif.to_numpy()))
        vcol = "red" if vmax > thr["vif_warn"] else "green"
        S.append(f"<p><b>{_t(lang, 'vif_max')}:</b> <span class='{vcol}'>{vmax:.2f}</span> "
                 f"{_t(lang, 'vif_gloss')}</p>")
    if "alias" in imgs or "alias" in failed:
        S.append(f"<h3>{_t(lang, 'h_alias')}</h3>")
        _img_or_placeholder(S, imgs, failed, "alias", lang, "alias")
        if "alias" in imgs:
            S.append(f"<p class='gloss'>{_t(lang, 'gloss_alias')}</p>")

    # 5. results
    if fit is not None:
        S.append(f"<h2>{_t(lang, 'h_results')}</h2>")
        S.append("<p>" + _t(lang, "results_line").format(
            r2=fit.r_squared, r2adj=fit.r_squared_adj, s2=fit.sigma2, dof=fit.dof) + "</p>")
        sf = fit.summary_frame().copy()
        sf["sig"] = [_stars(p) for p in sf["p_value"]]
        for c in ("estimate", "std_error", "t_value", "p_value"):
            sf[c] = sf[c].map(lambda x: f"{x:.4g}")
        S.append(sf.to_html(index=False, border=0, escape=False))
        S.append(f"<p class='gloss'>{_t(lang, 'gloss_sig')}</p>")
        _img_or_placeholder(S, imgs, failed, "halfnormal", lang, "half-normal")

    # 6. anomalous values
    if fit is not None:
        S.append(f"<h2>{_t(lang, 'h_anom')}</h2>")
        an = fit.anomalies()
        if len(an) == 0:
            S.append(f"<p class='green'>{_t(lang, 'no_anom')}</p>")
        else:
            _img_or_placeholder(S, imgs, failed, "resid", lang, "residuals")
            S.append(an.to_html(index=False, border=0, classes="anom"))
            S.append(f"<p class='gloss'>{_t(lang, 'gloss_anom')}</p>")

    # 6.5 recommended design for this case
    rec = _recommendation_block(design, model)
    if rec is not None:
        S.append(f"<h2>{_t(lang, 'h_rec_case')}</h2>")
        color = "green" if rec["matches"] else "amber"
        note = (_t(lang, "rec_match").format(actual=rec["actual"]) if rec["matches"]
                else _t(lang, "rec_diff").format(method=rec["method"], actual=rec["actual"]))
        S.append(f"<p class='{color}'>{note}</p>")
        tb = rec["table"].rename(columns=_t(lang, "rec_cols"))
        S.append(tb.to_html(index=False, border=0))

    # 7. conclusions and recommendations
    S.append(f"<h2>{_t(lang, 'h_conclusions')}</h2><ul>")
    for r in _recommendations(design, eff, fit, thr, lang):
        S.append(f"<li>{r}</li>")
    S.append("</ul>")

    # footer
    S.append(f"<div class='foot'>{_t(lang, 'foot').format(ver=__version__)}</div>")
    S.append("</body></html>")
    return "".join(S)
