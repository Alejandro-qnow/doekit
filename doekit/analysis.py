"""Analysis layer: linear fit, main effects and half-normal data.

Go from a design to a fitted model using only numpy/scipy (no GLM/statsmodels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .model import Model
from .designs.base import Design


@dataclass
class FitResult:
    """Result of an OLS fit, with anomaly diagnostics.

    Attributes
    ----------
    names : list of str
        Column (term) names of the model matrix.
    coef, se, tvalues, pvalues : ndarray
        Coefficient estimates, standard errors, t statistics and p-values.
    resid : ndarray
        Residuals ``y - fitted``.
    sigma2 : float
        Residual variance estimate ``RSS / dof``.
    r_squared, r_squared_adj : float
        Coefficient of determination and its adjusted version.
    dof : int
        Residual degrees of freedom ``N - p``.
    fitted : ndarray, optional
        Fitted values.
    leverage : ndarray, optional
        Hat-matrix diagonal ``h_ii = diag(X (X'X)^-1 X')``.
    studentized_resid : ndarray, optional
        Externally studentized (deleted) residuals.
    cooks_distance : ndarray, optional
        Cook's distance (influence) of each run.
    """

    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    resid: np.ndarray
    sigma2: float
    r_squared: float
    dof: int
    fitted: Optional[np.ndarray] = None
    r_squared_adj: float = float("nan")
    leverage: Optional[np.ndarray] = None            # h_ii = diag(X (X'X)^-1 X')
    studentized_resid: Optional[np.ndarray] = None   # deleted studentized residual
    cooks_distance: Optional[np.ndarray] = None       # influence of each run

    def summary_frame(self) -> pd.DataFrame:
        """Return a coefficient table (term, estimate, std_error, t/p-value)."""
        return pd.DataFrame({
            "term": self.names,
            "estimate": self.coef,
            "std_error": self.se,
            "t_value": self.tvalues,
            "p_value": self.pvalues,
        })

    def anomalies(self) -> pd.DataFrame:
        """Runs flagged as atypical / influential, with the reason.

        Rules: an *outlier* when ``|studentized residual| > 3``; *high leverage*
        when ``h_ii > 2p/N``; *influential* when ``Cook's D > 1`` (Cook &
        Weisberg absolute cutoff, robust for small designs — the ``4/N`` rule
        over-flags in DoE, where points are deliberately high-leverage).

        Returns
        -------
        DataFrame
            Columns ``run``, ``reason``, ``studentized_resid``, ``leverage``,
            ``cooks_distance`` (empty if diagnostics are unavailable).
        """
        n = len(self.resid)
        p = n - self.dof
        rows = []
        if self.leverage is None or self.studentized_resid is None:
            return pd.DataFrame(columns=["run", "reason", "studentized_resid",
                                         "leverage", "cooks_distance"])
        lev_cut = 2.0 * p / n
        cook_cut = 1.0
        for i in range(n):
            reasons = []
            if abs(self.studentized_resid[i]) > 3.0:
                reasons.append("outlier")
            if self.leverage[i] > lev_cut:
                reasons.append("high leverage")
            if self.cooks_distance is not None and self.cooks_distance[i] > cook_cut:
                reasons.append("influential")
            if reasons:
                rows.append({
                    "run": i,
                    "reason": ", ".join(reasons),
                    "studentized_resid": round(float(self.studentized_resid[i]), 3),
                    "leverage": round(float(self.leverage[i]), 3),
                    "cooks_distance": round(float(self.cooks_distance[i]), 3)
                    if self.cooks_distance is not None else None,
                })
        return pd.DataFrame(rows, columns=["run", "reason", "studentized_resid",
                                           "leverage", "cooks_distance"])

    def __repr__(self) -> str:
        with pd.option_context("display.float_format", lambda x: f"{x:.5g}"):
            body = repr(self.summary_frame())
        return f"FitResult(R^2={self.r_squared:.4f}, dof={self.dof})\n{body}"


def _resolve_model(design: Design, model: Optional[Model]) -> Model:
    """Return ``model`` or fall back to ``design.model`` / a main-effects model."""
    model = model or design.model
    if model is None:
        model = Model.main_effects(list(design.matrix.columns))
    return model


def fit_linear_model(design: Design, response, model: Optional[Model] = None,
                     report=None) -> FitResult:
    """Fit an OLS linear model to a design's responses.

    Parameters
    ----------
    design : Design
        The executed design providing the factor levels.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model to fit; taken from ``model`` or ``design.model`` if omitted.
    report : None, bool, str, Path or dict, optional
        If not ``None`` (a folder, ``True`` or an options ``dict``), a full HTML
        report is generated and its path is stored in ``fit.report_path``.

    Returns
    -------
    FitResult
        The fitted model with coefficients, statistics and anomaly diagnostics.
    """
    model = _resolve_model(design, model)
    X = model.matrix(design.matrix)
    y = np.asarray(response, dtype=float).reshape(-1)
    names = model.column_names(design.matrix)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    dof = n - p
    rss = float(resid @ resid)
    sigma2 = rss / dof if dof > 0 else np.nan

    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2) if dof > 0 else np.full(p, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
        pvals = 2 * stats.t.sf(np.abs(tvals), dof) if dof > 0 else np.full(p, np.nan)

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 and not np.isnan(r2) else np.nan

    # --- anomaly diagnostics ---
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # h_ii
    stud = np.full(n, np.nan)
    cooks = np.full(n, np.nan)
    if dof > 1 and sigma2 > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            internal = resid / np.sqrt(sigma2 * np.clip(1 - leverage, 1e-12, None))
            # externally studentized (deleted): follows a t with dof-1 d.f.
            denom = np.clip(dof - internal ** 2, 1e-12, None)
            stud = internal * np.sqrt((dof - 1) / denom)
            cooks = (resid ** 2 / (p * sigma2)) * (leverage / np.clip((1 - leverage) ** 2, 1e-12, None))

    fit = FitResult(names, beta, se, tvals, pvals, resid, sigma2, r2, dof,
                    fitted=fitted, r_squared_adj=r2_adj, leverage=leverage,
                    studentized_resid=stud, cooks_distance=cooks)
    if report is not None:
        from .report import run_report_arg  # noqa: PLC0415
        fit.report_path = run_report_arg(design, response=response, model=model,
                                         report=report)
    return fit


def main_effects(design: Design, response, model: Optional[Model] = None,
                 scale: str = "coefficient") -> pd.Series:
    """Estimated main effects of a screening design.

    Parameters
    ----------
    design : Design
        The executed design.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model to fit; a main-effects model (no intercept) is used if omitted.
    scale : {"coefficient", "effect"}, default "coefficient"
        Magnitude convention (the relative *ordering* is the same either way, so
        significance detection is unchanged):

        - ``"coefficient"``: the regression coefficients ``beta``.
        - ``"effect"``: the **classical DoE effect** (Montgomery), defined as
          ``mean(+1) - mean(-1)``. For an orthogonal factor in ``+/-1`` this
          equals ``2*beta``. This is the magnitude the screening literature plots
          in the half-normal plot.

    Returns
    -------
    pandas.Series
        Effects indexed by term name.

    Raises
    ------
    ValueError
        If ``scale`` is neither ``"coefficient"`` nor ``"effect"``.
    """
    if model is None:
        factor_cols = [c for c in design.matrix.columns]
        model = Model.main_effects(factor_cols, intercept=False)
    fit = fit_linear_model(design, response, model=model)
    coef = fit.coef
    if scale == "effect":
        coef = 2.0 * coef
    elif scale != "coefficient":
        raise ValueError("scale must be 'coefficient' or 'effect'")
    return pd.Series(coef, index=fit.names, name=scale)


def half_normal_data(effects, labels: Optional[Sequence[str]] = None):
    """Half-normal quantiles vs. effect magnitude, sorted.

    Parameters
    ----------
    effects : array-like
        Estimated effects (or coefficients).
    labels : sequence of str, optional
        Labels for each effect; defaults to ``e1..em``.

    Returns
    -------
    DataFrame
        Columns ``label``, ``abs_effect`` and ``half_normal_quantile``, ready to
        plot (see :mod:`doekit.plotting`).
    """
    eff = np.asarray(effects, dtype=float)
    if labels is None:
        labels = [f"e{i + 1}" for i in range(len(eff))]
    abs_eff = np.abs(eff)
    order = np.argsort(abs_eff)
    m = len(eff)
    ranks = np.arange(1, m + 1)
    quantiles = stats.norm.ppf(0.5 + 0.5 * (ranks - 0.5) / m)
    return pd.DataFrame({
        "label": np.asarray(labels)[order],
        "abs_effect": abs_eff[order],
        "half_normal_quantile": quantiles,
    })
