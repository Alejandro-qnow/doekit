"""Fit result DTOs (OLS and mixed)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...shared.serialize import jsonify as _jsonify, as_float_list as _as_float_list
from ...domain.criteria.linalg import leverage as _leverage_rows

from dataclasses import dataclass, field


@dataclass
class FitResult:
    """Result of an OLS fit, with anomaly diagnostics.

    Container for coefficient estimates, inferential statistics and per-run
    influence measures. Serialize with :meth:`to_dict` / :meth:`from_dict` for
    handoff to reports and agents.

    Formulas
    --------
    - **R²:** ``1 - RSS / TSS`` where ``TSS = sum((y - ybar)^2)``.
    - **Adjusted R²:** ``1 - (1 - R²) * (N - 1) / dof``.
    - **Residual variance:** ``sigma2 = RSS / dof``.

    Attributes
    ----------
    names : list of str
        Column (term) names of the model matrix (including block dummies).
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
        Hat-matrix diagonal ``h_ii``.
    studentized_resid : ndarray, optional
        Externally studentized (deleted) residuals.
    cooks_distance : ndarray, optional
        Cook's distance (influence) of each run.
    cov_type : str
        Covariance estimator used (``nonrobust``, ``HC0``, ``HC1``, ``HC3``).
    blocks : str or None
        Block column name, ``"array"``, or ``None`` if unblocked.
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
    leverage: Optional[np.ndarray] = None
    studentized_resid: Optional[np.ndarray] = None
    cooks_distance: Optional[np.ndarray] = None
    cov_type: str = "nonrobust"
    blocks: Optional[str] = None
    _sm_result: object = field(default=None, repr=False, compare=False)

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
        """Runs flagged as atypical or influential, with the reason.

        Combines three standard influence rules tuned for small DoE designs.
        Returns an empty frame when leverage / studentized residuals were not
        computed.

        Formulas
        --------
        - **Outlier:** ``|r_i^*| > 3``.
        - **High leverage:** ``h_ii > 2p / N``.
        - **Influential:** ``Cook's D_i > 1`` (Cook & Weisberg absolute cutoff;
          the ``4/N`` rule over-flags deliberately high-leverage DoE points).

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

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (``schema: doekit.FitResult/1``)."""
        return _jsonify({
            "schema": "doekit.FitResult/1",
            "kind": "ols",
            "names": list(self.names),
            "coef": _as_float_list(self.coef),
            "se": _as_float_list(self.se),
            "tvalues": _as_float_list(self.tvalues),
            "pvalues": _as_float_list(self.pvalues),
            "resid": _as_float_list(self.resid),
            "fitted": _as_float_list(self.fitted) if self.fitted is not None else None,
            "sigma2": self.sigma2,
            "r_squared": self.r_squared,
            "r_squared_adj": self.r_squared_adj,
            "dof": int(self.dof),
            "cov_type": self.cov_type,
            "blocks": self.blocks,
            "leverage": _as_float_list(self.leverage) if self.leverage is not None else None,
            "studentized_resid": (
                _as_float_list(self.studentized_resid)
                if self.studentized_resid is not None else None
            ),
            "cooks_distance": (
                _as_float_list(self.cooks_distance)
                if self.cooks_distance is not None else None
            ),
            "anomalies": self.anomalies().to_dict("records"),
        })

    @classmethod
    def from_dict(cls, d: dict) -> "FitResult":
        """Rebuild a :class:`FitResult` from :meth:`to_dict` output."""
        if d.get("schema") not in (None, "doekit.FitResult/1"):
            raise ValueError(f"unsupported FitResult schema: {d.get('schema')!r}")
        if d.get("kind", "ols") != "ols":
            raise ValueError(f"expected kind='ols', got {d.get('kind')!r}")

        def _arr(key, default=None):
            v = d.get(key, default)
            if v is None:
                return None
            return np.asarray(v, dtype=float)

        return cls(
            names=list(d["names"]),
            coef=_arr("coef"),
            se=_arr("se"),
            tvalues=_arr("tvalues"),
            pvalues=_arr("pvalues"),
            resid=_arr("resid"),
            sigma2=float(d["sigma2"]) if d.get("sigma2") is not None else float("nan"),
            r_squared=float(d["r_squared"]) if d.get("r_squared") is not None else float("nan"),
            dof=int(d["dof"]),
            fitted=_arr("fitted"),
            r_squared_adj=(
                float(d["r_squared_adj"])
                if d.get("r_squared_adj") is not None else float("nan")
            ),
            leverage=_arr("leverage"),
            studentized_resid=_arr("studentized_resid"),
            cooks_distance=_arr("cooks_distance"),
            cov_type=d.get("cov_type", "nonrobust"),
            blocks=d.get("blocks"),
        )

    def __repr__(self) -> str:
        with pd.option_context("display.float_format", lambda x: f"{x:.5g}"):
            body = repr(self.summary_frame())
        blk = f", blocks={self.blocks!r}" if self.blocks else ""
        return (f"FitResult(R^2={self.r_squared:.4f}, dof={self.dof}, "
                f"cov_type={self.cov_type!r}{blk})\n{body}")


@dataclass

class MixedFitResult:
    """Result of a linear mixed model (REML / ML) fit.

    Fixed-effect estimates with Wald statistics plus random-effect variance
    components from statsmodels ``MixedLM``.

    Formulas
    --------
    - **Wald z:** ``z_j = beta_j / SE(beta_j)``, ``p = 2 * Phi(-|z_j|)``.
    - **Information criteria:** AIC/BIC from the fitted mixed-model likelihood.

    Attributes
    ----------
    names : list of str
        Fixed-effect term names.
    coef, se, zvalues, pvalues : ndarray
        Fixed-effect estimates and Wald statistics.
    resid : ndarray
        Residuals (response minus fixed+random fitted values when available).
    sigma2 : float
        Residual (within-group) variance.
    groups : str
        Name of the grouping factor (or ``"array"``).
    n_groups : int
        Number of groups.
    re_var : dict
        Random-effect variance components (label -> variance).
    method : {"reml", "ml"}
        Estimation method.
    converged : bool
        Whether the optimizer reported convergence.
    llf, aic, bic : float
        Log-likelihood and information criteria.
    fitted : ndarray, optional
        Fitted values.
    """

    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    zvalues: np.ndarray
    pvalues: np.ndarray
    resid: np.ndarray
    sigma2: float
    groups: str
    n_groups: int
    re_var: dict
    method: str = "reml"
    converged: bool = True
    llf: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    fitted: Optional[np.ndarray] = None
    _sm_result: object = field(default=None, repr=False, compare=False)

    def summary_frame(self) -> pd.DataFrame:
        """Return a fixed-effects coefficient table."""
        return pd.DataFrame({
            "term": self.names,
            "estimate": self.coef,
            "std_error": self.se,
            "z_value": self.zvalues,
            "p_value": self.pvalues,
        })

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (``schema: doekit.MixedFitResult/1``)."""
        return _jsonify({
            "schema": "doekit.MixedFitResult/1",
            "kind": "mixed",
            "names": list(self.names),
            "coef": _as_float_list(self.coef),
            "se": _as_float_list(self.se),
            "zvalues": _as_float_list(self.zvalues),
            "pvalues": _as_float_list(self.pvalues),
            "resid": _as_float_list(self.resid),
            "fitted": _as_float_list(self.fitted) if self.fitted is not None else None,
            "sigma2": self.sigma2,
            "groups": self.groups,
            "n_groups": int(self.n_groups),
            "re_var": dict(self.re_var),
            "method": self.method,
            "converged": bool(self.converged),
            "llf": self.llf,
            "aic": self.aic,
            "bic": self.bic,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "MixedFitResult":
        """Rebuild a :class:`MixedFitResult` from :meth:`to_dict` output."""
        if d.get("schema") not in (None, "doekit.MixedFitResult/1"):
            raise ValueError(f"unsupported MixedFitResult schema: {d.get('schema')!r}")

        def _arr(key):
            v = d.get(key)
            return None if v is None else np.asarray(v, dtype=float)

        return cls(
            names=list(d["names"]),
            coef=_arr("coef"),
            se=_arr("se"),
            zvalues=_arr("zvalues"),
            pvalues=_arr("pvalues"),
            resid=_arr("resid"),
            sigma2=float(d["sigma2"]) if d.get("sigma2") is not None else float("nan"),
            groups=d["groups"],
            n_groups=int(d["n_groups"]),
            re_var=dict(d.get("re_var") or {}),
            method=d.get("method", "reml"),
            converged=bool(d.get("converged", True)),
            llf=float(d["llf"]) if d.get("llf") is not None else float("nan"),
            aic=float(d["aic"]) if d.get("aic") is not None else float("nan"),
            bic=float(d["bic"]) if d.get("bic") is not None else float("nan"),
            fitted=_arr("fitted"),
        )

    def __repr__(self) -> str:
        with pd.option_context("display.float_format", lambda x: f"{x:.5g}"):
            body = repr(self.summary_frame())
        return (f"MixedFitResult(method={self.method!r}, groups={self.groups!r}, "
                f"n_groups={self.n_groups}, sigma2={self.sigma2:.5g})\n{body}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

