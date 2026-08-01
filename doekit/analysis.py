"""Analysis layer: OLS (blocks, robust SE), ANOVA, lack-of-fit and mixed models.

Built on ``numpy``/``scipy`` and ``statsmodels`` (central dependency from 0.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

from .model import Model
from .designs.base import Design


_VALID_COV = frozenset({"nonrobust", "HC0", "HC1", "HC3"})


def _jsonify(obj):
    """Convert nested numpy/pandas types to native JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _as_float_list(a) -> list:
    """JSON-safe list of floats (NaN/Inf -> None)."""
    out = []
    for v in np.asarray(a, dtype=float).reshape(-1):
        fv = float(v)
        out.append(None if (np.isnan(fv) or np.isinf(fv)) else fv)
    return out


# ---------------------------------------------------------------------------
# Fit results
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result of an OLS fit, with anomaly diagnostics.

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

def _blocking_column(design: Design) -> Optional[str]:
    """Return the block column name declared in ``design.metadata['blocking']``."""
    blocking = design.metadata.get("blocking")
    if blocking is None:
        return None
    if isinstance(blocking, str):
        return blocking
    if isinstance(blocking, dict):
        return blocking.get("column")
    return None


def _factor_frame(design: Design, drop: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Design matrix without non-factor columns (blocks / grouping)."""
    drop_set = set(drop or [])
    blk = _blocking_column(design)
    if blk:
        drop_set.add(blk)
    cols = [c for c in design.matrix.columns if c not in drop_set]
    return design.matrix.loc[:, cols].copy()


def _resolve_model(design: Design, model: Optional[Model],
                   drop: Optional[Sequence[str]] = None) -> Model:
    """Return ``model`` or fall back to main effects on non-block columns."""
    model = model or design.model
    if model is None:
        frame = _factor_frame(design, drop=drop)
        model = Model.main_effects(list(frame.columns))
    return model


def _resolve_blocks(design: Design, blocks) -> tuple[Optional[np.ndarray], Optional[str], list[str]]:
    """Resolve block labels.

    Parameters
    ----------
    blocks : None, False, str, or array-like
        Column name in ``design.matrix``, or a per-run label array.
        ``None`` falls back to ``design.metadata['blocking']``; ``False``
        forces an unblocked fit even if metadata declares a block column.

    Returns
    -------
    labels, name, drop_cols
        ``labels`` is ``None`` when unblocked; ``name`` is the column name or
        ``"array"``; ``drop_cols`` are matrix columns to exclude from the model.
        When metadata declares a block column but ``blocks is False``, that
        column is still dropped from the factor model (it is not a factor).
    """
    meta_col = _blocking_column(design)

    if blocks is False:
        # Explicit opt-out of fixed-block dummies; still exclude the block column
        # from the default factor frame so it is not treated as a regressor.
        return None, None, [meta_col] if meta_col else []

    if blocks is None:
        if meta_col is None:
            return None, None, []
        blocks = meta_col

    if isinstance(blocks, str):
        if blocks not in design.matrix.columns:
            raise ValueError(
                f"blocks column {blocks!r} not found in design.matrix columns "
                f"{list(design.matrix.columns)}"
            )
        labels = design.matrix[blocks].to_numpy()
        return labels, blocks, [blocks]

    labels = np.asarray(blocks)
    if labels.shape[0] != design.n_runs:
        raise ValueError(
            f"blocks array length ({labels.shape[0]}) must match "
            f"n_runs ({design.n_runs})"
        )
    return labels, "array", []


def _augment_blocks(X: np.ndarray, names: list[str], labels: np.ndarray
                    ) -> tuple[np.ndarray, list[str]]:
    """Append drop-first block dummies to ``X``."""
    levels = pd.unique(pd.Series(labels))
    if len(levels) < 2:
        raise ValueError(
            f"blocks must have at least 2 distinct levels (got {len(levels)})"
        )
    dummies = pd.get_dummies(pd.Series(labels), drop_first=True, dtype=float)
    dummies.columns = [f"block[{c}]" for c in dummies.columns]
    X2 = np.hstack([X, dummies.to_numpy(dtype=float)])
    names2 = list(names) + list(dummies.columns)
    return X2, names2


def attach_blocks(design: Design, blocks, name: str = "block") -> Design:
    """Return a copy of ``design`` with a block column and ``metadata['blocking']``.

    Parameters
    ----------
    design : Design
        Source design.
    blocks : array-like
        Per-run block labels (length ``n_runs``).
    name : str, default "block"
        Column name for the block factor.

    Returns
    -------
    Design
        New design with the block column appended (or overwritten).
    """
    labels = np.asarray(blocks)
    if labels.shape[0] != design.n_runs:
        raise ValueError(
            f"blocks length ({labels.shape[0]}) must match n_runs ({design.n_runs})"
        )
    mat = design.matrix.copy()
    mat[name] = labels
    meta = dict(design.metadata)
    meta["blocking"] = {"column": name, "n_blocks": int(len(pd.unique(pd.Series(labels))))}
    return Design(matrix=mat, factors=list(design.factors or []),
                  model=design.model, metadata=meta)


# ---------------------------------------------------------------------------
# OLS / ANOVA / LOF
# ---------------------------------------------------------------------------

def fit_linear_model(design: Design, response, model: Optional[Model] = None,
                     blocks=None, cov_type: str = "nonrobust",
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
        Block columns are excluded automatically from the default main-effects
        model.
    blocks : None, False, str or array-like, optional
        Fixed block factor: a column name in ``design.matrix``, a per-run label
        array, or ``None``. If ``None`` but ``design.metadata['blocking']``
        names a column, that column is used. Pass ``False`` to force an
        unblocked fit (the block column is still excluded from the factor model).
    cov_type : {"nonrobust", "HC0", "HC1", "HC3"}, default "nonrobust"
        Covariance estimator for standard errors (via statsmodels).
    report : None, bool, str, Path or dict, optional
        If not ``None``, a full HTML report is generated and its path is stored
        in ``fit.report_path``.

    Returns
    -------
    FitResult
        The fitted model with coefficients, statistics and anomaly diagnostics.

    Raises
    ------
    ValueError
        If ``cov_type`` is unknown, blocks are invalid, or the design is
        saturated / rank-deficient after adding blocks.
    """
    if cov_type not in _VALID_COV:
        raise ValueError(
            f"cov_type must be one of {sorted(_VALID_COV)}, got {cov_type!r}"
        )

    block_labels, block_name, drop = _resolve_blocks(design, blocks)
    model = _resolve_model(design, model, drop=drop)
    frame = _factor_frame(design, drop=drop)
    X = np.asarray(model.matrix(frame), dtype=float)
    names = list(model.column_names(frame))
    y = np.asarray(response, dtype=float).reshape(-1)

    if y.shape[0] != design.n_runs:
        raise ValueError(
            f"response length ({y.shape[0]}) must match n_runs ({design.n_runs})"
        )

    if block_labels is not None:
        X, names = _augment_blocks(X, names, block_labels)

    n, p = X.shape
    dof = n - p
    if dof < 0:
        raise ValueError(
            f"saturated/over-parameterized design after blocks: n={n}, p={p} "
            f"(need n >= p). Reduce the model or add runs."
        )
    if np.linalg.matrix_rank(X) < p:
        raise ValueError(
            "model matrix is rank deficient; check collinear terms or blocks."
        )

    ols = sm.OLS(y, X)
    if cov_type == "nonrobust":
        res = ols.fit()
    else:
        res = ols.fit(cov_type=cov_type)

    beta = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    tvals = np.asarray(res.tvalues, dtype=float)
    pvals = np.asarray(res.pvalues, dtype=float)
    fitted = np.asarray(res.fittedvalues, dtype=float)
    resid = np.asarray(res.resid, dtype=float)
    sigma2 = float(res.mse_resid) if dof > 0 else float("nan")
    r2 = float(res.rsquared)
    r2_adj = float(res.rsquared_adj) if dof > 0 else float("nan")

    # Influence diagnostics from the classical (nonrobust) hat matrix
    XtX_inv = np.linalg.pinv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    stud = np.full(n, np.nan)
    cooks = np.full(n, np.nan)
    if dof > 1 and np.isfinite(sigma2) and sigma2 > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            internal = resid / np.sqrt(sigma2 * np.clip(1 - leverage, 1e-12, None))
            denom = np.clip(dof - internal ** 2, 1e-12, None)
            stud = internal * np.sqrt((dof - 1) / denom)
            cooks = ((resid ** 2 / (p * sigma2))
                     * (leverage / np.clip((1 - leverage) ** 2, 1e-12, None)))

    fit = FitResult(
        names, beta, se, tvals, pvals, resid, sigma2, r2, dof,
        fitted=fitted, r_squared_adj=r2_adj, leverage=leverage,
        studentized_resid=stud, cooks_distance=cooks,
        cov_type=cov_type, blocks=block_name, _sm_result=res,
    )
    if report is not None:
        from .report import run_report_arg  # noqa: PLC0415
        fit.report_path = run_report_arg(design, response=response, model=model,
                                         report=report)
    return fit


def anova_table(fit: FitResult, typ: Union[int, str] = 2) -> pd.DataFrame:
    """Partial-F (Wald) ANOVA table for an OLS :class:`FitResult`.

    For single-degree-of-freedom terms (the usual coded DoE case) the partial
    F equals ``t^2`` and matches Type III tests. Multi-df categorical blocks
    appear as separate dummy rows (``block[...]``).

    Parameters
    ----------
    fit : FitResult
        Result of :func:`fit_linear_model`.
    typ : {2, 3, "II", "III"}, default 2
        Accepted for API compatibility; the table is a partial-F / Wald table
        (Type III for single-df terms). A formula-based statsmodels Type II/III
        table is used when the underlying result was fit with a formula.

    Returns
    -------
    DataFrame
        Columns ``term``, ``df``, ``F``, ``p_value`` (plus residual row).
    """
    # Prefer statsmodels formula ANOVA when available
    res = fit._sm_result
    if res is not None and getattr(res.model, "formula", None):
        from statsmodels.stats.anova import anova_lm  # noqa: PLC0415
        return anova_lm(res, typ=typ)

    rows = []
    for name, t, p in zip(fit.names, fit.tvalues, fit.pvalues):
        if name in ("(Intercept)", "Intercept"):
            continue
        f_stat = float(t ** 2) if np.isfinite(t) else float("nan")
        rows.append({
            "term": name,
            "df": 1,
            "F": f_stat,
            "p_value": float(p) if np.isfinite(p) else float("nan"),
        })
    rows.append({
        "term": "Residual",
        "df": int(fit.dof),
        "F": float("nan"),
        "p_value": float("nan"),
    })
    return pd.DataFrame(rows)


def lack_of_fit(design: Design, response, model: Optional[Model] = None,
                blocks=None) -> pd.DataFrame:
    """Lack-of-fit vs pure-error decomposition when replicate runs exist.

    Identical rows of the (non-block) factor matrix are treated as replicates.
    Pure-error SS is the within-replicate sum of squares; lack-of-fit SS is
    ``RSS - SS_PE``. An F-test compares LOF against pure error.

    Parameters
    ----------
    design : Design
        Executed design.
    response : array-like
        Measured response per run.
    model : Model, optional
        Model used for the fitted RSS; defaults as in :func:`fit_linear_model`.
    blocks : str or array-like, optional
        Optional fixed blocks (same semantics as :func:`fit_linear_model`).

    Returns
    -------
    DataFrame
        One-row-per-source table with columns ``source``, ``df``, ``ss``,
        ``ms``, ``F``, ``p_value``.

    Raises
    ------
    ValueError
        If there are no replicate groups (no pure-error degrees of freedom).
    """
    fit = fit_linear_model(design, response, model=model, blocks=blocks)
    _, _, drop = _resolve_blocks(design, blocks)
    frame = _factor_frame(design, drop=drop)
    y = np.asarray(response, dtype=float).reshape(-1)

    keys = pd.Series(
        [tuple(row) for row in frame.to_numpy()],
        name="run_key",
    )
    # pure error from within-replicate variance
    ss_pe = 0.0
    df_pe = 0
    for _, idx in keys.groupby(keys).groups.items():
        yi = y[list(idx)]
        if len(yi) < 2:
            continue
        ss_pe += float(((yi - yi.mean()) ** 2).sum())
        df_pe += len(yi) - 1

    if df_pe <= 0:
        raise ValueError(
            "lack_of_fit requires replicate runs (identical factor-level rows); "
            "none were found in the design matrix."
        )

    rss = float(fit.resid @ fit.resid)
    ss_lof = max(0.0, rss - ss_pe)
    df_lof = fit.dof - df_pe
    if df_lof < 0:
        # e.g. blocks absorbed df; clamp and warn via nan F
        df_lof = 0

    ms_pe = ss_pe / df_pe if df_pe > 0 else float("nan")
    ms_lof = ss_lof / df_lof if df_lof > 0 else float("nan")
    if df_lof > 0 and ms_pe > 0:
        f_stat = ms_lof / ms_pe
        p_val = float(stats.f.sf(f_stat, df_lof, df_pe))
    else:
        f_stat = float("nan")
        p_val = float("nan")

    return pd.DataFrame([
        {"source": "lack_of_fit", "df": df_lof, "ss": ss_lof, "ms": ms_lof,
         "F": f_stat, "p_value": p_val},
        {"source": "pure_error", "df": df_pe, "ss": ss_pe, "ms": ms_pe,
         "F": float("nan"), "p_value": float("nan")},
        {"source": "residual", "df": fit.dof, "ss": rss,
         "ms": rss / fit.dof if fit.dof > 0 else float("nan"),
         "F": float("nan"), "p_value": float("nan")},
    ])


# ---------------------------------------------------------------------------
# Mixed models
# ---------------------------------------------------------------------------

def fit_mixed_model(design: Design, response, groups,
                    model: Optional[Model] = None,
                    re_formula: str = "1",
                    method: str = "reml",
                    reml: Optional[bool] = None) -> MixedFitResult:
    """Fit a linear mixed model (random intercept / slopes via MixedLM).

    Parameters
    ----------
    design : Design
        Executed design.
    response : array-like
        Measured response per run.
    groups : str or array-like
        Grouping factor for random effects (column name or per-run labels).
        Typical use: batch, whole-plot, or hard-to-change factor.
    model : Model, optional
        Fixed-effects model; defaults as in :func:`fit_linear_model`.
    re_formula : str, default "1"
        Random-effects formula in statsmodels style (``"1"`` = random intercept).
    method : {"reml", "ml"}, default "reml"
        Estimation method (ignored if ``reml`` is given explicitly).
    reml : bool, optional
        If set, overrides ``method`` (``True`` -> REML, ``False`` -> ML).

    Returns
    -------
    MixedFitResult
        Fixed effects, variance components and fit diagnostics.

    Raises
    ------
    ValueError
        If groups are invalid (fewer than 2 levels, length mismatch, etc.).
    """
    if isinstance(groups, str):
        if groups not in design.matrix.columns:
            raise ValueError(
                f"groups column {groups!r} not found in design.matrix columns "
                f"{list(design.matrix.columns)}"
            )
        group_labels = design.matrix[groups].to_numpy()
        group_name = groups
        drop = [groups]
    else:
        group_labels = np.asarray(groups)
        if group_labels.shape[0] != design.n_runs:
            raise ValueError(
                f"groups array length ({group_labels.shape[0]}) must match "
                f"n_runs ({design.n_runs})"
            )
        group_name = "array"
        drop = []

    # also drop block column from fixed effects if present as metadata
    blk = _blocking_column(design)
    if blk and blk not in drop:
        drop = drop + [blk]

    n_groups = int(len(pd.unique(pd.Series(group_labels))))
    if n_groups < 2:
        raise ValueError(
            f"groups must have at least 2 distinct levels (got {n_groups})"
        )

    model = _resolve_model(design, model, drop=drop)
    frame = _factor_frame(design, drop=drop)
    X = np.asarray(model.matrix(frame), dtype=float)
    names = list(model.column_names(frame))
    y = np.asarray(response, dtype=float).reshape(-1)

    if y.shape[0] != design.n_runs:
        raise ValueError(
            f"response length ({y.shape[0]}) must match n_runs ({design.n_runs})"
        )
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"not enough runs for mixed model fixed effects: n={X.shape[0]}, "
            f"p={X.shape[1]}"
        )

    use_reml = reml if reml is not None else (method.lower() == "reml")
    method_label = "reml" if use_reml else "ml"

    exog_re = None
    if re_formula.strip() not in ("1", "1.0"):
        from patsy import dmatrix  # noqa: PLC0415
        exog_re = np.asarray(dmatrix(re_formula, frame), dtype=float)

    md = MixedLM(endog=y, exog=X, groups=group_labels, exog_re=exog_re)
    try:
        res = md.fit(reml=use_reml, method="lbfgs", maxiter=200, disp=False)
    except Exception:
        res = md.fit(reml=use_reml, disp=False)

    fe = np.asarray(res.fe_params, dtype=float)
    # bse_fe may be unavailable if singular; fall back to nan
    try:
        se = np.asarray(res.bse_fe, dtype=float)
    except Exception:
        se = np.full(len(fe), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        zvals = fe / se
        pvals = 2 * stats.norm.sf(np.abs(zvals))

    fitted = np.asarray(res.fittedvalues, dtype=float)
    resid = y - fitted
    sigma2 = float(res.scale)

    re_var = {}
    cov_re = getattr(res, "cov_re", None)
    if cov_re is not None:
        cov_re = np.atleast_2d(np.asarray(cov_re, dtype=float))
        if cov_re.shape == (1, 1):
            re_var["Intercept"] = float(cov_re[0, 0])
        else:
            for i in range(cov_re.shape[0]):
                re_var[f"re_{i}"] = float(cov_re[i, i])

    converged = bool(getattr(res, "converged", True))

    return MixedFitResult(
        names=names if len(names) == len(fe) else [f"x{i}" for i in range(len(fe))],
        coef=fe, se=se, zvalues=zvals, pvalues=pvals, resid=resid,
        sigma2=sigma2, groups=group_name, n_groups=n_groups, re_var=re_var,
        method=method_label, converged=converged,
        llf=float(res.llf), aic=float(res.aic), bic=float(res.bic),
        fitted=fitted, _sm_result=res,
    )


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
        frame = _factor_frame(design)
        model = Model.main_effects(list(frame.columns), intercept=False)
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
