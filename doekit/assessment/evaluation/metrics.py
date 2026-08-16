"""Design evaluation and *benchmarking*: the quality report card of a DoE.

This is ``doekit``'s differentiator: whereas most Python libraries only
**build** designs, here we **evaluate** them against the theoretical optimum,
answering the experimenter's question: *"how far is my design from the right
design?"*.

Every metric is computed in **coded units** (``+/-1``). This is essential:
efficiencies are scale-invariant only when the factors are coded; measuring them
in natural units (e.g. 20-80 C) distorts them. If the ``Design`` carries
continuous/discrete factors, their columns are coded automatically before
building the model matrix ``X``.

Metrics (references: Montgomery, *Design and Analysis of Experiments*; Atkinson,
Donev & Tobias, *Optimum Experimental Designs*; SAS OPTEX and JMP docs):

- **D/A/G-efficiency (%)**: efficiency relative to the optimum. 100% = ideal
  design (orthogonal for D/A). ``> 90%`` is considered excellent.
- **Scaled prediction variance (SPV)** and its distribution over the region
  (basis of the **FDS plot**).
- **Power analysis**: probability of detecting an effect of a given size per
  coefficient (non-central t).
- **Alias matrix**: bias ``E[beta_hat] = beta + A*beta_2`` induced by terms
  omitted from the model.
- **VIF**: variance inflation factors (multicollinearity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ...domain.model import Model
from ...domain.design import Design
from ...domain.criteria.linalg import leverage as _leverage
from ...domain.region import region_from_design
from ..units import coded_frame as _coded_frame, resolve_model as _resolve_model


def _region_frame(design: Design, model: Model, n_region: int,
                  seed: Optional[int]) -> pd.DataFrame:
    """Sample the experimental region in coded units via :class:`HypercubeRegion`."""
    rng = np.random.default_rng(seed)
    return region_from_design(design, model).sample(n_region, rng=rng)


# --- individual metrics ------------------------------------------------------

def efficiencies(design: Design, model: Optional[Model] = None,
                 region: Optional[pd.DataFrame] = None, n_region: int = 20000,
                 seed: Optional[int] = None) -> dict:
    """D/A/G efficiencies (%) and scaled-prediction-variance statistics.

    Computed in **coded units** (``+/-1``). Values near 100% indicate a design
    close to the theoretical optimum for the fitted model. Mean SPV is the
    I-optimality criterion (smaller is better).

    Formulas
    --------
    - **D-eff** ``= 100 * det(X'X)^(1/p) / N`` (100% for an orthogonal design).
    - **A-eff** ``= 100 * p / (N * tr((X'X)^-1))``.
    - **G-eff** ``= 100 * p / max(SPV)`` over the region, with
      ``SPV(x) = N * x'(X'X)^-1 x`` (general equivalence theorem: ``max SPV >= p``).
    - **I** = mean SPV over the region; also report min/max SPV.

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Model whose matrix is scored; resolved from the design if omitted.
    region : DataFrame, optional
        Precomputed region sample; a fresh Monte Carlo sample is drawn if omitted.
    n_region : int, default 20000
        Region sample size when ``region`` is not provided.
    seed : int, optional
        Seed for the region sampling.

    Returns
    -------
    dict
        Keys ``n_runs``, ``n_params``, ``dof``, ``rank_deficient``,
        ``D_efficiency``, ``A_efficiency``, ``G_efficiency`` and
        ``spv_min``/``spv_mean``/``spv_max``. Efficiencies are ``nan`` when the
        design is rank deficient.

    Notes
    -----
    D/A are clamped to ``<= 100%`` (Hadamard / AM-HM). G may slightly exceed
    100% under Monte Carlo region sampling and is likewise clamped.

    Examples
    --------
    >>> import doekit as ed
    >>> eff = ed.efficiencies(ed.full_factorial(3), seed=0)
    >>> eff["D_efficiency"] > 90 and not eff["rank_deficient"]
    True
    """
    model = _resolve_model(design, model)
    coded = _coded_frame(design)
    X = model.matrix(coded)
    n, p = X.shape
    M = X.T @ X
    rank = int(np.linalg.matrix_rank(M))
    rank_deficient = rank < p or n < p

    reg = region if region is not None else _region_frame(design, model, n_region, seed)
    Xr = model.matrix(reg)

    if rank_deficient:
        # The design cannot estimate the model (saturated / supersaturated):
        # relative efficiencies are undefined.
        nan = float("nan")
        return {
            "n_runs": n, "n_params": p, "dof": n - p, "rank_deficient": True,
            "D_efficiency": nan, "A_efficiency": nan, "G_efficiency": nan,
            "spv_min": nan, "spv_mean": nan, "spv_max": nan,
        }

    sign, logdet = np.linalg.slogdet(M)
    d_eff = 100.0 * float(np.exp(logdet / p)) / n if sign > 0 else 0.0

    Minv = np.linalg.inv(M)
    tr = float(np.trace(Minv))
    a_eff = 100.0 * p / (n * tr) if tr > 0 else 0.0

    spv = n * _leverage(Xr, Minv)
    spv_max = float(spv.max())
    g_eff = 100.0 * p / spv_max if spv_max > 0 else 0.0

    # Theoretical bounds: D/A <= 100% (Hadamard/AM-HM), G <= 100% (max SPV >= p).
    # Monte Carlo sampling of the region can slightly exceed G -> clamp.
    d_eff, a_eff, g_eff = (min(100.0, d_eff), min(100.0, a_eff), min(100.0, g_eff))

    return {
        "n_runs": n,
        "n_params": p,
        "dof": n - p,
        "rank_deficient": False,
        "D_efficiency": d_eff,
        "A_efficiency": a_eff,
        "G_efficiency": g_eff,
        "spv_min": float(spv.min()),
        "spv_mean": float(spv.mean()),   # I-optimality (smaller is better)
        "spv_max": spv_max,
    }


def power_analysis(design: Design, model: Optional[Model] = None,
                   effect_size=1.0, sigma: float = 1.0, alpha: float = 0.05
                   ) -> pd.Series:
    """Per-coefficient power to detect an effect of size ``effect_size``.

    Uses a two-sided t-test at level ``alpha`` with ``N - p`` residual degrees
    of freedom. Requires an unsaturated design (``N > p``).

    Formulas
    --------
    - ``se_i = sigma * sqrt((X'X)^-1_ii)``
    - non-centrality ``delta_i = effect_size / se_i``
    - power from the non-central ``t`` distribution (normal fallback in the
      far tails when SciPy's ``nct`` is numerically unstable)

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Model whose terms are powered; resolved from the design if omitted.
    effect_size : float or array-like, default 1.0
        Anticipated effect size (scalar shared by all terms, or one per term).
    sigma : float, default 1.0
        Noise standard deviation.
    alpha : float, default 0.05
        Two-sided significance level.

    Returns
    -------
    pandas.Series
        Power per model term (index = term names), values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``N <= p`` (a saturated design has no residual degrees of freedom).

    Examples
    --------
    >>> import doekit as ed
    >>> pw = ed.power_analysis(ed.full_factorial(3), effect_size=2.0, sigma=1.0)
    >>> float(pw.min()) > 0.8
    True
    """
    model = _resolve_model(design, model)
    coded = _coded_frame(design)
    X = model.matrix(coded)
    names = model.column_names(coded)
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        raise ValueError(f"power_analysis requires N > p (N={n}, p={p}); saturated design")

    Minv = np.linalg.pinv(X.T @ X)
    se_unit = np.sqrt(np.clip(np.diag(Minv), 0, None))  # se per unit of sigma
    eff = np.broadcast_to(np.asarray(effect_size, dtype=float), (p,))
    with np.errstate(divide="ignore", invalid="ignore"):
        ncp = np.where(se_unit > 0, eff / (sigma * se_unit), np.inf)
    tcrit = stats.t.ppf(1 - alpha / 2, dof)

    # Two-sided power = upper + lower non-central-t tails. SciPy's nct is
    # numerically unstable in the far tails for large non-centrality (exactly
    # the well-powered orthogonal designs experimenters want): lower cdf can
    # return NaN erratically above ~9, and both tails return NaN when ncp is
    # inf (se_unit -> 0). In that regime power is saturated and the normal
    # approximation is exact for practical purposes.
    with np.errstate(invalid="ignore"):
        upper = stats.nct.sf(tcrit, dof, ncp)
        lower = stats.nct.cdf(-tcrit, dof, ncp)
        power = upper + lower
    degenerado = ~np.isfinite(power)
    if np.any(degenerado):
        aproximada = stats.norm.sf(tcrit - ncp) + stats.norm.cdf(-tcrit - ncp)
        power = np.where(degenerado, aproximada, power)
    power = np.clip(power, 0.0, 1.0)
    if not np.all(np.isfinite(power)):
        raise RuntimeError(
            "power_analysis produced non-finite values after fallback; "
            "this is a bug — please report it"
        )
    return pd.Series(power, index=names, name="power")


def vif(design: Design, model: Optional[Model] = None) -> pd.Series:
    """Variance inflation factors of the model terms (intercept excluded).

    ``VIF = 1`` indicates orthogonality; values ``> 5-10`` signal serious
    multicollinearity. Perfect collinearity yields ``inf`` (not a misleading
    finite value from a pseudoinverse).

    Formulas
    --------
    ``VIF_j = 1 / (1 - R2_j)`` where ``R2_j`` comes from regressing column
    ``j`` on the other non-constant columns (after centering/scaling).

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Model whose terms are checked; resolved from the design if omitted.

    Returns
    -------
    pandas.Series
        VIF per non-intercept term (empty if the model has no such terms).

    Examples
    --------
    >>> import doekit as ed
    >>> v = ed.vif(ed.full_factorial(3))
    >>> float(v.max()) < 1.01
    True
    """
    model = _resolve_model(design, model)
    coded = _coded_frame(design)
    X = model.matrix(coded)
    names = model.column_names(coded)
    # drop the intercept (constant) column
    keep = [i for i, nm in enumerate(names) if nm != "(Intercept)"]
    Xk = X[:, keep]
    kept_names = [names[i] for i in keep]
    if Xk.shape[1] == 0:
        return pd.Series(dtype=float, name="vif")

    std = Xk.std(axis=0)
    const = std == 0
    Xc = (Xk - Xk.mean(axis=0))
    Xc[:, ~const] = Xc[:, ~const] / std[~const]
    # VIF_j = 1/(1-R²_j) from regressing column j on the others.
    # Do NOT use diag(pinv(corr)): when corr is singular (perfect collinearity)
    # the pseudoinverse returns values < 1 (e.g. 0.25), which looks "excellent"
    # (VIF≈1) instead of infinite multicollinearity.
    vifs = np.empty(Xk.shape[1], dtype=float)
    for j in range(Xk.shape[1]):
        if const[j]:
            vifs[j] = np.nan
            continue
        others = (~const) & (np.arange(Xk.shape[1]) != j)
        y = Xc[:, j]
        if not np.any(others):
            vifs[j] = 1.0
            continue
        beta, _, _, _ = np.linalg.lstsq(Xc[:, others], y, rcond=None)
        resid = y - Xc[:, others] @ beta
        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(y, y))
        if ss_tot <= 0.0:
            vifs[j] = np.nan
            continue
        r2 = 1.0 - ss_res / ss_tot
        vifs[j] = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
    return pd.Series(vifs, index=kept_names, name="vif")


def alias_matrix(design: Design, model: Optional[Model] = None,
                 alias_model: Optional[Model] = None) -> pd.DataFrame:
    """Alias matrix relating primary-model bias to omitted terms.

    Quantifies the bias of the primary-model coefficients (``X1``) induced by
    potential omitted terms (``X2``). If ``alias_model`` is ``None``, the
    two-factor interactions not already in the primary model are used.

    Formulas
    --------
    - ``A = (X1'X1)^-1 X1'X2``
    - ``E[beta_hat_1] = beta_1 + A beta_2``

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Primary model ``X1``; resolved from the design if omitted.
    alias_model : Model, optional
        Model of potentially omitted terms ``X2``; defaults to the 2-factor
        interactions absent from the primary model.

    Returns
    -------
    DataFrame
        Alias matrix indexed by primary terms and columned by alias terms.

    Examples
    --------
    >>> import doekit as ed
    >>> A = ed.alias_matrix(ed.fractional_factorial(3, generators=["C=AB"]))
    >>> A.shape[0] > 0 and A.shape[1] > 0
    True
    """
    model = _resolve_model(design, model)
    coded = _coded_frame(design)
    if alias_model is None:
        alias_model = _default_two_factor_alias_model(model)

    X1 = model.matrix(coded)
    X2 = alias_model.matrix(coded)
    A = np.linalg.pinv(X1.T @ X1) @ X1.T @ X2
    return pd.DataFrame(A, index=model.column_names(coded),
                        columns=alias_model.column_names(coded))


def _default_two_factor_alias_model(model: Model) -> Model:
    """Build the model of 2-factor interactions absent from ``model`` (alias basis)."""
    from itertools import combinations
    from ...domain.model import Interaction
    names = model.factor_names
    existing = {t.label() for t in model.terms}
    terms = [Interaction((a, b)) for a, b in combinations(names, 2)
             if f"{a}:{b}" not in existing and f"{b}:{a}" not in existing]
    return Model.from_terms(terms, intercept=False)


def fds_data(design: Design, model: Optional[Model] = None,
             region: Optional[pd.DataFrame] = None, n_region: int = 20000,
             seed: Optional[int] = None) -> pd.DataFrame:
    """Data for the **Fraction of Design Space (FDS) plot**.

    Sorts SPV over a Monte Carlo sample of the experimental region. A flat,
    low curve means uniform and precise prediction across the whole region
    (the desirable case). Rank-deficient designs return ``spv`` as ``nan``.

    Formulas
    --------
    ``SPV(x) = N * x'(X'X)^-1 x``, sorted ascending; ``fraction`` is the
    empirical CDF abscissa ``(i - 0.5) / m`` for ``i = 1..m``.

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Model whose matrix is scored; resolved from the design if omitted.
    region : DataFrame, optional
        Precomputed region sample; a fresh sample is drawn if omitted.
    n_region : int, default 20000
        Region sample size when ``region`` is not provided.
    seed : int, optional
        Seed for the region sampling.

    Returns
    -------
    DataFrame
        Columns ``fraction`` (0→1) and ``spv``.

    Examples
    --------
    >>> import doekit as ed
    >>> fds = ed.fds_data(ed.full_factorial(3), n_region=500, seed=0)
    >>> list(fds.columns) == ["fraction", "spv"]
    True
    """
    model = _resolve_model(design, model)
    coded = _coded_frame(design)
    X = model.matrix(coded)
    n, p = X.shape
    M = X.T @ X
    rank = int(np.linalg.matrix_rank(M))
    reg = region if region is not None else _region_frame(design, model, n_region, seed)
    m = len(reg)
    fraction = (np.arange(1, m + 1) - 0.5) / m
    # Same gate as efficiencies: pinv would invent a finite SPV for a
    # non-estimable model and paint a reassuring FDS curve.
    if rank < p or n < p:
        return pd.DataFrame({"fraction": fraction, "spv": np.full(m, np.nan)})
    Minv = np.linalg.inv(M)
    Xr = model.matrix(reg)
    spv = np.sort(n * _leverage(Xr, Minv))
    return pd.DataFrame({"fraction": fraction, "spv": spv})


# --- aggregate report --------------------------------------------------------

@dataclass
class DesignEvaluation:
    """Quality report card of a design (result of :func:`evaluate`).

    Aggregates efficiencies, power, VIF and FDS into one object. Print or call
    :meth:`summary` for a human-readable card; use :meth:`to_dict` to persist.

    Attributes
    ----------
    n_runs, n_params, dof : int
        Number of runs, model parameters and residual degrees of freedom.
    efficiencies : dict
        Output of :func:`efficiencies`.
    power : pandas.Series
        Output of :func:`power_analysis` (empty for a saturated design).
    vif : pandas.Series
        Output of :func:`vif`.
    fds : DataFrame
        Output of :func:`fds_data`.

    Examples
    --------
    >>> import doekit as ed
    >>> ev = ed.evaluate(ed.full_factorial(3), n_region=500, seed=0)
    >>> ev.d_efficiency > 90
    True
    """

    n_runs: int
    n_params: int
    dof: int
    efficiencies: dict
    power: pd.Series
    vif: pd.Series
    fds: pd.DataFrame = field(repr=False)

    @property
    def d_efficiency(self) -> float:
        """D-efficiency (%)."""
        return self.efficiencies["D_efficiency"]

    @property
    def a_efficiency(self) -> float:
        """A-efficiency (%)."""
        return self.efficiencies["A_efficiency"]

    @property
    def g_efficiency(self) -> float:
        """G-efficiency (%)."""
        return self.efficiencies["G_efficiency"]

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the evaluation."""
        e = self.efficiencies
        lines = [
            f"Design evaluation  ({self.n_runs} runs, {self.n_params} params, dof={self.dof})",
            "-" * 56,
        ]
        if e.get("rank_deficient"):
            lines.append("  [!] Saturated/supersaturated design: cannot estimate the model.")
            lines.append("      Reduce the model or add runs. Efficiencies undefined.")
            return "\n".join(lines)
        lines += [
            f"  D-efficiency : {e['D_efficiency']:6.1f} %",
            f"  A-efficiency : {e['A_efficiency']:6.1f} %",
            f"  G-efficiency : {e['G_efficiency']:6.1f} %",
            f"  SPV (scaled prediction variance) over region:",
            f"      min={e['spv_min']:.3f}  mean={e['spv_mean']:.3f}  max={e['spv_max']:.3f}",
            f"  Power (effect/sigma anticipated):",
        ]
        for name, pw in self.power.items():
            if pw is None or not np.isfinite(pw):
                lines.append(f"      {name:<28} {'n/a':>5}")
            else:
                lines.append(f"      {name:<28} {float(pw):5.2f}")
        if len(self.vif):
            vmax = float(np.nanmax(self.vif.to_numpy()))
            vmax_s = "inf" if np.isinf(vmax) else f"{vmax:.2f}"
            lines.append(f"  Max VIF: {vmax_s}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (``schema: doekit.DesignEvaluation/1``)."""
        def _num(v):
            if v is None:
                return None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return v
            if np.isnan(fv) or np.isinf(fv):
                return None
            return fv

        eff = {k: _num(v) if isinstance(v, (int, float, np.floating, np.integer))
               else bool(v) if isinstance(v, (bool, np.bool_)) else v
               for k, v in self.efficiencies.items()}
        return {
            "schema": "doekit.DesignEvaluation/1",
            "n_runs": int(self.n_runs),
            "n_params": int(self.n_params),
            "dof": int(self.dof),
            "efficiencies": eff,
            "power": {str(k): _num(v) for k, v in self.power.items()},
            "vif": {str(k): _num(v) for k, v in self.vif.items()},
            "fds": self.fds.to_dict("list"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DesignEvaluation":
        """Rebuild a :class:`DesignEvaluation` from :meth:`to_dict` output."""
        if d.get("schema") not in (None, "doekit.DesignEvaluation/1"):
            raise ValueError(f"unsupported DesignEvaluation schema: {d.get('schema')!r}")

        def _maybe_float(v):
            if v is None:
                return float("nan")
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        power = d.get("power") or {}
        vif_d = d.get("vif") or {}
        fds_d = d.get("fds") or {}
        return cls(
            n_runs=int(d["n_runs"]),
            n_params=int(d["n_params"]),
            dof=int(d["dof"]),
            efficiencies=dict(d.get("efficiencies") or {}),
            power=pd.Series({k: _maybe_float(v) for k, v in power.items()}),
            vif=pd.Series({k: _maybe_float(v) for k, v in vif_d.items()}),
            fds=pd.DataFrame(fds_d) if fds_d else pd.DataFrame(),
        )

    def __repr__(self) -> str:
        return self.summary()


def evaluate(design: Design, model: Optional[Model] = None,
             effect_size=1.0, sigma: float = 1.0, alpha: float = 0.05,
             region: Optional[pd.DataFrame] = None, n_region: int = 20000,
             seed: Optional[int] = None, report=None) -> DesignEvaluation:
    """Evaluate a design end-to-end and return a :class:`DesignEvaluation`.

    Combines :func:`efficiencies`, :func:`power_analysis`, :func:`vif` and
    :func:`fds_data`. Metrics use coded factor units when ``Design.factors``
    is set. Saturated designs yield an empty ``power`` series instead of
    raising.

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Model to score; resolved from the design if omitted.
    effect_size : float or array-like, default 1.0
        Anticipated effect size for the power analysis (same units as the
        response; non-centrality uses ``effect_size / sigma``).
    sigma : float, default 1.0
        Noise standard deviation.
    alpha : float, default 0.05
        Two-sided significance level for the power analysis.
    region : DataFrame, optional
        Precomputed region sample shared by efficiency and FDS.
    n_region : int, default 20000
        Region sample size when ``region`` is not provided.
    seed : int, optional
        Seed for the region sampling.
    report : None, bool, str, Path or dict, optional
        If not ``None``, a design-quality HTML report is generated and its path
        is stored in ``result.report_path``.

    Returns
    -------
    DesignEvaluation
        The full quality report card.

    Notes
    -----
    See the Formulas sections of :func:`efficiencies`, :func:`power_analysis`,
    :func:`vif` and :func:`fds_data` for the underlying equations.

    Examples
    --------
    >>> import doekit as ed
    >>> ev = ed.evaluate(ed.plackett_burman(6), n_region=500, seed=0)
    >>> print(ev.d_efficiency > 80)
    True
    """
    model = _resolve_model(design, model)
    eff = efficiencies(design, model, region=region, n_region=n_region, seed=seed)
    fds = fds_data(design, model, region=region, n_region=n_region, seed=seed)
    v = vif(design, model)
    try:
        pw = power_analysis(design, model, effect_size=effect_size,
                            sigma=sigma, alpha=alpha)
    except ValueError:
        pw = pd.Series(dtype=float, name="power")  # saturated design
    result = DesignEvaluation(
        n_runs=eff["n_runs"], n_params=eff["n_params"], dof=eff["dof"],
        efficiencies=eff, power=pw, vif=v, fds=fds,
    )
    if report is not None:
        from ..._compat import maybe_report  # noqa: PLC0415
        result.report_path = maybe_report(
            design, report=report, model=model,
            effect_size=effect_size, sigma=sigma, alpha=alpha, seed=seed,
        )
    return result
