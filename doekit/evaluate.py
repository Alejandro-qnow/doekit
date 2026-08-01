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

from .model import Model
from .designs.base import Design


# --- coding utilities --------------------------------------------------------

def _coded_frame(design: Design) -> pd.DataFrame:
    """Return the design matrix with continuous/discrete factors in ``+/-1``.

    Categorical factors are left raw (the model dummy-codes them). Columns with
    no associated factor are assumed to be coded already.
    """
    df = design.matrix.copy()
    for f in (design.factors or []):
        if f.name in df.columns and not getattr(f, "is_categorical", False):
            df[f.name] = np.asarray(f.encode(df[f.name].to_numpy()), dtype=float)
    return df


def _resolve_model(design: Design, model: Optional[Model]) -> Model:
    """Return ``model`` or fall back to ``design.model`` / a main-effects model."""
    model = model or design.model
    if model is None:
        model = Model.main_effects(list(design.matrix.columns), intercept=True)
    return model


def _region_frame(design: Design, model: Model, n_region: int,
                  seed: Optional[int]) -> pd.DataFrame:
    """Sample the experimental region in coded units.

    Continuous factors -> uniform on ``[-1, 1]``; categorical factors -> sampling
    of their levels. This is the region over which the prediction variance (G, I)
    and the FDS plot are evaluated.
    """
    rng = np.random.default_rng(seed)
    cat = {f.name: list(f.levels) for f in (design.factors or [])
           if getattr(f, "is_categorical", False)}
    cols = {}
    for name in model.factor_names:
        if name in cat:
            cols[name] = rng.choice(np.array(cat[name], dtype=object), size=n_region)
        else:
            cols[name] = rng.uniform(-1.0, 1.0, size=n_region)
    return pd.DataFrame(cols)


def _leverage(Xr: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    """Return ``x' M^-1 x`` per row (unscaled prediction variance)."""
    return np.einsum("ij,jk,ik->i", Xr, Minv, Xr)


# --- individual metrics ------------------------------------------------------

def efficiencies(design: Design, model: Optional[Model] = None,
                 region: Optional[pd.DataFrame] = None, n_region: int = 20000,
                 seed: Optional[int] = None) -> dict:
    """D/A/G efficiencies (%) and scaled-prediction-variance statistics.

    - **D-eff** ``= 100 * det(X'X)^(1/p) / N`` (100% for an orthogonal design).
    - **A-eff** ``= 100 * p / (N * tr((X'X)^-1))``.
    - **G-eff** ``= 100 * p / max(SPV)`` over the region, with
      ``SPV(x) = N * x'(X'X)^-1 x`` (general equivalence theorem: ``max SPV >= p``).
    - **I** (mean SPV over the region) plus min/max of SPV.

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

    For coefficient ``beta_i`` with standard error ``se_i = sigma *
    sqrt((X'X)^-1_ii)``, the power of the two-sided t-test at level ``alpha`` uses
    the non-central ``t`` with non-centrality ``delta_i = effect_size / se_i`` and
    ``N - p`` degrees of freedom.

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
        Power per model term.

    Raises
    ------
    ValueError
        If ``N <= p`` (a saturated design has no residual degrees of freedom).
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
    power = stats.nct.sf(tcrit, dof, ncp) + stats.nct.cdf(-tcrit, dof, ncp)
    return pd.Series(power, index=names, name="power")


def vif(design: Design, model: Optional[Model] = None) -> pd.Series:
    """Variance inflation factors of the model terms (intercept excluded).

    ``VIF_j = 1 / (1 - R2_j)`` where ``R2_j`` regresses column ``j`` on the
    others. Computed as the diagonal of the inverse correlation matrix.
    ``VIF = 1`` indicates orthogonality; ``> 5-10`` signals serious
    multicollinearity.

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
    """
    from .model import Intercept
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
    C = (Xc.T @ Xc) / Xc.shape[0]
    try:
        Cinv = np.linalg.pinv(C)
        vifs = np.diag(Cinv).astype(float)
    except np.linalg.LinAlgError:
        vifs = np.full(Xk.shape[1], np.nan)
    vifs[const] = np.nan
    return pd.Series(vifs, index=kept_names, name="vif")


def alias_matrix(design: Design, model: Optional[Model] = None,
                 alias_model: Optional[Model] = None) -> pd.DataFrame:
    """Alias matrix ``A = (X1'X1)^-1 X1'X2``.

    Quantifies the bias of the primary-model coefficients (``X1``) induced by
    potential omitted terms (``X2``): ``E[beta_hat_1] = beta_1 + A beta_2``. If
    ``alias_model`` is ``None``, the two-factor interactions not already in the
    primary model are used by default.

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
        The alias matrix, indexed by primary terms and columned by alias terms.
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
    from .model import Interaction
    names = model.factor_names
    existing = {t.label() for t in model.terms}
    terms = [Interaction((a, b)) for a, b in combinations(names, 2)
             if f"{a}:{b}" not in existing and f"{b}:{a}" not in existing]
    return Model.from_terms(terms, intercept=False)


def fds_data(design: Design, model: Optional[Model] = None,
             region: Optional[pd.DataFrame] = None, n_region: int = 20000,
             seed: Optional[int] = None) -> pd.DataFrame:
    """Data for the **Fraction of Design Space (FDS) plot**.

    Sorts the scaled prediction variance ``SPV = N * x'(X'X)^-1 x`` over a region
    sample and returns a ``DataFrame`` with ``fraction`` (0->1) and ``spv``. A
    flat, low curve = uniform and precise prediction across the whole region (the
    desirable case). It is the gold standard of design evaluation.

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
        Columns ``fraction`` and ``spv``.
    """
    model = _resolve_model(design, model)
    coded = _coded_frame(design)
    X = model.matrix(coded)
    n = X.shape[0]
    Minv = np.linalg.pinv(X.T @ X)
    reg = region if region is not None else _region_frame(design, model, n_region, seed)
    Xr = model.matrix(reg)
    spv = np.sort(n * _leverage(Xr, Minv))
    m = len(spv)
    fraction = (np.arange(1, m + 1) - 0.5) / m
    return pd.DataFrame({"fraction": fraction, "spv": spv})


# --- aggregate report --------------------------------------------------------

@dataclass
class DesignEvaluation:
    """Quality report card of a design (result of :func:`evaluate`).

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
            lines.append(f"      {name:<28} {pw:5.2f}")
        if len(self.vif):
            lines.append(f"  Max VIF: {float(np.nanmax(self.vif.to_numpy())):.2f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def evaluate(design: Design, model: Optional[Model] = None,
             effect_size=1.0, sigma: float = 1.0, alpha: float = 0.05,
             region: Optional[pd.DataFrame] = None, n_region: int = 20000,
             seed: Optional[int] = None, report=None) -> DesignEvaluation:
    """Evaluate a design end-to-end and return a :class:`DesignEvaluation`.

    Combines :func:`efficiencies`, :func:`power_analysis`, :func:`vif` and
    :func:`fds_data`. ``effect_size`` is the anticipated effect size (in units of
    ``sigma``) for the power. Everything is computed in coded units.

    Parameters
    ----------
    design : Design
        The design to evaluate.
    model : Model, optional
        Model to score; resolved from the design if omitted.
    effect_size : float or array-like, default 1.0
        Anticipated effect size for the power analysis.
    sigma : float, default 1.0
        Noise standard deviation.
    alpha : float, default 0.05
        Two-sided significance level for the power analysis.
    region, n_region, seed
        Region-sampling controls shared by the efficiency and FDS computations.
    report : None, bool, str, Path or dict, optional
        If not ``None``, a design-quality HTML report is generated and its path is
        stored in ``result.report_path``.

    Returns
    -------
    DesignEvaluation
        The full quality report card.
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
        from .report import run_report_arg  # noqa: PLC0415
        result.report_path = run_report_arg(design, model=model, report=report,
                                            effect_size=effect_size, sigma=sigma,
                                            alpha=alpha, seed=seed)
    return result
