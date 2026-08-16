"""Visualization helpers (matplotlib optional, lazy import).

Install with ``pip install doekit[plot]``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...assessment.analysis import half_normal_data


def _require_mpl():
    """Import and return ``matplotlib.pyplot`` or raise a helpful ImportError."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install with "
            "'pip install doekit[plot]'."
        ) from exc
    return plt


def half_normal_plot(effects, labels: Optional[Sequence[str]] = None, ax=None,
                     annotate: bool = True):
    """Half-normal plot of effects: points off the line are significant.

    Daniel plot for screening: inactive effects follow the half-normal reference;
    departures indicate active factors.

    Parameters
    ----------
    effects : array-like
        Estimated effects (or coefficients).
    labels : sequence of str, optional
        One label per effect; defaults to ``e1..em``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    annotate : bool, default True
        Whether to annotate each point with its label.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Examples
    --------
    >>> import doekit as ed
    >>> ax = ed.presentation.render.figures_mpl.half_normal_plot([0.1, 3.0, -0.2])
    >>> ax.get_ylabel()
    '|Effect|'
    """
    plt = _require_mpl()
    data = half_normal_data(effects, labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data["half_normal_quantile"], data["abs_effect"], color="tab:blue")
    if annotate:
        for _, row in data.iterrows():
            ax.annotate(str(row["label"]),
                        (row["half_normal_quantile"], row["abs_effect"]),
                        textcoords="offset points", xytext=(5, 0), fontsize=8)
    ax.set_xlabel("Half-normal quantile")
    ax.set_ylabel("|Effect|")
    ax.set_title("Half-normal plot of effects")
    ax.grid(True, alpha=0.3)
    return ax


def effects_plot(effects, labels: Optional[Sequence[str]] = None, ax=None):
    """Bar chart of signed effects, sorted by ``|effect|``.

    Parameters
    ----------
    effects : array-like
        Estimated effects.
    labels : sequence of str, optional
        One label per effect; defaults to ``e1..em``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    plt = _require_mpl()
    eff = np.asarray(effects, dtype=float)
    if labels is None:
        labels = [f"e{i + 1}" for i in range(len(eff))]
    order = np.argsort(np.abs(eff))[::-1]
    eff, labels = eff[order], np.asarray(labels)[order]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:red" if v < 0 else "tab:green" for v in eff]
    ax.barh(range(len(eff)), eff, color=colors)
    ax.set_yticks(range(len(eff)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect")
    ax.set_title("Estimated effects")
    return ax


def correlation_plot(design, model=None, ax=None):
    """Correlation map of the model-matrix columns (aliasing).

    Parameters
    ----------
    design : Design
        The design whose model matrix is correlated.
    model : Model, optional
        Model to build the matrix; taken from ``design.model`` if omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    plt = _require_mpl()
    mdl = model or design.model
    X = mdl.matrix(design.matrix)
    names = mdl.column_names(design.matrix)
    corr = np.corrcoef(X, rowvar=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of model columns")
    return ax


# --- evaluation / benchmarking plots ---------------------------------------

def fds_plot(design, model=None, ax=None, label=None, surrogate=None,
             n_region: int = 2000, seed=None, **fds_kwargs):
    """Fraction of Design Space plot: SPV vs. region fraction (0->1).

    A low, flat curve indicates uniform and precise prediction across the whole
    region. When a ``surrogate`` is passed, the curve instead shows the sorted
    predictive **standard deviation** ``sigma(x)`` over a region cover — the
    optimize-intent analogue of SPV (how uniform the surrogate's uncertainty is).

    Formulas
    --------
    Design mode: ``SPV(x) = N * x'(X'X)^-1 x`` sorted over the region sample.
    Surrogate mode: ``sigma(x)`` from ``surrogate.predict`` sorted likewise.

    Parameters
    ----------
    design : Design or DataFrame
        A :class:`Design`, or a ``DataFrame`` already produced by
        :func:`doekit.fds_data`.
    model : Model, optional
        Model to build the matrix (only used when ``design`` is a ``Design``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; accepts repeated calls to overlay several designs.
    label : str, optional
        Legend label for the curve.
    surrogate : Surrogate, optional
        If given, plot sorted ``sigma(x)`` from the surrogate over a sampled
        region cover instead of the design SPV.
    n_region : int, default 2000
        Number of region points sampled for the surrogate curve.
    seed : int, optional
        RNG seed for the surrogate region cover.
    **fds_kwargs
        Extra keyword arguments forwarded to :func:`doekit.fds_data`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Examples
    --------
    >>> import doekit as ed
    >>> ax = ed.presentation.render.figures_mpl.fds_plot(
    ...     ed.full_factorial(3), seed=0)
    >>> ax.get_xlabel()
    'Fraction of design region'
    """
    plt = _require_mpl()
    import pandas as pd  # noqa: PLC0415
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if surrogate is not None:
        frame = _region_cover(surrogate, n_region, seed)
        _, std = surrogate.predict(frame)
        std = np.sort(np.asarray(std, dtype=float))
        frac = np.linspace(0, 1, len(std))
        ax.plot(frac, std, linewidth=2, label=label)
        ax.set_ylabel(r"Surrogate prediction std $\sigma(x)$")
        ax.set_title("Fraction of Design Space (surrogate uncertainty)")
    else:
        from ...assessment.evaluation import fds_data  # noqa: PLC0415
        data = (design if isinstance(design, pd.DataFrame)
                else fds_data(design, model, **fds_kwargs))
        ax.plot(data["fraction"], data["spv"], linewidth=2, label=label)
        ax.set_ylabel("Scaled prediction variance (SPV)")
        ax.set_title("Fraction of Design Space (FDS)")
    ax.set_xlabel("Fraction of design region")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


# --- surrogate / optimization plots ----------------------------------------

def _factor_bounds(surrogate, bounds=None) -> dict:
    """Per-factor ``(low, high)`` bounds from factors / training data."""
    import pandas as pd  # noqa: PLC0415
    out = dict(bounds or {})
    frame = getattr(surrogate, "_frame", None)
    for f in getattr(surrogate, "factors", []) or []:
        name = getattr(f, "name", None)
        if name in out:
            continue
        low, high = getattr(f, "low", None), getattr(f, "high", None)
        if low is not None and high is not None:
            out[name] = (float(low), float(high))
        elif frame is not None and name in frame and pd.api.types.is_numeric_dtype(frame[name]):
            out[name] = (float(frame[name].min()), float(frame[name].max()))
        else:
            out[name] = (-1.0, 1.0)
    for name in getattr(surrogate, "factor_names", []):
        if name not in out:
            if frame is not None and name in frame and pd.api.types.is_numeric_dtype(frame[name]):
                out[name] = (float(frame[name].min()), float(frame[name].max()))
            else:
                out[name] = (-1.0, 1.0)
    return out


def _region_cover(surrogate, n, seed=None):
    """Sample a factor-frame cover of the surrogate's region."""
    import pandas as pd  # noqa: PLC0415
    rng = np.random.default_rng(seed)
    bounds = _factor_bounds(surrogate)
    data = {name: rng.uniform(lo, hi, size=n) for name, (lo, hi) in bounds.items()}
    return pd.DataFrame(data)[list(surrogate.factor_names)]


def _surrogate_grid(surrogate, x, y, bounds, resolution, at):
    import pandas as pd  # noqa: PLC0415
    names = list(surrogate.factor_names)
    x = x or names[0]
    y = y or (names[1] if len(names) > 1 else names[0])
    b = _factor_bounds(surrogate, bounds)
    at = dict(at or {})
    xs = np.linspace(b[x][0], b[x][1], resolution)
    ys = np.linspace(b[y][0], b[y][1], resolution)
    XX, YY = np.meshgrid(xs, ys)
    cols = {}
    for n in names:
        cols[n] = np.full(XX.size, at.get(n, 0.5 * (b[n][0] + b[n][1])), dtype=float)
    cols[x] = XX.ravel()
    cols[y] = YY.ravel()
    frame = pd.DataFrame(cols)[names]
    return x, y, xs, ys, XX, YY, frame


def surrogate_surface(surrogate, x=None, y=None, bounds=None, at=None,
                      measured=None, proposed=None, optimum=None,
                      resolution: int = 60, ax=None, cmap="viridis",
                      colorbar: bool = True):
    """Contour of the surrogate mean ``mu(x)`` over two factors.

    Marks measured points, the current optimum, and any proposed runs — the
    "optimum vs. the non-optima" picture.

    Parameters
    ----------
    surrogate : Surrogate
        A fitted surrogate (``predict`` + ``factor_names``).
    x, y : str, optional
        Factor names for the two axes (first two factors by default).
    bounds : dict, optional
        ``name -> (low, high)`` overrides.
    at : dict, optional
        Fixed values for the other factors (defaults to their midpoint).
    measured : DataFrame, optional
        Measured design points to overlay (factor columns).
    proposed : DataFrame, optional
        Proposed next runs to overlay (e.g. ``proposal.added.matrix``).
    optimum : mapping or array, optional
        Point to star as the current best.
    resolution : int, default 60
        Grid resolution per axis.
    ax : matplotlib.axes.Axes, optional
    cmap : str, default "viridis"
    colorbar : bool, default True

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    x, y, xs, ys, XX, YY, frame = _surrogate_grid(
        surrogate, x, y, bounds, resolution, at)
    mean, _ = surrogate.predict(frame)
    Z = np.asarray(mean, dtype=float).reshape(XX.shape)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(XX, YY, Z, levels=20, cmap=cmap)
    ax.contour(XX, YY, Z, levels=8, colors="k", linewidths=0.3, alpha=0.4)
    if colorbar:
        ax.figure.colorbar(cs, ax=ax, shrink=0.85, label=r"surrogate $\mu(x)$")
    if measured is not None and x in measured and y in measured:
        ax.scatter(measured[x], measured[y], c="white", edgecolors="black",
                   s=35, label="measured", zorder=3)
    if proposed is not None and x in proposed and y in proposed:
        ax.scatter(proposed[x], proposed[y], marker="D", c="tab:red",
                   edgecolors="black", s=55, label="proposed", zorder=4)
    if optimum is not None:
        try:
            ox, oy = float(optimum[x]), float(optimum[y])
            ax.scatter([ox], [oy], marker="*", c="gold", edgecolors="black",
                       s=260, label="best", zorder=5)
        except (KeyError, TypeError, IndexError):
            pass
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Surrogate response surface")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    return ax


def acquisition_plot(surrogate, best=None, goal="max", kind="ei", x=None, y=None,
                     bounds=None, at=None, proposed=None, resolution: int = 60,
                     ax=None, cmap="magma", kappa: float = 2.0, xi: float = 0.01,
                     colorbar: bool = True):
    """Contour of an acquisition surface: where the loop wants to sample next.

    Parameters
    ----------
    surrogate : Surrogate
    best : float, optional
        Best objective so far (defaults to the best of the training data).
    goal : {"max", "min"}, default "max"
    kind : {"ei", "ucb", "pi"}, default "ei"
    x, y, bounds, at, resolution, ax
        As in :func:`surrogate_surface`.
    proposed : DataFrame, optional
        Proposed runs to overlay.
    kappa, xi
        Acquisition exploration parameters.
    colorbar : bool, default True

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    from ...orchestration.optimize import (  # noqa: PLC0415
        expected_improvement, upper_confidence_bound, probability_of_improvement,
    )
    x, y, xs, ys, XX, YY, frame = _surrogate_grid(
        surrogate, x, y, bounds, resolution, at)
    mean, std = surrogate.predict(frame)
    if best is None:
        yt = np.asarray(getattr(surrogate, "_y", mean), dtype=float)
        best = float(np.max(yt) if goal == "max" else np.min(yt))
    k = str(kind).lower()
    if k == "ucb":
        vals = upper_confidence_bound(mean, std, kappa=kappa, goal=goal)
    elif k == "pi":
        vals = probability_of_improvement(mean, std, best, goal=goal, xi=xi)
    else:
        vals = expected_improvement(mean, std, best, goal=goal, xi=xi)
    Z = np.asarray(vals, dtype=float).reshape(XX.shape)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(XX, YY, Z, levels=20, cmap=cmap)
    if colorbar:
        ax.figure.colorbar(cs, ax=ax, shrink=0.85, label=f"{k.upper()} acquisition")
    if proposed is not None and x in proposed and y in proposed:
        ax.scatter(proposed[x], proposed[y], marker="D", c="cyan",
                   edgecolors="black", s=55, label="proposed", zorder=4)
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Acquisition surface ({k.upper()})")
    return ax


def convergence_plot(history, ax=None, label=None, optimum=None,
                     marker="o"):
    """Best-so-far ``y*`` per generation (are we approaching the optimum?).

    Parameters
    ----------
    history : array-like
        Best-so-far value per generation (or a 2-column ``(gen, value)``).
    ax : matplotlib.axes.Axes, optional
    label : str, optional
        Legend label (overlay multiple strategies).
    optimum : float, optional
        Known optimum, drawn as a dashed reference line.
    marker : str, default "o"

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    arr = np.asarray(history, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        gens, vals = arr[:, 0], arr[:, 1]
    else:
        vals = arr.reshape(-1)
        gens = np.arange(len(vals))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gens, vals, marker=marker, linewidth=2, label=label)
    if optimum is not None:
        ax.axhline(float(optimum), color="black", linestyle="--", linewidth=1,
                   label="known optimum")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best so far ($y^*$)")
    ax.set_title("Optimization convergence")
    ax.grid(True, alpha=0.3)
    if label or optimum is not None:
        ax.legend(fontsize=8)
    return ax


def parity_plot(surrogate, design=None, y=None, ax=None, loo: bool = False):
    """Predicted vs. observed with ``sigma`` error bars — audit the surrogate.

    Parameters
    ----------
    surrogate : Surrogate
    design, y : optional
        Data to evaluate; defaults to the surrogate's training data.
    ax : matplotlib.axes.Axes, optional
    loo : bool, default False
        Reserved (in-sample parity by default).

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    _ = loo
    frame = design if design is not None else getattr(surrogate, "_frame", None)
    obs = (np.asarray(y, dtype=float) if y is not None
           else np.asarray(getattr(surrogate, "_y", None), dtype=float))
    if frame is None or obs is None:
        raise ValueError("parity_plot needs a fitted surrogate or design+y")
    mean, std = surrogate.predict(frame)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(obs, mean, yerr=std, fmt="o", color="tab:blue",
                ecolor="gray", elinewidth=1, capsize=2, alpha=0.8)
    lo = float(min(np.min(obs), np.min(mean)))
    hi = float(max(np.max(obs), np.max(mean)))
    pad = 0.05 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted ($\\pm\\sigma$)")
    ax.set_title("Parity (predicted vs. observed)")
    ax.grid(True, alpha=0.3)
    return ax


def calibration_plot(surrogate, levels=(0.5, 0.8, 0.95), ax=None, label=None):
    """LOO interval coverage vs. nominal — the moat plot (is ``sigma`` honest?).

    Points on the diagonal mean the surrogate's uncertainty is well-calibrated;
    below the diagonal means it is over-confident.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    cal = surrogate.calibration(levels=levels)
    lv = list(cal["levels"])
    cov = [cal["coverage"][l] for l in lv]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    ax.plot(lv, cov, marker="o", linewidth=2, label=label or "surrogate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal interval")
    ax.set_ylabel("LOO coverage")
    ax.set_title("Calibration of surrogate intervals")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def pareto_plot(Y, goals=None, columns=None, x=None, y=None, ax=None,
                annotate: bool = False):
    """Scatter of two objectives, highlighting the non-dominated (Pareto) front.

    Parameters
    ----------
    Y : DataFrame or ndarray
        Objective values (n x m).
    goals : dict, optional
        ``{column: "max"|"min"}``.
    columns : sequence of str, optional
        Objective names (taken from a DataFrame when available).
    x, y : str or int, optional
        Which two objectives to plot.
    ax : matplotlib.axes.Axes, optional
    annotate : bool, default False

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    import pandas as pd  # noqa: PLC0415
    from ...orchestration.optimize import pareto_mask, to_max_space  # noqa: PLC0415
    if isinstance(Y, pd.DataFrame):
        columns = columns or list(Y.columns)
        arr = Y.loc[:, columns].to_numpy(dtype=float)
    else:
        arr = np.atleast_2d(np.asarray(Y, dtype=float))
        columns = list(columns) if columns is not None else [
            f"obj{i + 1}" for i in range(arr.shape[1])]
    xi = columns.index(x) if isinstance(x, str) else (x or 0)
    yi = columns.index(y) if isinstance(y, str) else (y or 1)
    Z = to_max_space(arr, goals, columns)
    mask = pareto_mask(Z)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr[~mask, xi], arr[~mask, yi], c="lightgray",
               edgecolors="gray", label="dominated", zorder=2)
    order = np.argsort(arr[mask, xi])
    fx, fy = arr[mask, xi][order], arr[mask, yi][order]
    ax.plot(fx, fy, "-o", color="tab:red", label="Pareto front", zorder=3)
    if annotate:
        for i in np.where(mask)[0]:
            ax.annotate(str(i), (arr[i, xi], arr[i, yi]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
    ax.set_xlabel(columns[xi])
    ax.set_ylabel(columns[yi])
    ax.set_title("Pareto front (optimum vs. non-optima)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def slice_plot(surrogate, factor=None, at=None, bounds=None, resolution: int = 120,
               ax=None, label=None):
    """1D slice / partial dependence of ``mu(x) +/- sigma`` along one factor.

    Parameters
    ----------
    surrogate : Surrogate
    factor : str, optional
        Factor to vary (first factor by default).
    at : dict, optional
        Fixed values for the other factors (midpoints by default) — e.g. the
        optimum, to slice *through* it.
    bounds : dict, optional
    resolution : int, default 120
    ax : matplotlib.axes.Axes, optional
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_mpl()
    import pandas as pd  # noqa: PLC0415
    names = list(surrogate.factor_names)
    factor = factor or names[0]
    b = _factor_bounds(surrogate, bounds)
    at = dict(at or {})
    xs = np.linspace(b[factor][0], b[factor][1], resolution)
    cols = {n: np.full(resolution, at.get(n, 0.5 * (b[n][0] + b[n][1])), dtype=float)
            for n in names}
    cols[factor] = xs
    frame = pd.DataFrame(cols)[names]
    mean, std = surrogate.predict(frame)
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, mean, linewidth=2, label=label or f"mu | {factor}")
    ax.fill_between(xs, mean - std, mean + std, alpha=0.2, label=r"$\pm\sigma$")
    ax.set_xlabel(factor)
    ax.set_ylabel("Surrogate prediction")
    ax.set_title(f"Slice through {factor}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def power_plot(power, ax=None, alpha_ref: float = 0.8):
    """Bar chart of per-term power (output of ``power_analysis``).

    Parameters
    ----------
    power : pandas.Series or array-like
        Power per model term.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    alpha_ref : float, default 0.8
        Reference power level marked with a dashed line.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Examples
    --------
    >>> import doekit as ed
    >>> ev = ed.evaluate(ed.full_factorial(3), seed=0)
    >>> ax = ed.presentation.render.figures_mpl.power_plot(ev.power)
    >>> ax.get_xlabel()
    'Power'
    """
    plt = _require_mpl()
    import pandas as pd  # noqa: PLC0415
    s = power if isinstance(power, pd.Series) else pd.Series(power)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:green" if v >= alpha_ref else "tab:orange" for v in s.to_numpy()]
    ax.barh(range(len(s)), s.to_numpy(), color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(list(s.index), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(alpha_ref, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Power")
    ax.set_title("Power analysis by term")
    return ax


def alias_heatmap(alias_df, ax=None):
    """Heatmap of the alias matrix (output of ``alias_matrix``).

    Parameters
    ----------
    alias_df : pandas.DataFrame
        The alias matrix, indexed by primary terms and columned by alias terms.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    plt = _require_mpl()
    A = alias_df.to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    vmax = max(1.0, float(np.abs(A).max()))
    im = ax.imshow(A, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(A.shape[1]))
    ax.set_yticks(range(A.shape[0]))
    ax.set_xticklabels(list(alias_df.columns), rotation=90, fontsize=7)
    ax.set_yticklabels(list(alias_df.index), fontsize=7)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Alias matrix (bias from omitted terms)")
    return ax
