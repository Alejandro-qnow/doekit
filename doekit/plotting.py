"""Visualization helpers (matplotlib optional, lazy import).

Install with ``pip install doekit[plot]``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .analysis import half_normal_data


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

def fds_plot(design, model=None, ax=None, label=None, **fds_kwargs):
    """Fraction of Design Space plot: SPV vs. region fraction (0->1).

    A low, flat curve indicates uniform and precise prediction across the whole
    region.

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
    **fds_kwargs
        Extra keyword arguments forwarded to :func:`doekit.fds_data`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    plt = _require_mpl()
    import pandas as pd  # noqa: PLC0415
    from .evaluate import fds_data  # noqa: PLC0415
    data = design if isinstance(design, pd.DataFrame) else fds_data(design, model, **fds_kwargs)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["fraction"], data["spv"], linewidth=2, label=label)
    ax.set_xlabel("Fraction of design region")
    ax.set_ylabel("Scaled prediction variance (SPV)")
    ax.set_title("Fraction of Design Space (FDS)")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
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
