"""Thin bridge from Bayesian-optimization search spaces to doekit candidates.

This module does **not** replace the transparent advisor. It only turns a
bound specification (or an optional scikit-optimize / Optuna space) into a
:class:`~doekit.domain.design.Design` candidate set that
:func:`~doekit.orchestration.sequential.augment_design` /
:func:`~doekit.orchestration.sequential.propose_next_runs` can consume — so BO
and classical DoE share efficiencies / FDS / power as a common language.

Optional extras: ``pip install "doekit[bo]"`` pulls ``scikit-optimize`` when you
want ``candidates_from_skopt_space``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ..domain.factors import ContinuousFactor, DiscreteFactor, CategoricalFactor, as_factors
from ..domain.design import Design
from ..generation.catalog.random_design import random_design
from ..domain.model import Model


BoundSpec = Union[
    ContinuousFactor,
    DiscreteFactor,
    CategoricalFactor,
    tuple,  # (name, low, high) or (name, levels_list)
]


def _factor_from_spec(spec) -> object:
    if hasattr(spec, "name") and hasattr(spec, "to_dict"):
        return spec
    if not isinstance(spec, (tuple, list)) or len(spec) < 2:
        raise ValueError(
            "each bound must be a Factor or (name, low, high) / (name, levels)"
        )
    name = spec[0]
    if len(spec) == 3 and all(isinstance(v, (int, float)) for v in spec[1:]):
        return ContinuousFactor(str(name), float(spec[1]), float(spec[2]))
    levels = list(spec[1]) if len(spec) == 2 else list(spec[1:])
    if all(isinstance(v, (int, float)) for v in levels):
        return DiscreteFactor(str(name), levels)
    return CategoricalFactor(str(name), levels)


def candidates_from_bounds(bounds: Sequence[BoundSpec], n: int = 200,
                           model: Optional[Model] = None,
                           seed: Optional[int] = None) -> Design:
    """Sample a candidate :class:`Design` from factor bounds.

    Builds a random candidate set over the declared factors for
    :func:`~doekit.orchestration.sequential.augment_design` or optimal subsampling.
    Attaches a main-effects model when none is provided.

    Parameters
    ----------
    bounds : sequence
        Factors, or tuples ``(name, low, high)`` / ``(name, levels)``.
    n : int, default 200
        Number of candidate runs.
    model : Model, optional
        Attached to the returned design when provided.
    seed : int, optional
        RNG seed.

    Returns
    -------
    Design
        Candidate set suitable for ``optimal_design`` / ``augment_design``.

    Examples
    --------
    >>> import doekit as ed
    >>> cand = ed.candidates_from_bounds([("x1", -1, 1), ("x2", -1, 1)], n=50, seed=0)
    >>> cand.n_runs == 50 and cand.metadata["kind"] == "BOCandidates"
    True
    """
    facs = [_factor_from_spec(b) for b in bounds]
    facs = as_factors(facs)
    cand = random_design(facs, n=n, seed=seed)
    if model is not None:
        cand.model = model
    elif facs:
        cand.model = Model.main_effects([f.name for f in facs])
    cand.metadata["kind"] = "BOCandidates"
    cand.metadata["source"] = "bounds"
    return cand


def candidates_from_skopt_space(space, n: int = 200, names: Optional[Sequence[str]] = None,
                                model: Optional[Model] = None,
                                seed: Optional[int] = None) -> Design:
    """Build candidates from a ``skopt.space.Space`` (requires ``doekit[bo]``).

    Maps scikit-optimize dimensions to doekit factors and samples ``n`` points
    with a fixed RNG seed for reproducibility.

    Parameters
    ----------
    space : skopt.space.Space
        Search space.
    n : int, default 200
        Number of candidate points (via dimension bounds).
    names : sequence of str, optional
        Dimension names; defaults to ``dim.name`` or ``x0..``.
    model : Model, optional
        Attached model.
    seed : int, optional
        RNG seed.

    Returns
    -------
    Design
        Candidate design with ``metadata["source"] == "skopt"``.

    Raises
    ------
    ImportError
        If scikit-optimize is not installed.
    """
    try:
        from skopt.space import Real, Integer, Categorical  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-optimize is required for candidates_from_skopt_space. "
            "Install with: pip install 'doekit[bo]'"
        ) from exc

    import pandas as pd

    dims = list(space.dimensions) if hasattr(space, "dimensions") else list(space)
    col_names = []
    for i, dim in enumerate(dims):
        if names is not None and i < len(names):
            col_names.append(str(names[i]))
        else:
            col_names.append(getattr(dim, "name", None) or f"x{i}")

    rng = np.random.default_rng(seed)
    # skopt Space.rvs uses its own RNG; sample via dimension bounds for stability
    cols = {}
    facs = []
    for name, dim in zip(col_names, dims):
        if isinstance(dim, Real):
            lo, hi = float(dim.low), float(dim.high)
            cols[name] = rng.uniform(lo, hi, size=n)
            facs.append(ContinuousFactor(name, lo, hi))
        elif isinstance(dim, Integer):
            lo, hi = int(dim.low), int(dim.high)
            cols[name] = rng.integers(lo, hi + 1, size=n)
            facs.append(DiscreteFactor(name, list(range(lo, hi + 1))))
        elif isinstance(dim, Categorical):
            cats = list(dim.categories)
            cols[name] = rng.choice(np.array(cats, dtype=object), size=n)
            facs.append(CategoricalFactor(name, cats))
        else:
            # Fallback: try low/high
            lo, hi = float(getattr(dim, "low", -1)), float(getattr(dim, "high", 1))
            cols[name] = rng.uniform(lo, hi, size=n)
            facs.append(ContinuousFactor(name, lo, hi))

    d = Design(matrix=pd.DataFrame(cols), factors=facs,
               model=model or Model.main_effects(col_names),
               metadata={"kind": "BOCandidates", "source": "skopt"})
    return d
