"""Random designs: sampling from distributions and Latin Hypercube.

Uses ``scipy.stats`` for independent draws and ``scipy.stats.qmc`` for Latin
Hypercube Sampling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ...domain.factors import (CategoricalFactor, ContinuousFactor, DiscreteFactor,
                       Factor)
from ...domain.model import Model
from ...domain.design import Design


def random_design(factors, n: int, seed: Optional[int] = None,
                  model: Optional[Model] = None) -> Design:
    """Generate ``n`` random experimental runs.

    Draws independently from ``scipy.stats`` distributions (dict spec) or
    uniform/choice sampling over factor ranges and levels.

    Parameters
    ----------
    factors : dict or sequence of Factor
        Either a dict ``{name: dist}`` of ``scipy.stats`` distributions (uses
        ``.rvs``), or a list of :class:`Factor` (uniform over the range, or a
        choice of levels).
    n : int
        Number of runs to sample.
    seed : int, optional
        Seed for the random generator.
    model : Model, optional
        Model to attach to the design.

    Returns
    -------
    Design
        Random design with ``metadata['kind']='RandomDesign'``.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.random_design(ed.as_factors(2), n=5, seed=0)
    >>> d.n_runs
    5
    """
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}

    if isinstance(factors, dict):
        for name, dist in factors.items():
            columns[name] = dist.rvs(size=n, random_state=rng)
    else:
        for f in factors:
            if isinstance(f, ContinuousFactor):
                columns[f.name] = rng.uniform(f.low, f.high, size=n)
            elif isinstance(f, DiscreteFactor):
                columns[f.name] = rng.choice(f.levels, size=n)
            elif isinstance(f, CategoricalFactor):
                columns[f.name] = rng.choice(np.array(f.levels, dtype=object), size=n)
            else:
                raise TypeError(f"unsupported factor: {f!r}")

    df = pd.DataFrame(columns)
    return Design(matrix=df, model=model, metadata={"kind": "RandomDesign"})


def latin_hypercube(factors, n: int, optimize: bool = False,
                    seed: Optional[int] = None,
                    model: Optional[Model] = None) -> Design:
    """Build a Latin Hypercube design (space-filling sample).

    Each factor is stratified into ``n`` equal-probability intervals with one
    sample per interval. Optional discrepancy minimization improves coverage.

    Parameters
    ----------
    factors : int or sequence of Factor
        An integer (columns in ``[0, 1)``) or continuous/discrete factors (scaled
        to their natural range).
    n : int
        Number of runs to sample.
    optimize : bool, default False
        If ``True``, use scipy's discrepancy-minimizing variant (better space
        coverage).
    seed : int, optional
        Seed for the sampler.
    model : Model, optional
        Model to attach to the design.

    Returns
    -------
    Design
        Latin Hypercube design with ``metadata['kind']='LatinHypercube'``.

    Examples
    --------
    >>> import doekit as ed
    >>> d = ed.latin_hypercube(3, n=10, seed=0)
    >>> d.n_runs, d.n_factors
    (10, 3)
    """
    if isinstance(factors, int):
        names = [f"factor{i + 1}" for i in range(factors)]
        facs: Optional[list] = None
    else:
        facs = list(factors)
        names = [f.name for f in facs]
    d = len(names)

    optimization = "random-cd" if optimize else None
    sampler = qmc.LatinHypercube(d=d, optimization=optimization, seed=seed)
    sample = sampler.random(n)  # in [0, 1)

    if facs is not None:
        cols = []
        for j, f in enumerate(facs):
            if isinstance(f, ContinuousFactor):
                cols.append(f.low + sample[:, j] * (f.high - f.low))
            elif isinstance(f, DiscreteFactor):
                lv = np.asarray(f.levels)
                idx = np.clip((sample[:, j] * len(lv)).astype(int), 0, len(lv) - 1)
                cols.append(lv[idx])
            else:
                raise TypeError("Latin Hypercube supports continuous/discrete factors")
        data = np.column_stack(cols)
    else:
        data = sample

    df = pd.DataFrame(data, columns=names)
    meta = {"kind": "LatinHypercube", "optimized": bool(optimize)}
    return Design(matrix=df, factors=facs or [], model=model, metadata=meta)
