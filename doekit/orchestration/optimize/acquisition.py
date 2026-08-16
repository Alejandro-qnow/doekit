"""Acquisition functions: score candidates on ``(mean, std)`` — larger is better.

Mirror of :mod:`doekit.domain.criteria`: where a criterion scores a *design* for
information, an acquisition scores a *candidate point* for optimization. The
single-objective trio (EI/UCB/PI) trades off exploiting a good predicted mean vs.
exploring a large ``sigma``; the multi-objective :func:`expected_hypervolume_improvement`
extends the same idea to a Pareto front.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from .pareto import hypervolume, to_max_space, default_reference


def _signed_mean(mean: np.ndarray, goal: str) -> np.ndarray:
    """Map to maximization space (``min`` goals get negated)."""
    mean = np.asarray(mean, dtype=float)
    return -mean if str(goal).lower() == "min" else mean


def expected_improvement(mean, std, best: float, goal: str = "max",
                         xi: float = 0.0) -> np.ndarray:
    """Expected improvement over the best observed value (non-negative).

    Scores each candidate by the expected gain over ``best`` under a Gaussian
    surrogate predictive distribution. Larger values favour points with high
    mean and/or high uncertainty.

    Formulas
    --------
    In maximization space with ``imp = mean - best - xi`` and standard normal
    ``z = imp / std`` (when ``std > 0``):

    - ``EI = imp * Phi(z) + std * phi(z)`` where ``Phi`` / ``phi`` are the
      normal CDF and PDF.
    - When ``std == 0``, ``EI = max(imp, 0)``.

    Parameters
    ----------
    mean, std : array-like
        Surrogate predictive mean and standard deviation per candidate.
    best : float
        Best objective observed so far (in the original objective units).
    goal : {"max", "min"}, default "max"
        Optimization direction.
    xi : float, default 0.0
        Exploration margin; larger values favour exploration.

    Returns
    -------
    ndarray
        Expected improvement per candidate (>= 0).

    Examples
    --------
    >>> import doekit as ed
    >>> import numpy as np
    >>> ei = ed.expected_improvement([1.5, 0.5], [0.2, 0.1], best=1.0, goal="max")
    >>> float(ei[0]) > float(ei[1])
    True
    """
    m = _signed_mean(mean, goal)
    best_m = -best if str(goal).lower() == "min" else best
    std = np.asarray(std, dtype=float)
    imp = m - best_m - xi
    out = np.zeros_like(m, dtype=float)
    pos = std > 1e-12
    z = np.zeros_like(m)
    z[pos] = imp[pos] / std[pos]
    out[pos] = imp[pos] * stats.norm.cdf(z[pos]) + std[pos] * stats.norm.pdf(z[pos])
    out[~pos] = np.maximum(imp[~pos], 0.0)
    return np.clip(out, 0.0, None)


def probability_of_improvement(mean, std, best: float, goal: str = "max",
                               xi: float = 0.0) -> np.ndarray:
    """Probability that a candidate improves over ``best``.

    Formulas
    --------
    With ``imp = mean - best - xi`` in maximization space:

    - ``PI = Phi(imp / std)`` when ``std > 0``.
    - ``PI = 1`` when ``imp > 0`` and ``std == 0``, else ``0``.

    Parameters
    ----------
    mean, std : array-like
        Surrogate predictive mean and standard deviation per candidate.
    best : float
        Best objective observed so far (original units).
    goal : {"max", "min"}, default "max"
        Optimization direction.
    xi : float, default 0.0
        Exploration margin.

    Returns
    -------
    ndarray
        Probability of improvement per candidate in ``[0, 1]``.

    Examples
    --------
    >>> import doekit as ed
    >>> pi = ed.probability_of_improvement([1.2], [0.3], best=1.0, goal="max")
    >>> 0.0 <= float(pi[0]) <= 1.0
    True
    """
    m = _signed_mean(mean, goal)
    best_m = -best if str(goal).lower() == "min" else best
    std = np.asarray(std, dtype=float)
    imp = m - best_m - xi
    out = np.where(imp > 0, 1.0, 0.0).astype(float)
    pos = std > 1e-12
    out[pos] = stats.norm.cdf(imp[pos] / std[pos])
    return out


def upper_confidence_bound(mean, std, kappa: float = 2.0,
                           goal: str = "max") -> np.ndarray:
    """Upper (or lower) confidence-bound acquisition — larger is better.

    Formulas
    --------
    In maximization space: ``UCB = mean + kappa * std``.

    For ``goal="min"``, ``mean`` is negated first so the bound still rewards
    high scores.

    Parameters
    ----------
    mean, std : array-like
        Surrogate predictive mean and standard deviation per candidate.
    kappa : float, default 2.0
        Exploration weight on the predictive standard deviation.
    goal : {"max", "min"}, default "max"
        Optimization direction.

    Returns
    -------
    ndarray
        Confidence-bound score per candidate.

    Examples
    --------
    >>> import doekit as ed
    >>> ucb = ed.upper_confidence_bound([0.5, 0.5], [0.1, 0.3], kappa=2.0)
    >>> float(ucb[1]) > float(ucb[0])
    True
    """
    m = _signed_mean(mean, goal)
    std = np.asarray(std, dtype=float)
    return m + float(kappa) * std


def expected_hypervolume_improvement(means, stds, front, ref=None,
                                     goals: Optional[Mapping[str, str]] = None,
                                     columns: Optional[Sequence[str]] = None,
                                     n_samples: int = 128,
                                     seed: Optional[int] = None) -> np.ndarray:
    """Monte-Carlo Expected Hypervolume Improvement for multi-objective candidates.

    Estimates how much the Pareto hypervolume would increase if each candidate
    were observed, by sampling from the surrogate predictive distribution.

    Formulas
    --------
    For candidate ``i`` with predictive ``N(mean_i, std_i)`` per objective:

    - Draw ``S`` samples ``y_s ~ N(mean_i, std_i)``.
    - ``EHVI_i = mean_s max(0, HV(front union {y_s}, ref) - HV(front, ref))``.

    All objectives are converted to maximization space before hypervolume
    computation.

    Parameters
    ----------
    means, stds : array-like, shape (q, m)
        Per-candidate predictive mean / std for each of ``m`` objectives.
    front : array-like, shape (k, m)
        Currently observed objective vectors (original units).
    ref : sequence of float, optional
        Reference point in maximization space; a nadir just below the observed
        front is used when omitted.
    goals : mapping, optional
        ``{column: "max"|"min"}``; columns defaulting to ``"max"``.
    columns : sequence of str, optional
        Objective column names aligned with the last axis (needed with ``goals``).
    n_samples : int, default 128
        Monte-Carlo samples per candidate.
    seed : int, optional
        RNG seed for reproducible EHVI.

    Returns
    -------
    ndarray, shape (q,)
        EHVI per candidate (>= 0).

    Examples
    --------
    >>> import doekit as ed
    >>> import numpy as np
    >>> front = np.array([[1.0, 2.0], [2.0, 1.0]])
    >>> means = np.array([[1.5, 1.5], [0.5, 0.5]])
    >>> stds = np.array([[0.1, 0.1], [0.1, 0.1]])
    >>> ehvi = ed.expected_hypervolume_improvement(
    ...     means, stds, front, seed=0, n_samples=64)
    >>> len(ehvi) == 2 and float(ehvi.max()) >= 0
    True
    """
    means = np.atleast_2d(np.asarray(means, dtype=float))
    stds = np.atleast_2d(np.asarray(stds, dtype=float))
    front = np.atleast_2d(np.asarray(front, dtype=float))
    q, m = means.shape

    front_max = to_max_space(front, goals, columns)
    means_max = to_max_space(means, goals, columns)
    # stds are direction-agnostic (magnitude only)
    if ref is None:
        ref = default_reference(front, goals, columns)
    ref = np.asarray(ref, dtype=float)

    base_hv = hypervolume(front_max, ref)
    rng = np.random.default_rng(seed)
    out = np.zeros(q, dtype=float)
    for i in range(q):
        draws = rng.normal(means_max[i], np.maximum(stds[i], 1e-12),
                           size=(n_samples, m))
        imp = np.empty(n_samples, dtype=float)
        for s in range(n_samples):
            aug = np.vstack([front_max, draws[s]])
            imp[s] = max(0.0, hypervolume(aug, ref) - base_hv)
        out[i] = float(imp.mean())
    return out


_ACQUISITIONS = {
    "ei": expected_improvement,
    "expected_improvement": expected_improvement,
    "pi": probability_of_improvement,
    "probability_of_improvement": probability_of_improvement,
    "ucb": upper_confidence_bound,
    "upper_confidence_bound": upper_confidence_bound,
    "ehvi": expected_hypervolume_improvement,
    "expected_hypervolume_improvement": expected_hypervolume_improvement,
}


def get_acquisition(name: str):
    """Look up an acquisition function by short or long name.

    Parameters
    ----------
    name : str
        Short or long alias (e.g. ``"ei"``, ``"expected_improvement"``,
        ``"ucb"``, ``"ehvi"``).

    Returns
    -------
    callable
        The corresponding acquisition function.

    Raises
    ------
    ValueError
        If ``name`` is not a registered alias.

    Examples
    --------
    >>> import doekit as ed
    >>> fn = ed.get_acquisition("ei")
    >>> fn.__name__ == "expected_improvement"
    True
    """
    key = str(name).strip().lower()
    if key not in _ACQUISITIONS:
        raise ValueError(
            f"unknown acquisition {name!r}; choose from "
            f"{sorted(set(_ACQUISITIONS))}"
        )
    return _ACQUISITIONS[key]
