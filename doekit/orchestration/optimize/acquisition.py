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
    """Expected improvement over ``best`` (>= 0 everywhere).

    Parameters
    ----------
    mean, std : array-like
        Surrogate predictive mean and standard deviation per candidate.
    best : float
        Best objective observed so far (in the original objective units).
    goal : {"max", "min"}, default "max"
        Optimization direction.
    xi : float, default 0.0
        Exploration margin; larger favours exploration.
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
    """Probability that a candidate improves over ``best`` (in ``[0, 1]``)."""
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
    """Confidence-bound acquisition ``mean + kappa*std`` (max) — larger is better."""
    m = _signed_mean(mean, goal)
    std = np.asarray(std, dtype=float)
    return m + float(kappa) * std


def expected_hypervolume_improvement(means, stds, front, ref=None,
                                     goals: Optional[Mapping[str, str]] = None,
                                     columns: Optional[Sequence[str]] = None,
                                     n_samples: int = 128,
                                     seed: Optional[int] = None) -> np.ndarray:
    """Monte-Carlo Expected Hypervolume Improvement for multi-objective candidates.

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
        RNG seed (deterministic EHVI).

    Returns
    -------
    ndarray, shape (q,)
        EHVI per candidate (>= 0).
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
    """Look up an acquisition function by short or long name."""
    key = str(name).strip().lower()
    if key not in _ACQUISITIONS:
        raise ValueError(
            f"unknown acquisition {name!r}; choose from "
            f"{sorted(set(_ACQUISITIONS))}"
        )
    return _ACQUISITIONS[key]
