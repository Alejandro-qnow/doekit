"""
decision.uncertainty - Cuantificacion de incertidumbre para decisiones DoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, pi, sqrt
from typing import Any, Dict, Optional


@dataclass
class UncertaintyEstimate:
    """Estimacion compacta de incertidumbre para Fase 3."""

    sigma_hat: float
    ci_low: float
    ci_high: float
    probability_of_improvement: float
    expected_improvement: float
    normalized_uncertainty: float

    reference_sigma: float = 0.25
    threshold: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma_hat": self.sigma_hat,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "probability_of_improvement": self.probability_of_improvement,
            "expected_improvement": self.expected_improvement,
            "normalized_uncertainty": self.normalized_uncertainty,
            "reference_sigma": self.reference_sigma,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


class UncertaintyQuantifier:
    """Calcula PI/EI e incertidumbre normalizada desde señales del experimento."""

    def __init__(self, reference_sigma: float = 0.25):
        self.reference_sigma = max(1e-9, float(reference_sigma))

    def estimate(
        self,
        expected_gain: float,
        sigma_hat: float,
        threshold: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UncertaintyEstimate:
        sigma = max(1e-9, abs(float(sigma_hat)))
        mu = float(expected_gain)

        pi_value = self.probability_of_improvement(mu, sigma, threshold)
        ei_value = self.expected_improvement(mu, sigma, threshold)

        ci_low, ci_high = self.confidence_interval(mu, sigma)
        normalized = min(1.0, max(0.0, sigma / self.reference_sigma))

        return UncertaintyEstimate(
            sigma_hat=sigma,
            ci_low=ci_low,
            ci_high=ci_high,
            probability_of_improvement=pi_value,
            expected_improvement=ei_value,
            normalized_uncertainty=normalized,
            reference_sigma=self.reference_sigma,
            threshold=threshold,
            metadata=metadata or {},
        )

    def from_proposal(
        self,
        proposal: Any,
        comparison: Any = None,
        threshold: float = 0.0,
    ) -> UncertaintyEstimate:
        comparison = comparison or getattr(proposal, "comparison", None)
        delta = getattr(comparison, "delta", {}) or {}

        d_eff = float(delta.get("D_efficiency", 0.0))
        mean_power = float(delta.get("mean_power", 0.0))
        g_eff = float(delta.get("G_efficiency", 0.0))

        expected_gain = (
            0.6 * self._normalize(d_eff, 20.0)
            + 0.4 * self._normalize(mean_power, 0.2)
        )

        sigma_from_proposal = getattr(proposal, "sigma_hat", None)
        if sigma_from_proposal is None:
            sigma_hat = 0.05 + abs(g_eff) / 20.0
        else:
            sigma_hat = float(sigma_from_proposal)

        return self.estimate(
            expected_gain=expected_gain,
            sigma_hat=sigma_hat,
            threshold=threshold,
            metadata={
                "source": "proposal",
                "d_eff_delta": d_eff,
                "mean_power_delta": mean_power,
                "g_eff_delta": g_eff,
            },
        )

    def confidence_interval(self, mean: float, sigma: float, z: float = 1.96) -> tuple[float, float]:
        return float(mean - z * sigma), float(mean + z * sigma)

    def probability_of_improvement(self, mean: float, sigma: float, threshold: float = 0.0) -> float:
        z = (mean - threshold) / max(1e-9, sigma)
        return self._normal_cdf(z)

    def expected_improvement(self, mean: float, sigma: float, threshold: float = 0.0) -> float:
        sigma = max(1e-9, sigma)
        z = (mean - threshold) / sigma
        return (mean - threshold) * self._normal_cdf(z) + sigma * self._normal_pdf(z)

    def _normalize(self, value: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        x = value / scale
        return max(-1.0, min(1.0, x))

    def _normal_pdf(self, z: float) -> float:
        return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)

    def _normal_cdf(self, z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))
