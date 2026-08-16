"""Factor contract (ABC) enforced for all concrete factor types."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Factor(ABC):
    """Common interface for experimental factors.

    Concrete types (:class:`~doekit.domain.factors.ContinuousFactor`,
    :class:`~doekit.domain.factors.DiscreteFactor`, etc.) implement
    :meth:`encode` / :meth:`decode` between natural and coded units and
    :meth:`to_dict` for serialization. Do not instantiate this ABC directly.

    Attributes
    ----------
    name : str
        Factor name (matches a run-matrix column).
    """

    name: str

    @abstractmethod
    def encode(self, values):
        """Map natural-unit values to coded units for model construction."""

    @abstractmethod
    def decode(self, coded):
        """Map coded values back to natural units."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize the factor to a plain ``dict``."""

    @property
    def is_categorical(self) -> bool:
        """Whether the factor is categorical (dummy-coded in the model matrix)."""
        return False
