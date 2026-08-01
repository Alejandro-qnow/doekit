"""Factor contract (ABC) enforced for all concrete factor types."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Factor(ABC):
    """Common interface of a factor. Do not instantiate directly."""

    name: str

    @abstractmethod
    def encode(self, values):
        """Map natural-unit ``values`` to coded units."""

    @abstractmethod
    def decode(self, coded):
        """Map ``coded`` values back to natural units."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize the factor to a plain ``dict``."""

    @property
    def is_categorical(self) -> bool:
        """Whether the factor is categorical (dummy-coded in the model matrix)."""
        return False
