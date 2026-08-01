"""Shared typing helpers and unit-space policy."""

from __future__ import annotations

from enum import Enum


class Space(str, Enum):
    """Coordinate space for design matrices and metrics.

    Evaluation efficiencies are only scale-invariant in ``CODED`` units.
    Statistical analysis typically fits in ``NATURAL`` units (as stored on
    :class:`~doekit.domain.design.entity.Design`).
    """

    CODED = "coded"
    NATURAL = "natural"
