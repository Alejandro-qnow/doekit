"""Rule shortlist for the design advisor (classical methodology)."""

from __future__ import annotations

# Re-export candidate construction from recommend for Open/Closed extension points.
# New design kinds should register builders here rather than editing narrative prose.

from .recommend import _candidate_designs  # noqa: F401

__all__ = ["_candidate_designs"]
