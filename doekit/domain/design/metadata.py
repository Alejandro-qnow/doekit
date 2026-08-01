"""Typed helpers around Design.metadata (kind-tagged extension bag)."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class DesignMetadata(TypedDict, total=False):
    """Known metadata keys; unknown keys remain allowed via plain dict merge."""

    kind: str
    resolution: Any
    aliases: Any
    generators: Any
    alpha: Any
    face: Any
    blocking: Any
    criteria: dict
    criterion: str
    algorithm: str
    n_starts: int
    selected_rows: list
    report_path: str


def get_kind(metadata: dict) -> Optional[str]:
    """Return the design ``kind`` tag if present."""
    return metadata.get("kind")


def with_kind(metadata: dict, kind: str) -> dict:
    """Return a copy of ``metadata`` with ``kind`` set."""
    out = dict(metadata)
    out["kind"] = kind
    return out
