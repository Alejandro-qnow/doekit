"""Path helpers and layout constants for experiment workspaces."""

from __future__ import annotations

import re
from pathlib import Path

WAVE_SUBDIRS = (
    "doe-configuration",
    "data",
    "results",
    "reports",
    "automatic-conclusions",
    "metadata",
    "assets",
)

WAVE_STATUSES = (
    "planned",
    "awaiting_response",
    "analyzed",
    "concluded",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Stable filesystem slug from a human project name."""
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("project name must contain alphanumeric characters")
    return s


def project_dirname(name: str) -> str:
    """Directory name under the experiments root (``experiment_project_<slug>``)."""
    return f"experiment_project_{slugify(name)}"


def wave_dirname(index: int) -> str:
    """Zero-padded wave folder name (``wave_001``)."""
    if index < 1:
        raise ValueError("wave index must be >= 1")
    return f"wave_{index:03d}"


def ensure_wave_layout(wave_root: Path) -> Path:
    """Create the standard subdirectories under a wave root."""
    wave_root = Path(wave_root)
    wave_root.mkdir(parents=True, exist_ok=True)
    for name in WAVE_SUBDIRS:
        (wave_root / name).mkdir(parents=True, exist_ok=True)
    return wave_root
