"""Path helpers and layout constants for experiment workspaces.

Defines wave subdirectory names, allowed manifest statuses, slug rules, and
the standard on-disk layout created by :func:`ensure_wave_layout`.
"""

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
    """Derive a stable filesystem slug from a human project name.

    Lowercases, replaces non-alphanumeric runs with ``-``, and strips edges.

    Parameters
    ----------
    name : str
        Human-readable project or experiment name.

    Returns
    -------
    str
        Slug suitable for directory names (e.g. ``"My Study"`` → ``"my-study"``).

    Raises
    ------
    ValueError
        When ``name`` contains no alphanumeric characters.
    """
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("project name must contain alphanumeric characters")
    return s


def project_dirname(name: str) -> str:
    """Directory name for a project under the experiments root.

    Parameters
    ----------
    name : str
        Human-readable project name.

    Returns
    -------
    str
        ``experiment_project_<slug>`` (see :func:`slugify`).
    """
    return f"experiment_project_{slugify(name)}"


def wave_dirname(index: int) -> str:
    """Zero-padded wave folder name.

    Parameters
    ----------
    index : int
        1-based wave sequence number.

    Returns
    -------
    str
        ``wave_NNN`` (e.g. ``wave_001``).

    Raises
    ------
    ValueError
        When ``index`` is less than 1.
    """
    if index < 1:
        raise ValueError("wave index must be >= 1")
    return f"wave_{index:03d}"


def ensure_wave_layout(wave_root: Path) -> Path:
    """Create the standard wave subdirectories if missing.

    Creates :data:`WAVE_SUBDIRS` under ``wave_root`` (configuration, data,
    results, reports, automatic-conclusions, metadata, assets).

    Parameters
    ----------
    wave_root : Path
        Wave directory root.

    Returns
    -------
    Path
        Resolved ``wave_root`` (created when absent).
    """
    wave_root = Path(wave_root)
    wave_root.mkdir(parents=True, exist_ok=True)
    for name in WAVE_SUBDIRS:
        (wave_root / name).mkdir(parents=True, exist_ok=True)
    return wave_root
