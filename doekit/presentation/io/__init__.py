"""Filesystem and browser IO for reports."""

from pathlib import Path
import webbrowser


def write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text and return ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def open_browser(url: str | Path) -> None:
    """Open ``url`` in the default browser (best-effort)."""
    webbrowser.open(str(url))
