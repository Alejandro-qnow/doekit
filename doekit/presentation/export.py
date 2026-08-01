"""Export run sheets and collection templates (CSV / Excel)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ..domain.design import Design


PathLike = Union[str, Path]


def run_sheet(design: Design, response_names: Optional[Sequence[str]] = None,
              include_run_id: bool = True) -> pd.DataFrame:
    """Build a lab collection template from a design matrix.

    Adds ``run_id`` and empty columns for each response name.
    """
    df = design.matrix.copy().reset_index(drop=True)
    if include_run_id:
        df.insert(0, "run_id", range(1, len(df) + 1))
    for name in (response_names or ("y",)):
        if name not in df.columns:
            df[name] = pd.NA
    return df


def export_csv(design: Design, path: PathLike,
               response_names: Optional[Sequence[str]] = None, **kwargs) -> Path:
    """Write a CSV run sheet. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_sheet(design, response_names=response_names).to_csv(path, index=False, **kwargs)
    return path


def export_excel(design: Design, path: PathLike,
                 response_names: Optional[Sequence[str]] = None,
                 sheet_name: str = "runs") -> Path:
    """Write an Excel run sheet (requires ``openpyxl`` / ``doekit[export]``)."""
    try:
        import openpyxl  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openpyxl is required for Excel export. "
            "Install with: pip install 'doekit[export]'"
        ) from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        run_sheet(design, response_names=response_names).to_excel(
            writer, sheet_name=sheet_name, index=False
        )
    return path
