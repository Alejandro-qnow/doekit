"""Export run sheets and lab collection templates (CSV / Excel).

Run sheets combine the coded design matrix with ``run_id`` and empty response
columns for bench data entry. Used by wave sync and the CLI ``--export`` flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ..domain.design import Design


PathLike = Union[str, Path]


def run_sheet(design: Design, response_names: Optional[Sequence[str]] = None,
              include_run_id: bool = True) -> pd.DataFrame:
    """Build a lab collection template from a design matrix.

    Copies the coded factor columns and appends empty response columns for
    bench entry. ``run_id`` is 1-based row index for traceability.

    Parameters
    ----------
    design : Design
        Design whose factor matrix forms the template rows.
    response_names : sequence of str, optional
        Response column names; defaults to ``("y",)``.
    include_run_id : bool, default True
        When True, insert a ``run_id`` column as the first column.

    Returns
    -------
    DataFrame
        Factor columns plus ``run_id`` (optional) and empty response columns.

    Examples
    --------
    >>> import doekit as ed
    >>> sheet = ed.run_sheet(ed.full_factorial(2))
    >>> list(sheet.columns[:3])
    ['run_id', 'A', 'B']
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
    """Write a CSV run sheet to disk.

    Parameters
    ----------
    design : Design
        Design to export.
    path : str or Path
        Output CSV path (parent directories are created).
    response_names : sequence of str, optional
        Response column names passed to :func:`run_sheet`.
    **kwargs
        Forwarded to :meth:`DataFrame.to_csv` (e.g. ``index=False`` is set internally).

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_sheet(design, response_names=response_names).to_csv(path, index=False, **kwargs)
    return path


def export_excel(design: Design, path: PathLike,
                 response_names: Optional[Sequence[str]] = None,
                 sheet_name: str = "runs") -> Path:
    """Write an Excel run sheet (requires ``openpyxl`` / ``doekit[export]``).

    Parameters
    ----------
    design : Design
        Design to export.
    path : str or Path
        Output ``.xlsx`` path (parent directories are created).
    response_names : sequence of str, optional
        Response column names passed to :func:`run_sheet`.
    sheet_name : str, default ``"runs"``
        Worksheet name in the workbook.

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    ImportError
        When ``openpyxl`` is not installed.
    """
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
