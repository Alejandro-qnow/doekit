#!/usr/bin/env python3
"""Descarga/copia datasets publicos a archivos locales versionados para el protocolo."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine


def save_dataset(df: pd.DataFrame, out_csv: Path, metadata: dict) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta_path = out_csv.with_suffix(".metadata.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


def main() -> None:
    measures_root = Path(__file__).resolve().parents[1]
    out_dir = measures_root / "data" / "public"

    # Task-01: optimizacion/regresion tabular con buena escala y variables continuas.
    cal = fetch_california_housing(as_frame=True)
    cal_df = cal.frame.copy()
    save_dataset(
        cal_df,
        out_dir / "california_housing.csv",
        {
            "dataset_id": "california_housing",
            "library": "scikit-learn",
            "license": "public benchmark via sklearn",
            "rows": int(cal_df.shape[0]),
            "cols": int(cal_df.shape[1]),
            "target": "MedHouseVal",
            "recommended_for_task": "Task-01",
            "why": "dataset real tabular, continuo, util para diseno inicial y propuesta secuencial",
        },
    )

    # Task-02: modelado y diagnostico de regresion con target continuo.
    dia = load_diabetes(as_frame=True)
    dia_df = dia.frame.copy()
    save_dataset(
        dia_df,
        out_dir / "diabetes.csv",
        {
            "dataset_id": "diabetes",
            "library": "scikit-learn",
            "license": "public benchmark via sklearn",
            "rows": int(dia_df.shape[0]),
            "cols": int(dia_df.shape[1]),
            "target": "target",
            "recommended_for_task": "Task-02",
            "why": "dataset real pequeno-mediano ideal para ajuste, residuales y colinealidad",
        },
    )

    # Task-03: iteracion/diagnostico multivariable (clasificacion tratable como score continuo/proxy).
    wine = load_wine(as_frame=True)
    wine_df = wine.frame.copy()
    save_dataset(
        wine_df,
        out_dir / "wine.csv",
        {
            "dataset_id": "wine",
            "library": "scikit-learn",
            "license": "public benchmark via sklearn",
            "rows": int(wine_df.shape[0]),
            "cols": int(wine_df.shape[1]),
            "target": "target",
            "recommended_for_task": "Task-03",
            "why": "dataset real compacto para iteracion rapida y comparacion de estrategias",
        },
    )

    print("Datasets preparados en:", out_dir)


if __name__ == "__main__":
    main()
