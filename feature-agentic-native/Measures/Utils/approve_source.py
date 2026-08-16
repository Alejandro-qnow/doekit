#!/usr/bin/env python3
"""Marca una fuente como APPROVED en REAL_DATA_SOURCES.md de forma controlada."""

from __future__ import annotations

import argparse
from pathlib import Path


def normalize_row(parts: list[str]) -> list[str]:
    cols = parts + [""] * (7 - len(parts))
    return [c.strip() for c in cols[:7]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aprueba una fuente de datos real")
    parser.add_argument("source_id", help="Ej: SRC-001")
    parser.add_argument("--dataset-path", required=True, help="Ruta relativa al workspace")
    parser.add_argument("--version", default="v1", help="Version de dataset")
    parser.add_argument("--owner", required=True, help="Owner responsable")
    parser.add_argument("--provenance", required=True, help="Origen/captura de la data")
    parser.add_argument("--notes", default="", help="Notas de aprobacion")
    args = parser.parse_args()

    measures_root = Path(__file__).resolve().parents[1]
    md_path = measures_root / "REAL_DATA_SOURCES.md"
    if not md_path.exists():
        raise SystemExit(f"No existe: {md_path}")

    lines = md_path.read_text(encoding="utf-8").splitlines()
    updated = False
    out: list[str] = []

    for line in lines:
        if line.strip().startswith("| ") and args.source_id in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            cols = normalize_row(parts)
            if cols[0] == args.source_id:
                cols[1] = args.dataset_path
                cols[2] = args.version
                cols[3] = args.owner
                cols[4] = args.provenance
                cols[5] = "APPROVED"
                cols[6] = args.notes
                line = "| " + " | ".join(cols) + " |"
                updated = True
        out.append(line)

    if not updated:
        appended = (
            f"| {args.source_id} | {args.dataset_path} | {args.version} | {args.owner} | "
            f"{args.provenance} | APPROVED | {args.notes} |"
        )
        out.append(appended)

    md_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Fuente aprobada: {args.source_id}")
    print(f"Archivo: {md_path}")


if __name__ == "__main__":
    main()
