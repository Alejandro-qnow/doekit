#!/usr/bin/env python3
"""Orquesta la ejecucion completa del meta-experimento de forma reproducible.

Flujo:
1) Marca corridas pendientes como running.
2) Ejecuta corridas running con data real y artefactos auditables.
3) Consolida corridas running a completed en metrics_meta_template.csv.
4) Reporta estado final.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from execute_meta_runs_real import execute_meta_run

UTC = timezone.utc


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def resolve_run_path(measures_root: Path, condition: str, meta_run_id: str) -> Path:
    if condition == "Agent_With":
        return measures_root / "Agent_With" / meta_run_id
    if condition == "Agent_Without":
        return measures_root / "Agent_Without" / meta_run_id
    raise ValueError(f"Condicion no soportada: {condition}")


def load_plan(plan_csv: Path) -> List[Dict[str, str]]:
    import csv

    with plan_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r.get("run_order", "0") or "0"))
    return rows


def _row_to_running_candidate(measures_root: Path, row: Dict[str, str]) -> str | None:
    meta_run_id = str(row.get("meta_run_id", "")).strip()
    condition = str(row.get("agent_condition", "")).strip()
    if not meta_run_id or not condition:
        return None

    run_path = resolve_run_path(measures_root, condition, meta_run_id)
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        return None

    manifest = load_json(manifest_path)
    status = str(manifest.get("status", "pending"))
    if status == "completed":
        return None

    if status == "pending":
        manifest["status"] = "running"
        if not manifest.get("started_at"):
            manifest["started_at"] = now_iso()
        save_json(manifest_path, manifest)

    return meta_run_id if manifest.get("status") == "running" else None


def mark_pending_as_running(measures_root: Path, rows: List[Dict[str, str]], limit: int | None) -> List[str]:
    selected: List[str] = []

    for row in rows:
        meta_run_id = _row_to_running_candidate(measures_root, row)
        if meta_run_id:
            selected.append(meta_run_id)

        if limit is not None and len(selected) >= limit:
            break

    return selected


def execute_running(measures_root: Path, project_root: Path, target_ids: List[str]) -> List[str]:
    executed: List[str] = []

    for meta_run_id in target_ids:
        candidate_paths = [
            measures_root / "Agent_With" / meta_run_id,
            measures_root / "Agent_Without" / meta_run_id,
        ]
        run_path = next((p for p in candidate_paths if p.exists()), None)
        if run_path is None:
            continue

        if execute_meta_run(run_path, project_root):
            executed.append(meta_run_id)

    return executed


def consolidate_running(measures_root: Path, target_ids: List[str]) -> List[str]:
    completed: List[str] = []

    cmd_base = [sys.executable, str(measures_root / "Utils" / "run_meta_experiment.py"), "complete"]
    for meta_run_id in target_ids:
        cmd = cmd_base + [meta_run_id]
        proc = subprocess.run(cmd, cwd=str(measures_root), capture_output=True, text=True)
        if proc.returncode == 0:
            completed.append(meta_run_id)
        else:
            print(f"[WARN] No se pudo consolidar {meta_run_id}: {proc.stderr.strip() or proc.stdout.strip()}")

    return completed


def run_status(measures_root: Path) -> str:
    cmd = [sys.executable, str(measures_root / "Utils" / "run_meta_experiment.py"), "status"]
    proc = subprocess.run(cmd, cwd=str(measures_root), capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orquesta ejecucion completa del meta experimento")
    p.add_argument(
        "--plan-csv",
        default="meta_experiment_plan.csv",
        help="Plan CSV dentro de Measures (default: meta_experiment_plan.csv).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita cantidad de corridas a procesar en esta pasada. 0 = sin limite.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    measures_root = Path(__file__).resolve().parents[1]
    project_root = measures_root.parent
    plan_csv = measures_root / args.plan_csv

    if not plan_csv.exists():
        raise FileNotFoundError(f"No existe plan CSV: {plan_csv}")

    rows = load_plan(plan_csv)
    limit = args.limit if args.limit > 0 else None

    target_ids = mark_pending_as_running(measures_root, rows, limit)
    print(f"Corridas objetivo en running: {len(target_ids)}")

    executed_ids = execute_running(measures_root, project_root, target_ids)
    print(f"Corridas ejecutadas: {len(executed_ids)}")

    completed_ids = consolidate_running(measures_root, executed_ids)
    print(f"Corridas consolidadas: {len(completed_ids)}")

    print("Estado final")
    print(run_status(measures_root))


if __name__ == "__main__":
    main()
