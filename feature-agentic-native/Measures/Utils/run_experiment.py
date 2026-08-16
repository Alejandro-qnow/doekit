#!/usr/bin/env python3
"""Runner para experimento Con vs Sin DoEkit usando subagentes en sandbox."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List


UTC = timezone.utc


@dataclass
class RunItem:
    run_id: str
    condition: str
    sandbox: Path
    seed: int
    run_path: Path
    task_id: str


TASK_FILES = {
    "Task-01": "TASK_01_PROMPT_UNICO.md",
    "Task-02": "TASK_02_PROMPT_CODIGO_MODELADO.md",
    "Task-03": "TASK_03_PROMPT_ITERACION_Y_DIAGNOSTICO.md",
}


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def measures_root_from_here() -> Path:
    # Utils/run_experiment.py -> Measures/
    return Path(__file__).resolve().parents[1]


def project_root_from_measures(measures_root: Path) -> Path:
    # .../doekit-sggestions/Measures -> .../doekit-sggestions
    return measures_root.parent


def load_config(measures_root: Path, config_name: str) -> Dict[str, Any]:
    config_path = measures_root / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"No existe config: {config_path}")
    return load_json(config_path)


def load_task_prompts(measures_root: Path) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for task_id, file_name in TASK_FILES.items():
        prompt_path = measures_root / file_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"No existe prompt de tarea: {prompt_path}")
        prompts[task_id] = prompt_path.read_text(encoding="utf-8")
    return prompts


def condition_sandbox(project_root: Path, condition: Dict[str, Any]) -> Path:
    raw = condition["sandbox_path"].replace("/", "\\")
    return project_root / raw


def pick_seeds(config: Dict[str, Any], runs_per_condition: int) -> List[int]:
    seeds = list(config.get("seeds", []))
    if len(seeds) >= runs_per_condition:
        return seeds[:runs_per_condition]
    if not seeds:
        seeds = [1001]
    cur = seeds[-1]
    while len(seeds) < runs_per_condition:
        cur += 1
        seeds.append(cur)
    return seeds


def build_plan(project_root: Path, measures_root: Path, config: Dict[str, Any]) -> List[RunItem]:
    runs_per_condition = int(config["runs_per_condition"])
    seeds = pick_seeds(config, runs_per_condition)
    plan: List[RunItem] = []

    for condition in config["conditions"]:
        condition_name = condition["name"]
        sandbox = condition_sandbox(project_root, condition)
        for idx in range(1, runs_per_condition + 1):
            task_id = f"Task-{((idx - 1) % 3) + 1:02d}"
            run_id = f"{condition_name.lower()}_{idx:04d}"
            run_path = sandbox / f"run_{idx:04d}"
            plan.append(
                RunItem(
                    run_id=run_id,
                    condition=condition_name,
                    sandbox=sandbox,
                    seed=seeds[idx - 1],
                    run_path=run_path,
                    task_id=task_id,
                )
            )
    return plan


def default_metrics_template(run: RunItem) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "agent_condition": run.condition,
        "seed": run.seed,
        "task_id": run.task_id,
        "start_ts": "",
        "end_ts": "",
        "total_time_sec": "",
        "time_to_first_valid_plan_sec": "",
        "iterations_count": "",
        "d_efficiency": "",
        "mean_power": "",
        "predicted_gain": "",
        "uncertainty_index": "",
        "budget_used_ratio": "",
        "invalid_assumptions_count": "",
        "budget_violations_count": "",
        "model_mismatch_flags": "",
        "decision_reversal_count": "",
        "output_completeness_score": "",
        "format_compliance_score": "",
        "impact_score": "",
        "notes": "",
    }


def create_run_files(run: RunItem, task_prompt: str, force: bool = False) -> None:
    run.run_path.mkdir(parents=True, exist_ok=True)

    manifest_path = run.run_path / "run_manifest.json"
    prompt_path = run.run_path / "task_prompt.md"
    rec_path = run.run_path / "recommendation.json"
    metrics_path = run.run_path / "metrics.json"
    trace_path = run.run_path / "trace.log"
    evidence_path = run.run_path / "evidence.json"

    if manifest_path.exists() and not force:
        return

    manifest = {
        "run_id": run.run_id,
        "condition": run.condition,
        "seed": run.seed,
        "status": "pending",
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "doekit_allowed": run.condition == "Agent_With",
        "task_id": run.task_id,
        "artifacts": {
            "prompt": str(prompt_path.name),
            "manifest": str(manifest_path.name),
            "recommendation": str(rec_path.name),
            "metrics": str(metrics_path.name),
            "trace": str(trace_path.name),
            "evidence": str(evidence_path.name),
        },
    }

    scoped_prompt = (
        f"# Run {run.run_id}\n\n"
        f"Condition: {run.condition}\n\n"
        f"Task: {run.task_id}\n\n"
        f"Seed: {run.seed}\n\n"
        "## Instrucciones especificas de sandbox\n"
        f"- Solo puedes operar en esta carpeta: {run.run_path}\n"
        "- Debes escribir: recommendation.json, metrics.json, trace.log\n"
        "- Debes registrar supuestos y decisiones en trace.log\n"
        "- Debes completar evidence.json con data real y artefactos de codigo\n\n"
        "---\n\n"
        f"{task_prompt}\n"
    )

    write_json(manifest_path, manifest)
    prompt_path.write_text(scoped_prompt, encoding="utf-8")
    if not rec_path.exists() or force:
        write_json(rec_path, {})
    if not metrics_path.exists() or force:
        write_json(metrics_path, default_metrics_template(run))
    if not trace_path.exists() or force:
        trace_path.write_text("REAL_DATA_CONFIRMED=\n", encoding="utf-8")
    if not evidence_path.exists() or force:
        write_json(
            evidence_path,
            {
                "run_id": run.run_id,
                "task_id": run.task_id,
                "condition": run.condition,
                "data_source": {
                    "source_id": "",
                    "kind": "",
                    "dataset_path": "",
                    "dataset_version": "",
                    "row_count": "",
                },
                "mock_data_used": "",
                "code_artifacts_generated": [],
                "execution_artifacts": [],
                "notes": "",
            },
        )


def write_schedule(measures_root: Path, plan: List[RunItem]) -> Path:
    out_path = measures_root / "Utils" / "run_schedule.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "condition", "task_id", "seed", "run_path", "status"])
        for run in plan:
            status = "pending"
            manifest = run.run_path / "run_manifest.json"
            if manifest.exists():
                payload = load_json(manifest)
                status = payload.get("status", status)
            writer.writerow([run.run_id, run.condition, run.task_id, run.seed, str(run.run_path), status])
    return out_path


def load_approved_sources(measures_root: Path) -> Dict[str, Dict[str, str]]:
    path = measures_root / "REAL_DATA_SOURCES.md"
    approved: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return approved

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("| SRC-"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        source_id, dataset_path, version, owner, provenance, status, notes = parts[:7]
        if status.upper() == "APPROVED":
            approved[source_id] = {
                "dataset_path": dataset_path,
                "version": version,
                "owner": owner,
                "provenance": provenance,
                "notes": notes,
            }
    return approved


def validate_real_data_run(measures_root: Path, run: RunItem) -> List[str]:
    errors: List[str] = []
    approved_sources = load_approved_sources(measures_root)

    evidence_path = run.run_path / "evidence.json"
    trace_path = run.run_path / "trace.log"
    recommendation_path = run.run_path / "recommendation.json"
    metrics_path = run.run_path / "metrics.json"

    for p in [evidence_path, trace_path, recommendation_path, metrics_path]:
        if not p.exists():
            errors.append(f"Falta artefacto requerido: {p.name}")

    if errors:
        return errors

    evidence = load_json(evidence_path)
    data_source = evidence.get("data_source", {})
    source_id = str(data_source.get("source_id", "")).strip()
    if str(data_source.get("kind", "")).lower() != "real":
        errors.append("evidence.json: data_source.kind debe ser 'real'")
    if not source_id:
        errors.append("evidence.json: data_source.source_id es obligatorio")
    elif source_id not in approved_sources:
        errors.append("evidence.json: data_source.source_id no esta APPROVED en REAL_DATA_SOURCES.md")
    if not str(data_source.get("dataset_path", "")).strip():
        errors.append("evidence.json: data_source.dataset_path es obligatorio")
    elif source_id in approved_sources:
        expected_path = approved_sources[source_id].get("dataset_path", "")
        if expected_path and str(data_source.get("dataset_path", "")).strip() != expected_path:
            errors.append("evidence.json: data_source.dataset_path no coincide con REAL_DATA_SOURCES.md")
    if str(evidence.get("mock_data_used", "")).lower() not in {"false", "0"}:
        errors.append("evidence.json: mock_data_used debe ser false")

    code_artifacts = evidence.get("code_artifacts_generated", [])
    if not isinstance(code_artifacts, list) or len(code_artifacts) == 0:
        errors.append("evidence.json: code_artifacts_generated debe tener al menos 1 archivo")

    trace_text = trace_path.read_text(encoding="utf-8")
    if "REAL_DATA_CONFIRMED=true" not in trace_text:
        errors.append("trace.log: debe incluir REAL_DATA_CONFIRMED=true")

    metrics = parse_metrics_json(metrics_path)
    notes = str(metrics.get("notes", "")).lower()
    synthetic_markers = ["baseline manual", "simulado", "mock", "synthetic"]
    if any(marker in notes for marker in synthetic_markers):
        errors.append("metrics.json: notes contiene marcadores de datos sinteticos/mock")

    return errors


def update_manifest_status(run_path: Path, to_status: str) -> None:
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe run_manifest.json en {run_path}")
    payload = load_json(manifest_path)

    payload["status"] = to_status
    if to_status == "running":
        payload["started_at"] = now_iso()
    if to_status == "completed":
        payload["completed_at"] = now_iso()

    write_json(manifest_path, payload)


def parse_metrics_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def upsert_metrics_row(measures_root: Path, row: Dict[str, Any]) -> None:
    csv_path = measures_root / "metrics_template.csv"
    rows = read_csv_rows(csv_path)
    if not rows:
        fieldnames = list(row.keys())
        write_csv_rows(csv_path, [row], fieldnames)
        return

    fieldnames = list(rows[0].keys())
    row = {k: row.get(k, "") for k in fieldnames}
    found = False
    for idx, existing in enumerate(rows):
        if existing.get("run_id") == row.get("run_id"):
            rows[idx] = {k: str(row.get(k, "")) for k in fieldnames}
            found = True
            break
    if not found:
        rows.append({k: str(row.get(k, "")) for k in fieldnames})

    write_csv_rows(csv_path, rows, fieldnames)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize_positive(value: float, target_good: float) -> float:
    if target_good <= 0:
        return 0.0
    return max(0.0, min(100.0, (value / target_good) * 100.0))


def normalize_inverse(value: float, target_bad: float) -> float:
    if target_bad <= 0:
        return 0.0
    return max(0.0, min(100.0, (1.0 - (value / target_bad)) * 100.0))


def compute_impact_score(row: Dict[str, Any]) -> float:
    quality = mean(
        [
            normalize_positive(safe_float(row.get("d_efficiency", 0.0)), 1.0),
            normalize_positive(safe_float(row.get("mean_power", 0.0)), 1.0),
            normalize_positive(safe_float(row.get("predicted_gain", 0.0)), 1.0),
            normalize_inverse(safe_float(row.get("uncertainty_index", 1.0)), 1.0),
        ]
    )

    efficiency = mean(
        [
            normalize_inverse(safe_float(row.get("total_time_sec", 99999.0)), 900.0),
            normalize_inverse(safe_float(row.get("time_to_first_valid_plan_sec", 99999.0)), 300.0),
            normalize_inverse(safe_float(row.get("iterations_count", 99.0)), 10.0),
        ]
    )

    risk = mean(
        [
            normalize_inverse(safe_float(row.get("invalid_assumptions_count", 0.0)), 5.0),
            normalize_inverse(safe_float(row.get("budget_violations_count", 0.0)), 3.0),
            normalize_inverse(safe_float(row.get("model_mismatch_flags", 0.0)), 3.0),
            normalize_inverse(safe_float(row.get("decision_reversal_count", 0.0)), 3.0),
        ]
    )

    ux = mean(
        [
            normalize_positive(safe_float(row.get("output_completeness_score", 0.0)), 1.0),
            normalize_positive(safe_float(row.get("format_compliance_score", 0.0)), 1.0),
        ]
    )

    impact = (0.35 * quality) + (0.30 * efficiency) + (0.20 * risk) + (0.15 * ux)
    return round(impact, 4)


def cmd_prepare(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    config = load_config(measures_root, args.config)
    task_prompts = load_task_prompts(measures_root)
    plan = build_plan(project_root, measures_root, config)

    for run in plan:
        create_run_files(run, task_prompt=task_prompts[run.task_id], force=args.force)

    schedule_path = write_schedule(measures_root, plan)
    print(f"Preparado plan con {len(plan)} runs")
    print(f"Schedule: {schedule_path}")


def cmd_status(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    config = load_config(measures_root, args.config)
    plan = build_plan(project_root, measures_root, config)

    counts = {"pending": 0, "running": 0, "completed": 0, "missing": 0}
    for run in plan:
        manifest_path = run.run_path / "run_manifest.json"
        if not manifest_path.exists():
            counts["missing"] += 1
            continue
        payload = load_json(manifest_path)
        status = payload.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1

    print("Estado de runs")
    for k in ["pending", "running", "completed", "missing"]:
        print(f"- {k}: {counts.get(k, 0)}")


def locate_run(plan: List[RunItem], run_id: str) -> RunItem:
    for run in plan:
        if run.run_id == run_id:
            return run
    raise ValueError(f"run_id no encontrado: {run_id}")


def get_run_status(run: RunItem) -> str:
    manifest_path = run.run_path / "run_manifest.json"
    if not manifest_path.exists():
        return "missing"
    payload = load_json(manifest_path)
    return payload.get("status", "pending")


def cmd_start(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    config = load_config(measures_root, args.config)
    plan = build_plan(project_root, measures_root, config)
    run = locate_run(plan, args.run_id)
    update_manifest_status(run.run_path, "running")
    print(f"Run iniciado: {run.run_id}")
    print(f"Prompt local: {run.run_path / 'task_prompt.md'}")


def cmd_next_pair(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    config = load_config(measures_root, args.config)
    plan = build_plan(project_root, measures_root, config)

    by_condition: Dict[str, List[RunItem]] = {}
    for run in plan:
        by_condition.setdefault(run.condition, []).append(run)

    required = ["Agent_With", "Agent_Without"]
    selected: List[RunItem] = []
    for cond in required:
        candidates = by_condition.get(cond, [])
        next_pending = None
        for run in candidates:
            if get_run_status(run) == "pending":
                next_pending = run
                break
        if next_pending is None:
            raise RuntimeError(f"No hay runs pendientes para {cond}")
        selected.append(next_pending)

    if args.mark_running:
        for run in selected:
            update_manifest_status(run.run_path, "running")

    print("Proximo par seleccionado")
    for run in selected:
        status = get_run_status(run)
        print(f"- {run.run_id} | {run.condition} | {run.task_id} | status={status}")
        print(f"  prompt: {run.run_path / 'task_prompt.md'}")


def cmd_complete(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    config = load_config(measures_root, args.config)
    plan = build_plan(project_root, measures_root, config)
    run = locate_run(plan, args.run_id)

    if not args.allow_nonreal:
        errors = validate_real_data_run(measures_root, run)
        if errors:
            raise RuntimeError("Validacion de data real fallo:\n- " + "\n- ".join(errors))

    manifest_path = run.run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Debe correr prepare antes de complete")
    manifest = load_json(manifest_path)

    metrics_payload = parse_metrics_json(run.run_path / "metrics.json")
    row = default_metrics_template(run)

    for k in row.keys():
        if k in metrics_payload:
            row[k] = metrics_payload[k]

    row["run_id"] = run.run_id
    row["agent_condition"] = run.condition
    row["seed"] = run.seed
    row["start_ts"] = manifest.get("started_at") or row.get("start_ts", "")
    row["end_ts"] = now_iso()

    if not row.get("impact_score"):
        row["impact_score"] = compute_impact_score(row)

    upsert_metrics_row(measures_root, row)
    update_manifest_status(run.run_path, "completed")

    print(f"Run completado y consolidado: {run.run_id}")


def cmd_report(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    csv_path = measures_root / "metrics_template.csv"
    rows = read_csv_rows(csv_path)
    if not rows:
        print("Sin datos en metrics_template.csv")
        return

    grouped: Dict[str, List[float]] = {}
    for r in rows:
        cond = r.get("agent_condition", "")
        if not cond:
            continue
        score = safe_float(r.get("impact_score", 0.0), 0.0)
        grouped.setdefault(cond, []).append(score)

    print("Resumen ImpactScore")
    for cond, vals in grouped.items():
        print(f"- {cond}: n={len(vals)}, mean={round(mean(vals), 4)}, median={round(median(vals), 4)}")

    if "Agent_With" in grouped and "Agent_Without" in grouped and grouped["Agent_Without"]:
        mean_with = mean(grouped["Agent_With"]) if grouped["Agent_With"] else 0.0
        mean_without = mean(grouped["Agent_Without"])
        delta = mean_with - mean_without
        pct = (delta / mean_without * 100.0) if mean_without else 0.0
        print(f"Delta mean (With - Without): {round(delta, 4)} ({round(pct, 2)}%)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Runner de experimento Con vs Sin DoEkit")
    p.add_argument("--config", default="experiment_config.json", help="Config JSON dentro de Measures")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_prepare = sub.add_parser("prepare", help="Crea plan y artefactos de run")
    sp_prepare.add_argument("--force", action="store_true", help="Recrear artefactos existentes")
    sp_prepare.set_defaults(func=cmd_prepare)

    sp_status = sub.add_parser("status", help="Resumen de estados")
    sp_status.set_defaults(func=cmd_status)

    sp_start = sub.add_parser("start", help="Marca run como running")
    sp_start.add_argument("run_id", help="Ej: agent_with_0001")
    sp_start.set_defaults(func=cmd_start)

    sp_next_pair = sub.add_parser(
        "next-pair",
        help="Selecciona siguiente par balanceado (With/Without) y opcionalmente lo marca running",
    )
    sp_next_pair.add_argument(
        "--mark-running",
        action="store_true",
        help="Marca ambos runs del par como running",
    )
    sp_next_pair.set_defaults(func=cmd_next_pair)

    sp_complete = sub.add_parser("complete", help="Consolida metrics.json y marca completed")
    sp_complete.add_argument("run_id", help="Ej: agent_with_0001")
    sp_complete.add_argument(
        "--allow-nonreal",
        action="store_true",
        help="Permite consolidar aunque falle validacion estricta de data real (solo depuracion)",
    )
    sp_complete.set_defaults(func=cmd_complete)

    sp_report = sub.add_parser("report", help="Reporte resumido por condicion")
    sp_report.set_defaults(func=cmd_report)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
