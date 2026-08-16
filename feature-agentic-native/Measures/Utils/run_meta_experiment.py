#!/usr/bin/env python3
"""Runner para ejecutar el plan meta-experimental generado con DoEkit.

Soporta ciclo operativo:
- prepare
- status
- next-pair [--mark-running]
- start <meta_run_id>
- complete <meta_run_id>
"""

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

TASK_FILES = {
    "Task-01": "TASK_01_PROMPT_UNICO.md",
    "Task-02": "TASK_02_PROMPT_CODIGO_MODELADO.md",
    "Task-03": "TASK_03_PROMPT_ITERACION_Y_DIAGNOSTICO.md",
}
RUN_MANIFEST_FILE = "run_manifest.json"
PROMPT_FILE = "task_prompt.md"
RECOMMENDATION_FILE = "recommendation.json"
METRICS_FILE = "metrics.json"
TRACE_FILE = "trace.log"
EVIDENCE_FILE = "evidence.json"


@dataclass
class MetaRunItem:
    meta_run_id: str
    pair_id: str
    condition: str
    seed: int
    task_id: str
    source_id: str
    dataset_path: str
    difficulty_index: float
    difficulty_stratum: str
    f2_prompt_strictness: int
    f3_context_visibility: int
    f4_timeout_budget: int
    prompt_strictness_level: str
    context_visibility_level: str
    timeout_budget_level: str
    run_order: int
    run_path: Path


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


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


def measures_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root_from_measures(measures_root: Path) -> Path:
    return measures_root.parent


def load_task_prompts(measures_root: Path) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for task_id, filename in TASK_FILES.items():
        path = measures_root / filename
        if not path.exists():
            raise FileNotFoundError(f"No existe prompt de tarea: {path}")
        prompts[task_id] = path.read_text(encoding="utf-8")
    return prompts


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def condition_sandbox(project_root: Path, condition: str) -> Path:
    if condition == "Agent_With":
        return project_root / "Measures" / "Agent_With"
    if condition == "Agent_Without":
        return project_root / "Measures" / "Agent_Without"
    raise ValueError(f"Condicion no soportada: {condition}")


def load_plan(project_root: Path, plan_csv: str) -> List[MetaRunItem]:
    plan_path = project_root / plan_csv
    rows = read_csv_rows(plan_path)
    if not rows:
        raise FileNotFoundError(f"Plan vacio o inexistente: {plan_path}")

    items: List[MetaRunItem] = []
    for row in rows:
        condition = row.get("agent_condition", "")
        sandbox = condition_sandbox(project_root, condition)
        meta_run_id = str(row.get("meta_run_id", "")).strip()
        if not meta_run_id:
            raise ValueError("Fila sin meta_run_id en plan")

        items.append(
            MetaRunItem(
                meta_run_id=meta_run_id,
                pair_id=str(row.get("pair_id", "")).strip(),
                condition=condition,
                seed=parse_int(row.get("seed", 0), 0),
                task_id=str(row.get("task_id", "")).strip(),
                source_id=str(row.get("source_id", "")).strip(),
                dataset_path=str(row.get("dataset_path", "")).strip(),
                difficulty_index=parse_float(row.get("difficulty_index", 0.0), 0.0),
                difficulty_stratum=str(row.get("difficulty_stratum", "")).strip(),
                f2_prompt_strictness=parse_int(row.get("F2_prompt_strictness", 0), 0),
                f3_context_visibility=parse_int(row.get("F3_context_visibility", 0), 0),
                f4_timeout_budget=parse_int(row.get("F4_timeout_budget", 0), 0),
                prompt_strictness_level=str(row.get("prompt_strictness_level", "")).strip(),
                context_visibility_level=str(row.get("context_visibility_level", "")).strip(),
                timeout_budget_level=str(row.get("timeout_budget_level", "")).strip(),
                run_order=parse_int(row.get("run_order", 0), 0),
                run_path=sandbox / meta_run_id,
            )
        )

    items.sort(key=lambda x: x.run_order)
    return items


def default_metrics_template(run: MetaRunItem) -> Dict[str, Any]:
    return {
        "meta_run_id": run.meta_run_id,
        "pair_id": run.pair_id,
        "agent_condition": run.condition,
        "seed": run.seed,
        "task_id": run.task_id,
        "source_id": run.source_id,
        "dataset_path": run.dataset_path,
        "difficulty_index": run.difficulty_index,
        "difficulty_stratum": run.difficulty_stratum,
        "prompt_strictness_level": run.prompt_strictness_level,
        "context_visibility_level": run.context_visibility_level,
        "timeout_budget_level": run.timeout_budget_level,
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


def create_run_files(run: MetaRunItem, task_prompt: str, force: bool = False) -> None:
    run.run_path.mkdir(parents=True, exist_ok=True)

    manifest_path = run.run_path / RUN_MANIFEST_FILE
    prompt_path = run.run_path / PROMPT_FILE
    rec_path = run.run_path / RECOMMENDATION_FILE
    metrics_path = run.run_path / METRICS_FILE
    trace_path = run.run_path / TRACE_FILE
    evidence_path = run.run_path / EVIDENCE_FILE

    if manifest_path.exists() and not force:
        return

    manifest = {
        "meta_run_id": run.meta_run_id,
        "pair_id": run.pair_id,
        "condition": run.condition,
        "seed": run.seed,
        "task_id": run.task_id,
        "status": "pending",
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "doekit_allowed": run.condition == "Agent_With",
        "meta_factors": {
            "F2_prompt_strictness": run.f2_prompt_strictness,
            "F3_context_visibility": run.f3_context_visibility,
            "F4_timeout_budget": run.f4_timeout_budget,
            "prompt_strictness_level": run.prompt_strictness_level,
            "context_visibility_level": run.context_visibility_level,
            "timeout_budget_level": run.timeout_budget_level,
            "difficulty_index": run.difficulty_index,
            "difficulty_stratum": run.difficulty_stratum,
            "source_id": run.source_id,
            "dataset_path": run.dataset_path,
        },
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
        f"# Meta Run {run.meta_run_id}\n\n"
        f"Pair: {run.pair_id}\n\n"
        f"Condition: {run.condition}\n\n"
        f"Task: {run.task_id}\n\n"
        f"Seed: {run.seed}\n\n"
        "## Meta-factores operativos (debes respetarlos)\n"
        f"- prompt_strictness_level: {run.prompt_strictness_level}\n"
        f"- context_visibility_level: {run.context_visibility_level}\n"
        f"- timeout_budget_level: {run.timeout_budget_level}\n"
        f"- difficulty_stratum: {run.difficulty_stratum} (index={run.difficulty_index})\n"
        f"- source_id: {run.source_id}\n"
        f"- dataset_path: {run.dataset_path}\n\n"
        "## Instrucciones especificas de sandbox\n"
        f"- Solo puedes operar en esta carpeta: {run.run_path}\n"
        "- Debes escribir: recommendation.json, metrics.json, trace.log\n"
        "- Debes registrar supuestos y decisiones en trace.log\n"
        "- Debes completar evidence.json con data real y artefactos de codigo\n"
        "- Debes respetar la trazabilidad exacta de source_id y dataset_path\n\n"
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
                "meta_run_id": run.meta_run_id,
                "pair_id": run.pair_id,
                "task_id": run.task_id,
                "condition": run.condition,
                "data_source": {
                    "source_id": run.source_id,
                    "kind": "",
                    "dataset_path": run.dataset_path,
                    "dataset_version": "",
                    "row_count": "",
                },
                "mock_data_used": "",
                "code_artifacts_generated": [],
                "execution_artifacts": [],
                "notes": "",
            },
        )


def get_run_status(run: MetaRunItem) -> str:
    manifest_path = run.run_path / RUN_MANIFEST_FILE
    if not manifest_path.exists():
        return "missing"
    payload = load_json(manifest_path)
    return str(payload.get("status", "pending"))


def update_manifest_status(run: MetaRunItem, to_status: str) -> None:
    manifest_path = run.run_path / RUN_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe {RUN_MANIFEST_FILE} en {run.run_path}")
    payload = load_json(manifest_path)

    payload["status"] = to_status
    if to_status == "running":
        payload["started_at"] = now_iso()
    if to_status == "completed":
        payload["completed_at"] = now_iso()

    write_json(manifest_path, payload)


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


def parse_metrics_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _check_required_artifacts(paths: List[Path]) -> List[str]:
    return [f"Falta artefacto requerido: {p.name}" for p in paths if not p.exists()]


def _validate_data_source_fields(
    evidence: Dict[str, Any], approved_sources: Dict[str, Dict[str, str]]
) -> List[str]:
    errors: List[str] = []
    data_source = evidence.get("data_source", {})
    source_id = str(data_source.get("source_id", "")).strip()

    if str(data_source.get("kind", "")).lower() != "real":
        errors.append("evidence.json: data_source.kind debe ser 'real'")

    if not source_id:
        errors.append("evidence.json: data_source.source_id es obligatorio")
    elif source_id not in approved_sources:
        errors.append("evidence.json: data_source.source_id no esta APPROVED en REAL_DATA_SOURCES.md")

    dataset_path = str(data_source.get("dataset_path", "")).strip()
    if not dataset_path:
        errors.append("evidence.json: data_source.dataset_path es obligatorio")
    elif source_id in approved_sources:
        expected = approved_sources[source_id].get("dataset_path", "")
        if expected and expected != dataset_path:
            errors.append("evidence.json: data_source.dataset_path no coincide con REAL_DATA_SOURCES.md")

    return errors


def _validate_evidence_payload(evidence: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(evidence.get("mock_data_used", "")).lower() not in {"false", "0"}:
        errors.append("evidence.json: mock_data_used debe ser false")

    code_artifacts = evidence.get("code_artifacts_generated", [])
    if not isinstance(code_artifacts, list) or len(code_artifacts) == 0:
        errors.append("evidence.json: code_artifacts_generated debe tener al menos 1 archivo")

    return errors


def _validate_trace_and_metrics(trace_path: Path, metrics_path: Path) -> List[str]:
    errors: List[str] = []
    trace_text = trace_path.read_text(encoding="utf-8")
    if "REAL_DATA_CONFIRMED=true" not in trace_text:
        errors.append("trace.log: debe incluir REAL_DATA_CONFIRMED=true")

    metrics = parse_metrics_json(metrics_path)
    notes = str(metrics.get("notes", "")).lower()
    synthetic_markers = ["baseline manual", "simulado", "mock", "synthetic"]
    if any(marker in notes for marker in synthetic_markers):
        errors.append("metrics.json: notes contiene marcadores de datos sinteticos/mock")

    return errors


def validate_real_data_run(measures_root: Path, run: MetaRunItem) -> List[str]:
    approved_sources = load_approved_sources(measures_root)

    evidence_path = run.run_path / EVIDENCE_FILE
    trace_path = run.run_path / TRACE_FILE
    recommendation_path = run.run_path / RECOMMENDATION_FILE
    metrics_path = run.run_path / METRICS_FILE

    errors = _check_required_artifacts([evidence_path, trace_path, recommendation_path, metrics_path])
    if errors:
        return errors

    evidence = load_json(evidence_path)
    errors.extend(_validate_data_source_fields(evidence, approved_sources))
    errors.extend(_validate_evidence_payload(evidence))
    errors.extend(_validate_trace_and_metrics(trace_path, metrics_path))

    return errors


def upsert_meta_metrics_row(measures_root: Path, row: Dict[str, Any]) -> None:
    csv_path = measures_root / "metrics_meta_template.csv"
    rows = read_csv_rows(csv_path)

    if not rows:
        write_csv_rows(csv_path, [row], list(row.keys()))
        return

    fieldnames = list(rows[0].keys())
    row_norm = {k: row.get(k, "") for k in fieldnames}
    found = False

    for idx, existing in enumerate(rows):
        if existing.get("meta_run_id") == row_norm.get("meta_run_id"):
            rows[idx] = {k: str(row_norm.get(k, "")) for k in fieldnames}
            found = True
            break

    if not found:
        rows.append({k: str(row_norm.get(k, "")) for k in fieldnames})

    write_csv_rows(csv_path, rows, fieldnames)


def locate_run(items: List[MetaRunItem], meta_run_id: str) -> MetaRunItem:
    for run in items:
        if run.meta_run_id == meta_run_id:
            return run
    raise ValueError(f"meta_run_id no encontrado: {meta_run_id}")


def cmd_prepare(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    items = load_plan(project_root, args.plan_csv)
    prompts = load_task_prompts(measures_root)

    for run in items:
        if run.task_id not in prompts:
            raise ValueError(f"Task no soportada en plan: {run.task_id}")
        create_run_files(run, prompts[run.task_id], force=args.force)

    print(f"Plan meta preparado con {len(items)} corridas")


def cmd_status(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    items = load_plan(project_root, args.plan_csv)

    counts = {"pending": 0, "running": 0, "completed": 0, "missing": 0}
    for run in items:
        st = get_run_status(run)
        counts[st] = counts.get(st, 0) + 1

    pairs = {}
    for run in items:
        pairs.setdefault(run.pair_id, []).append(run)

    pending_pairs = 0
    running_pairs = 0
    completed_pairs = 0
    for pair_rows in pairs.values():
        pair_statuses = {get_run_status(r) for r in pair_rows}
        if pair_statuses == {"completed"}:
            completed_pairs += 1
        elif "running" in pair_statuses:
            running_pairs += 1
        elif pair_statuses == {"pending"}:
            pending_pairs += 1

    print("Estado corridas meta")
    for k in ["pending", "running", "completed", "missing"]:
        print(f"- {k}: {counts.get(k, 0)}")

    print("Estado pares")
    print(f"- pending_pairs: {pending_pairs}")
    print(f"- running_pairs: {running_pairs}")
    print(f"- completed_pairs: {completed_pairs}")


def cmd_next_pair(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    items = load_plan(project_root, args.plan_csv)

    by_pair: Dict[str, List[MetaRunItem]] = {}
    for run in items:
        by_pair.setdefault(run.pair_id, []).append(run)

    selected: List[MetaRunItem] = []
    for pair_rows in by_pair.values():
        if len(pair_rows) != 2:
            continue
        statuses = [get_run_status(r) for r in pair_rows]
        if statuses[0] == "pending" and statuses[1] == "pending":
            selected = pair_rows
            break

    if not selected:
        raise RuntimeError("No hay pares pendientes")

    if args.mark_running:
        for run in selected:
            update_manifest_status(run, "running")

    selected = sorted(selected, key=lambda r: r.condition)

    print("Proximo par meta seleccionado")
    print(f"- pair_id: {selected[0].pair_id}")
    for run in selected:
        status = get_run_status(run)
        print(
            f"- {run.meta_run_id} | {run.condition} | {run.task_id} | "
            f"difficulty={run.difficulty_stratum}({run.difficulty_index}) | status={status}"
        )
        print(f"  prompt: {run.run_path / PROMPT_FILE}")


def cmd_start(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    items = load_plan(project_root, args.plan_csv)
    run = locate_run(items, args.meta_run_id)
    update_manifest_status(run, "running")
    print(f"Meta run iniciado: {run.meta_run_id}")
    print(f"Prompt local: {run.run_path / PROMPT_FILE}")


def cmd_complete(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    project_root = project_root_from_measures(measures_root)
    items = load_plan(project_root, args.plan_csv)
    run = locate_run(items, args.meta_run_id)

    if not args.allow_nonreal:
        errors = validate_real_data_run(measures_root, run)
        if errors:
            raise RuntimeError("Validacion de data real fallo:\n- " + "\n- ".join(errors))

    manifest_path = run.run_path / RUN_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError("Debe correr prepare antes de complete")
    manifest = load_json(manifest_path)

    metrics_payload = parse_metrics_json(run.run_path / METRICS_FILE)
    row = default_metrics_template(run)

    for k in row.keys():
        if k in metrics_payload:
            row[k] = metrics_payload[k]

    row["meta_run_id"] = run.meta_run_id
    row["pair_id"] = run.pair_id
    row["agent_condition"] = run.condition
    row["seed"] = run.seed
    row["task_id"] = run.task_id
    row["source_id"] = run.source_id
    row["dataset_path"] = run.dataset_path
    row["difficulty_index"] = run.difficulty_index
    row["difficulty_stratum"] = run.difficulty_stratum
    row["prompt_strictness_level"] = run.prompt_strictness_level
    row["context_visibility_level"] = run.context_visibility_level
    row["timeout_budget_level"] = run.timeout_budget_level
    row["start_ts"] = manifest.get("started_at") or row.get("start_ts", "")
    row["end_ts"] = now_iso()

    if not row.get("impact_score"):
        row["impact_score"] = compute_impact_score(row)

    upsert_meta_metrics_row(measures_root, row)
    update_manifest_status(run, "completed")

    print(f"Meta run completado y consolidado: {run.meta_run_id}")


def cmd_report(args: argparse.Namespace) -> None:
    measures_root = measures_root_from_here()
    csv_path = measures_root / "metrics_meta_template.csv"
    rows = read_csv_rows(csv_path)
    if not rows:
        print("Sin datos en metrics_meta_template.csv")
        return

    grouped: Dict[str, List[float]] = {}
    for r in rows:
        cond = r.get("agent_condition", "")
        score = safe_float(r.get("impact_score", 0.0), 0.0)
        if cond:
            grouped.setdefault(cond, []).append(score)

    print("Resumen ImpactScore (meta experimento)")
    for cond, vals in grouped.items():
        print(f"- {cond}: n={len(vals)}, mean={round(mean(vals), 4)}, median={round(median(vals), 4)}")

    if "Agent_With" in grouped and "Agent_Without" in grouped and grouped["Agent_Without"]:
        mean_with = mean(grouped["Agent_With"]) if grouped["Agent_With"] else 0.0
        mean_without = mean(grouped["Agent_Without"])
        delta = mean_with - mean_without
        pct = (delta / mean_without * 100.0) if mean_without else 0.0
        print(f"Delta mean (With - Without): {round(delta, 4)} ({round(pct, 2)}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runner del meta-experimento con plan CSV")
    parser.add_argument(
        "--plan-csv",
        default="Measures/meta_experiment_plan.csv",
        help="Ruta al plan meta CSV relativa a raiz del proyecto.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_prepare = sub.add_parser("prepare", help="Crea artefactos de corrida desde el plan")
    sp_prepare.add_argument("--force", action="store_true", help="Recrear artefactos existentes")
    sp_prepare.set_defaults(func=cmd_prepare)

    sp_status = sub.add_parser("status", help="Estado de corridas y pares")
    sp_status.set_defaults(func=cmd_status)

    sp_next = sub.add_parser("next-pair", help="Selecciona el siguiente par pendiente")
    sp_next.add_argument("--mark-running", action="store_true", help="Marca el par como running")
    sp_next.set_defaults(func=cmd_next_pair)

    sp_start = sub.add_parser("start", help="Marca una corrida meta como running")
    sp_start.add_argument("meta_run_id", help="Ej: meta_run_00001")
    sp_start.set_defaults(func=cmd_start)

    sp_complete = sub.add_parser("complete", help="Consolida metrics y marca completed")
    sp_complete.add_argument("meta_run_id", help="Ej: meta_run_00001")
    sp_complete.add_argument(
        "--allow-nonreal",
        action="store_true",
        help="Permite consolidar aunque falle validacion estricta de data real (solo depuracion)",
    )
    sp_complete.set_defaults(func=cmd_complete)

    sp_report = sub.add_parser("report", help="Reporte resumido por condicion")
    sp_report.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
