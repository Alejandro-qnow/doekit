#!/usr/bin/env python3
"""Auditoria de integridad y sesgo para runs del protocolo Measures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def inspect_run(measures_root: Path, run_path: Path, approved_sources: Dict[str, Dict[str, str]]) -> Dict[str, object]:
    required = [
        "run_manifest.json",
        "recommendation.json",
        "metrics.json",
        "trace.log",
        "task_prompt.md",
        "evidence.json",
    ]
    missing = [name for name in required if not (run_path / name).exists()]

    synthetic_flags: List[str] = []

    metrics = {}
    recommendation = {}
    trace_text = ""

    if (run_path / "metrics.json").exists():
        metrics = load_json(run_path / "metrics.json")
        notes = str(metrics.get("notes", "")).lower()
        if "baseline manual" in notes or "without doekit" in notes:
            synthetic_flags.append("metrics_notes_baseline_manual")

    if (run_path / "recommendation.json").exists():
        recommendation = load_json(run_path / "recommendation.json")
        method = str(recommendation.get("method", "")).lower()
        if method == "manual_two_level_plus_center":
            synthetic_flags.append("recommendation_manual_method")

    if (run_path / "trace.log").exists():
        trace_text = (run_path / "trace.log").read_text(encoding="utf-8").lower()
        if "heuristic_quality_estimate" in trace_text:
            synthetic_flags.append("trace_heuristic_baseline")
        if "real_data_confirmed=true" not in trace_text:
            synthetic_flags.append("trace_missing_real_data_confirmation")

    if (run_path / "evidence.json").exists():
        evidence = load_json(run_path / "evidence.json")
        data_source = evidence.get("data_source", {})
        source_id = str(data_source.get("source_id", "")).strip()
        if str(data_source.get("kind", "")).lower() != "real":
            synthetic_flags.append("evidence_not_real_data")
        if not source_id:
            synthetic_flags.append("evidence_missing_source_id")
        elif source_id not in approved_sources:
            synthetic_flags.append("evidence_source_not_approved")
        elif str(data_source.get("dataset_path", "")).strip() != approved_sources[source_id].get("dataset_path", ""):
            synthetic_flags.append("evidence_dataset_path_mismatch")
        if str(evidence.get("mock_data_used", "")).lower() not in {"false", "0"}:
            synthetic_flags.append("evidence_mock_data_used")
        code_artifacts = evidence.get("code_artifacts_generated", [])
        if not isinstance(code_artifacts, list) or len(code_artifacts) == 0:
            synthetic_flags.append("evidence_missing_code_artifacts")

    provenance = "likely_simulated_or_scripted" if synthetic_flags else "likely_real_subagent_or_real_tooling"

    return {
        "run_path": str(run_path),
        "missing_files": missing,
        "has_all_required_files": len(missing) == 0,
        "synthetic_flags": synthetic_flags,
        "provenance_assessment": provenance,
        "summary": {
            "condition": recommendation.get("agent_condition"),
            "task_id": recommendation.get("task_id"),
            "method": recommendation.get("method"),
            "seed": recommendation.get("seed") or metrics.get("seed"),
            "d_efficiency": metrics.get("d_efficiency"),
            "impact_score": metrics.get("impact_score"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita runs del protocolo de Measures")
    parser.add_argument("--pair-index", type=int, required=True, help="Indice de par, ej 1 para run_0001")
    args = parser.parse_args()

    measures_root = Path(__file__).resolve().parents[1]
    with_run = measures_root / "Agent_With" / f"run_{args.pair_index:04d}"
    without_run = measures_root / "Agent_Without" / f"run_{args.pair_index:04d}"
    approved_sources = load_approved_sources(measures_root)

    if not with_run.exists() or not without_run.exists():
        raise SystemExit("No existe el par solicitado")

    result = {
        "pair": args.pair_index,
        "approved_sources_count": len(approved_sources),
        "with": inspect_run(measures_root, with_run, approved_sources),
        "without": inspect_run(measures_root, without_run, approved_sources),
    }

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
