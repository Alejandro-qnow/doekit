#!/usr/bin/env python3
"""Genera un meta-diseno experimental robusto usando DoEkit y bloqueo por dificultad.

Objetivo:
- Disenar un plan de corridas para estimar el efecto de DoEkit y su interaccion
  con dificultad y condiciones operativas (prompt, contexto, timeout).

Salida:
- Measures/meta_experiment_plan.csv
- Measures/meta_experiment_plan_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import doekit as ed
from sklearn.linear_model import LinearRegression


TASK_SOURCE_MAP: Dict[str, Dict[str, str]] = {
    "Task-01": {
        "source_id": "SRC-101",
        "dataset_path": "Measures/data/public/california_housing.csv",
        "target": "MedHouseVal",
    },
    "Task-02": {
        "source_id": "SRC-102",
        "dataset_path": "Measures/data/public/diabetes.csv",
        "target": "target",
    },
    "Task-03": {
        "source_id": "SRC-103",
        "dataset_path": "Measures/data/public/wine.csv",
        "target": "target",
    },
}


PROMPT_STRICTNESS = {-1: "medium", 1: "high"}
CONTEXT_VISIBILITY = {-1: "task_only", 1: "task_plus_history"}
TIMEOUT_BUDGET = {-1: "short", 1: "long"}
AGENT_CONDITION = {-1: "Agent_Without", 1: "Agent_With"}


@dataclass
class TaskDifficulty:
    task_id: str
    source_id: str
    dataset_path: str
    rows: int
    n_features: int
    collinearity_mean_abs: float
    linear_noise_proxy: float
    difficulty_index: float
    difficulty_stratum: str


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_linear_noise_proxy(x: pd.DataFrame, y: pd.Series) -> float:
    if len(x) < 10 or x.shape[1] < 1:
        return 0.7
    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)
    var_y = float(np.var(y.values))
    if var_y <= 1e-12:
        return 0.5
    r2 = 1.0 - float(np.var(y.values - pred) / var_y)
    return clip01(1.0 - max(-1.0, min(1.0, r2)))


def compute_task_difficulty(project_root: Path, task_id: str, cfg: Dict[str, str]) -> Dict[str, Any]:
    csv_path = project_root / cfg["dataset_path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado para {task_id}: {csv_path}")

    df = pd.read_csv(csv_path)
    target = cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe en {cfg['dataset_path']}")

    x = df.drop(columns=[target])
    y = df[target]

    num_cols = list(x.select_dtypes(include=[np.number]).columns)
    x_num = x[num_cols].copy() if num_cols else x.copy()
    n_rows = int(len(df))
    n_features = int(x_num.shape[1])

    if n_features >= 2 and n_rows >= 10:
        corr = x_num.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        collinearity = float(np.nanmean(upper.values)) if np.isfinite(np.nanmean(upper.values)) else 0.0
    else:
        collinearity = 0.0

    noise_proxy = compute_linear_noise_proxy(x_num if n_features > 0 else x, y)

    size_score = clip01(np.log10(max(10, n_rows)) / 5.0)
    dim_score = clip01(n_features / 20.0)
    col_score = clip01(collinearity)
    noise_score = clip01(noise_proxy)

    difficulty = 100.0 * (0.30 * size_score + 0.25 * dim_score + 0.20 * col_score + 0.25 * noise_score)

    return {
        "task_id": task_id,
        "source_id": cfg["source_id"],
        "dataset_path": cfg["dataset_path"],
        "rows": n_rows,
        "n_features": n_features,
        "collinearity_mean_abs": round(collinearity, 6),
        "linear_noise_proxy": round(noise_proxy, 6),
        "difficulty_index": round(float(difficulty), 4),
    }


def assign_strata(task_rows: List[Dict[str, Any]]) -> List[TaskDifficulty]:
    vals = np.array([r["difficulty_index"] for r in task_rows], dtype=float)
    q1 = float(np.quantile(vals, 0.33))
    q2 = float(np.quantile(vals, 0.66))

    out: List[TaskDifficulty] = []
    for r in task_rows:
        d = float(r["difficulty_index"])
        if d <= q1:
            s = "low"
        elif d <= q2:
            s = "medium"
        else:
            s = "high"
        out.append(
            TaskDifficulty(
                task_id=str(r["task_id"]),
                source_id=str(r["source_id"]),
                dataset_path=str(r["dataset_path"]),
                rows=int(r["rows"]),
                n_features=int(r["n_features"]),
                collinearity_mean_abs=float(r["collinearity_mean_abs"]),
                linear_noise_proxy=float(r["linear_noise_proxy"]),
                difficulty_index=d,
                difficulty_stratum=s,
            )
        )
    return out


def generate_base_design_with_doekit() -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    # Meta-diseno para factores operativos (sin F1), luego se cruza con F1 para pares justos.
    factors = {
        "F2_prompt_strictness": (-1, 1),
        "F3_context_visibility": (-1, 1),
        "F4_timeout_budget": (-1, 1),
    }
    rec = ed.recommend_design(goal="optimization", factors=factors, budget=8, model_order="linear")
    d = rec.design.to_dict()

    cols = d["matrix"]["columns"]
    data = d["matrix"]["data"]

    rows: List[Dict[str, int]] = []
    for r in data:
        mapped: Dict[str, int] = {}
        for c, v in zip(cols, r):
            mapped[c] = 1 if float(v) >= 0 else -1
        rows.append(mapped)

    design_meta = {
        "method": rec.method,
        "rationale": rec.rationale,
        "base_runs": len(rows),
        "factor_columns": cols,
    }
    return rows, design_meta


def build_plan(
    project_root: Path,
    target_total_runs: int,
    random_seed: int,
    seed_start: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    task_raw = [compute_task_difficulty(project_root, t, cfg) for t, cfg in TASK_SOURCE_MAP.items()]
    tasks = assign_strata(task_raw)

    base_rows, design_meta = generate_base_design_with_doekit()

    runs_per_replicate = len(tasks) * len(base_rows) * 2
    replicates = int(np.ceil(target_total_runs / float(runs_per_replicate)))

    rng = np.random.default_rng(random_seed)

    pairs: List[List[Dict[str, Any]]] = []
    pair_seq = 1
    seed_seq = seed_start

    for rep in range(1, replicates + 1):
        for t in tasks:
            for cell_idx, cell in enumerate(base_rows, start=1):
                pair_id = f"PAIR_{pair_seq:04d}"
                pair_seq += 1

                base_common = {
                    "replicate": rep,
                    "task_id": t.task_id,
                    "source_id": t.source_id,
                    "dataset_path": t.dataset_path,
                    "difficulty_index": round(t.difficulty_index, 4),
                    "difficulty_stratum": t.difficulty_stratum,
                    "rows": t.rows,
                    "n_features": t.n_features,
                    "collinearity_mean_abs": round(t.collinearity_mean_abs, 6),
                    "linear_noise_proxy": round(t.linear_noise_proxy, 6),
                    "doe_cell": cell_idx,
                    "pair_id": pair_id,
                    "seed": seed_seq,
                    "F2_prompt_strictness": cell["F2_prompt_strictness"],
                    "F3_context_visibility": cell["F3_context_visibility"],
                    "F4_timeout_budget": cell["F4_timeout_budget"],
                }

                row_without = dict(base_common)
                row_without.update(
                    {
                        "F1_doekit_access": -1,
                        "agent_condition": AGENT_CONDITION[-1],
                        "prompt_strictness_level": PROMPT_STRICTNESS[cell["F2_prompt_strictness"]],
                        "context_visibility_level": CONTEXT_VISIBILITY[cell["F3_context_visibility"]],
                        "timeout_budget_level": TIMEOUT_BUDGET[cell["F4_timeout_budget"]],
                    }
                )

                row_with = dict(base_common)
                row_with.update(
                    {
                        "F1_doekit_access": 1,
                        "agent_condition": AGENT_CONDITION[1],
                        "prompt_strictness_level": PROMPT_STRICTNESS[cell["F2_prompt_strictness"]],
                        "context_visibility_level": CONTEXT_VISIBILITY[cell["F3_context_visibility"]],
                        "timeout_budget_level": TIMEOUT_BUDGET[cell["F4_timeout_budget"]],
                    }
                )

                pair_rows = [row_without, row_with]
                if rng.uniform() < 0.5:
                    pair_rows = [row_with, row_without]
                pairs.append(pair_rows)
                seed_seq += 1

    rng.shuffle(pairs)

    final_rows: List[Dict[str, Any]] = []
    run_order = 1
    run_seq = 1
    for pair in pairs:
        for r in pair:
            rr = dict(r)
            rr["run_order"] = run_order
            rr["meta_run_id"] = f"meta_run_{run_seq:05d}"
            final_rows.append(rr)
            run_order += 1
            run_seq += 1

    df = pd.DataFrame(final_rows)

    summary = {
        "design_method": design_meta["method"],
        "design_rationale": design_meta["rationale"],
        "target_total_runs": int(target_total_runs),
        "actual_total_runs": int(len(df)),
        "replicates": replicates,
        "pairs": int(len(df) // 2),
        "runs_per_replicate": int(runs_per_replicate),
        "tasks": [t.task_id for t in tasks],
        "difficulty": [
            {
                "task_id": t.task_id,
                "difficulty_index": round(t.difficulty_index, 4),
                "difficulty_stratum": t.difficulty_stratum,
                "rows": t.rows,
                "n_features": t.n_features,
            }
            for t in tasks
        ],
        "balance": {
            "by_condition": df["agent_condition"].value_counts().to_dict(),
            "by_task": df["task_id"].value_counts().to_dict(),
            "by_stratum": df["difficulty_stratum"].value_counts().to_dict(),
            "by_prompt": df["prompt_strictness_level"].value_counts().to_dict(),
            "by_context": df["context_visibility_level"].value_counts().to_dict(),
            "by_timeout": df["timeout_budget_level"].value_counts().to_dict(),
        },
    }

    return df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera plan meta-experimental con DoEkit")
    parser.add_argument(
        "--target-total-runs",
        type=int,
        default=120,
        help="Cantidad objetivo total de corridas (el plan final sera >= objetivo).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260814,
        help="Semilla para aleatorizacion de orden de ejecucion.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=3001,
        help="Semilla inicial para asignar a cada par experimental.",
    )
    parser.add_argument(
        "--output-csv",
        default="Measures/meta_experiment_plan.csv",
        help="Ruta de salida CSV relativa a raiz de proyecto.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="Measures/meta_experiment_plan_summary.json",
        help="Ruta de salida JSON relativa a raiz de proyecto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    measures_root = Path(__file__).resolve().parents[1]
    project_root = measures_root.parent

    df, summary = build_plan(
        project_root=project_root,
        target_total_runs=args.target_total_runs,
        random_seed=args.random_seed,
        seed_start=args.seed_start,
    )

    csv_path = project_root / args.output_csv
    json_path = project_root / args.output_summary_json

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Plan generado: {csv_path}")
    print(f"Resumen: {json_path}")
    print(f"Corridas totales: {len(df)}")
    print(f"Pares: {len(df) // 2}")


if __name__ == "__main__":
    main()
