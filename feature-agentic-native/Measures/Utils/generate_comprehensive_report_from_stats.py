#!/usr/bin/env python3
"""Genera un reporte comprensivo narrativo desde metricas consolidadas CSV.

Uso:
    python Utils/generate_comprehensive_report_from_stats.py \
        --input-csv Measures/metrics_template.csv \
        --output-md Measures/REPORTE_COMPREHENSIVO_DESDE_ESTADISTICAS.md
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None

UTC = timezone.utc

CORE_METRICS = [
    "impact_score",
    "total_time_sec",
    "d_efficiency",
    "mean_power",
    "predicted_gain",
    "uncertainty_index",
]

RISK_METRICS = [
    "invalid_assumptions_count",
    "budget_violations_count",
    "model_mismatch_flags",
    "decision_reversal_count",
]

HIGHER_IS_BETTER = {"impact_score", "d_efficiency", "mean_power", "predicted_gain"}
LOWER_IS_BETTER = {"total_time_sec", "uncertainty_index"}


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def clean_number(v: Any) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def summary_stats(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "p90": np.nan}

    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "p90": float(s.quantile(0.9)),
    }


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    xa = pd.to_numeric(a, errors="coerce").dropna()
    xb = pd.to_numeric(b, errors="coerce").dropna()
    if len(xa) < 2 or len(xb) < 2:
        return np.nan

    sa = float(xa.std(ddof=1))
    sb = float(xb.std(ddof=1))
    pooled = math.sqrt((((len(xa) - 1) * sa * sa) + ((len(xb) - 1) * sb * sb)) / (len(xa) + len(xb) - 2))
    if pooled == 0:
        return np.nan
    return float((float(xa.mean()) - float(xb.mean())) / pooled)


def mann_whitney_pvalue(a: pd.Series, b: pd.Series) -> float:
    if mannwhitneyu is None:
        return np.nan
    xa = pd.to_numeric(a, errors="coerce").dropna()
    xb = pd.to_numeric(b, errors="coerce").dropna()
    if len(xa) == 0 or len(xb) == 0:
        return np.nan
    try:
        return float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def describe_signal(metric: str, delta: float, p_value: float) -> str:
    if math.isnan(delta):
        return "sin evidencia por datos incompletos"

    good = (metric in HIGHER_IS_BETTER and delta > 0) or (metric in LOWER_IS_BETTER and delta < 0)

    if math.isnan(p_value):
        return "senal descriptiva sin contraste inferencial"
    if p_value < 0.05:
        return "evidencia estadistica favorable" if good else "evidencia estadistica desfavorable"
    if p_value < 0.10:
        return "senal marginal, requiere mas muestra"
    return "sin evidencia estadistica concluyente"


def format_num(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "NA"
    return f"{value:.{digits}f}"


def compute_overall(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for metric in CORE_METRICS:
        if metric not in df.columns:
            continue

        with_values = df[df["agent_condition"] == "Agent_With"][metric]
        without_values = df[df["agent_condition"] == "Agent_Without"][metric]

        s_with = summary_stats(with_values)
        s_without = summary_stats(without_values)

        delta = (
            float(s_with["mean"] - s_without["mean"])
            if not math.isnan(s_with["mean"]) and not math.isnan(s_without["mean"])
            else np.nan
        )
        delta_pct = (
            float((delta / s_without["mean"]) * 100.0)
            if not math.isnan(delta) and s_without["mean"] not in {0, np.nan}
            else np.nan
        )

        p_value = mann_whitney_pvalue(with_values, without_values)
        effect = cohen_d(with_values, without_values)

        out[metric] = {
            "with": s_with,
            "without": s_without,
            "delta": delta,
            "delta_pct": delta_pct,
            "p_value": p_value,
            "cohen_d": effect,
            "signal": describe_signal(metric, delta, p_value),
        }
    return out


def _metric_delta_for_subset(sub: pd.DataFrame, metric: str) -> Dict[str, float]:
    with_values = pd.to_numeric(sub[sub["agent_condition"] == "Agent_With"][metric], errors="coerce").dropna()
    without_values = pd.to_numeric(sub[sub["agent_condition"] == "Agent_Without"][metric], errors="coerce").dropna()

    mean_with = float(with_values.mean()) if len(with_values) else np.nan
    mean_without = float(without_values.mean()) if len(without_values) else np.nan
    delta = float(mean_with - mean_without) if not math.isnan(mean_with) and not math.isnan(mean_without) else np.nan
    delta_pct = float((delta / mean_without) * 100.0) if not math.isnan(delta) and mean_without != 0 else np.nan

    return {
        "with_mean": mean_with,
        "without_mean": mean_without,
        "delta": delta,
        "delta_pct": delta_pct,
    }


def compute_by_task(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    if "task_id" not in df.columns:
        return {}

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task_id, sub in df.groupby("task_id"):
        task_row = {
            metric: _metric_delta_for_subset(sub, metric)
            for metric in CORE_METRICS
            if metric in sub.columns
        }
        result[str(task_id)] = task_row
    return result


def compute_by_stratum(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if "difficulty_stratum" not in df.columns or "impact_score" not in df.columns:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for stratum, sub in df.groupby("difficulty_stratum"):
        with_values = pd.to_numeric(sub[sub["agent_condition"] == "Agent_With"]["impact_score"], errors="coerce").dropna()
        without_values = pd.to_numeric(
            sub[sub["agent_condition"] == "Agent_Without"]["impact_score"], errors="coerce"
        ).dropna()
        mw = float(with_values.mean()) if len(with_values) else np.nan
        mo = float(without_values.mean()) if len(without_values) else np.nan
        delta = float(mw - mo) if not math.isnan(mw) and not math.isnan(mo) else np.nan
        out[str(stratum)] = {
            "with_mean": mw,
            "without_mean": mo,
            "delta": delta,
            "delta_pct": float((delta / mo) * 100.0) if not math.isnan(delta) and mo != 0 else np.nan,
        }
    return out


def compute_risk(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for metric in RISK_METRICS:
        if metric not in df.columns:
            continue
        with_values = pd.to_numeric(df[df["agent_condition"] == "Agent_With"][metric], errors="coerce")
        without_values = pd.to_numeric(df[df["agent_condition"] == "Agent_Without"][metric], errors="coerce")
        mw = float(with_values.mean()) if len(with_values.dropna()) else np.nan
        mo = float(without_values.mean()) if len(without_values.dropna()) else np.nan
        out[metric] = {
            "with_mean": mw,
            "without_mean": mo,
            "delta": float(mw - mo) if not math.isnan(mw) and not math.isnan(mo) else np.nan,
        }
    return out


def build_directions(overall: Dict[str, Dict[str, Any]]) -> List[str]:
    directions: List[str] = []

    impact = overall.get("impact_score")
    timing = overall.get("total_time_sec")
    deff = overall.get("d_efficiency")
    gain = overall.get("predicted_gain")
    uncertainty = overall.get("uncertainty_index")

    if impact:
        good_impact = clean_number(impact["delta"]) > 0
        if good_impact:
            directions.append(
                "Escalar la experimentacion priorizando estabilidad del ImpactScore, manteniendo pares balanceados y control de trazabilidad real."
            )
        else:
            directions.append(
                "Pausar escalamiento y revisar definicion de tareas y calibracion de flujo antes de invertir en mas corridas."
            )

    if timing and clean_number(timing["delta"]) > 0:
        directions.append(
            "Optimizar latencia del flujo con DoEkit separando tiempo de computo y tiempo de decision util, para no confundir costo tecnico con valor analitico."
        )

    if deff and clean_number(deff["delta"]) < 0:
        directions.append(
            "Recalibrar la estrategia de diseno (factors, budget, model_order) por tarea, especialmente donde d_efficiency cae en escenarios de mayor dificultad."
        )

    if gain and uncertainty:
        good_gain = clean_number(gain["delta"]) > 0
        good_unc = clean_number(uncertainty["delta"]) < 0
        if good_gain and good_unc:
            directions.append(
                "Mantener DoEkit en etapas de decision secuencial, donde se observa mejor ganancia esperada y menor incertidumbre residual."
            )

    if not directions:
        directions.append(
            "No hay direccion robusta con la evidencia actual; la prioridad es aumentar muestra y reducir incertidumbre inferencial."
        )

    return directions


def _build_metric_sentence(overall: Dict[str, Dict[str, Any]], metric: str) -> str:
    if metric not in overall:
        return ""
    node = overall[metric]
    return (
        f"{metric}: delta {format_num(node['delta'], 4)} "
        f"({format_num(node['delta_pct'], 2)}%), lectura={node['signal']}"
    )


def _build_task_paragraph(by_task: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    if not by_task:
        return ""
    chunks: List[str] = []
    for task_id, metrics in by_task.items():
        impact = metrics.get("impact_score")
        if impact:
            chunks.append(
                f"{task_id} muestra delta de ImpactScore {format_num(impact['delta'], 2)} "
                f"({format_num(impact['delta_pct'], 2)}%)"
            )
    return ("Por tarea, " + "; ".join(chunks) + ".") if chunks else ""


def _build_stratum_paragraph(by_stratum: Dict[str, Dict[str, float]]) -> str:
    if not by_stratum:
        return ""
    chunks = [
        f"estrato {name}: delta ImpactScore {format_num(m['delta'], 2)} ({format_num(m['delta_pct'], 2)}%)"
        for name, m in by_stratum.items()
    ]
    return "Por dificultad, " + "; ".join(chunks) + "."


def _build_risk_paragraph(risk: Dict[str, Dict[str, float]]) -> str:
    if not risk:
        return ""
    pieces = [f"{metric} delta {format_num(values['delta'], 3)}" for metric, values in risk.items()]
    return "En riesgo operativo, " + "; ".join(pieces) + "."


def render_report(
    experiment_name: str,
    input_csv: Path,
    rows_count: int,
    overall: Dict[str, Dict[str, Any]],
    by_task: Dict[str, Dict[str, Dict[str, float]]],
    by_stratum: Dict[str, Dict[str, float]],
    risk: Dict[str, Dict[str, float]],
) -> str:
    impact = overall.get("impact_score", {})
    time_metric = overall.get("total_time_sec", {})

    sample_note = (
        "La muestra ya permite una lectura mas estable de tendencia."
        if rows_count >= 60
        else "La muestra todavia es limitada, por lo que la lectura debe tratarse como senal inicial."
    )

    impact_sentence = (
        f"En impacto global, Agent_With presenta media {format_num(impact.get('with', {}).get('mean', np.nan), 2)} "
        f"frente a {format_num(impact.get('without', {}).get('mean', np.nan), 2)} de Agent_Without, "
        f"con delta {format_num(impact.get('delta', np.nan), 2)} "
        f"({format_num(impact.get('delta_pct', np.nan), 2)}%), p={format_num(impact.get('p_value', np.nan), 3)} "
        f"y d={format_num(impact.get('cohen_d', np.nan), 2)}."
    )

    time_sentence = (
        f"En tiempo total, Agent_With marca media {format_num(time_metric.get('with', {}).get('mean', np.nan), 4)} "
        f"vs {format_num(time_metric.get('without', {}).get('mean', np.nan), 4)}, "
        f"con delta {format_num(time_metric.get('delta', np.nan), 4)} "
        f"({format_num(time_metric.get('delta_pct', np.nan), 2)}%)."
    )

    quality_snippets = [
        s
        for s in [_build_metric_sentence(overall, m) for m in ["mean_power", "predicted_gain", "uncertainty_index", "d_efficiency"]]
        if s
    ]

    task_paragraph = _build_task_paragraph(by_task)
    stratum_paragraph = _build_stratum_paragraph(by_stratum)
    risk_paragraph = _build_risk_paragraph(risk)

    directions = build_directions(overall)
    directions_text = "\n".join([f"{idx}. {item}" for idx, item in enumerate(directions, start=1)])

    content = f"""# Reporte comprensivo desde estadisticas

Fecha de generacion: {now_iso()}

Experimento analizado: {experiment_name}

Fuente de metricas: {input_csv}

Este reporte transforma el consolidado estadistico en una lectura ejecutiva y tecnica para tomar direccion. Se mantiene una postura realista: no solo se observa si hay mejora de score, sino tambien el costo operativo y la estabilidad por tarea y, cuando aplica, por estrato de dificultad.

Se analizaron {rows_count} corridas validas de las condiciones Agent_With y Agent_Without. {sample_note}

{impact_sentence} {time_sentence}

En calidad tecnica, {"; ".join(quality_snippets)}.

{task_paragraph}

{stratum_paragraph}

{risk_paragraph}

La lectura integrada sugiere que la decision de adopcion no debe basarse en una sola metrica. Si el impacto compuesto mejora pero el costo temporal crece, la decision correcta depende del contexto: donde la calidad de decision y la trazabilidad pesan mas que la latencia, el despliegue de DoEkit tiene sentido; donde la latencia domina, la prioridad pasa por optimizar pipeline y configuracion experimental antes de escalar.

## Direcciones sugeridas
{directions_text}

## Nota metodologica
Se reportan estadisticos descriptivos, contraste Mann-Whitney (cuando SciPy esta disponible) y tamano de efecto Cohen d para metricas continuas clave. Esta lectura no reemplaza un modelo mixto completo, pero sirve como tablero de decision recurrente corrida tras corrida.
"""
    return content


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else project_root / path


def default_output_for_input(input_csv: Path, measures_root: Path) -> Path:
    name = input_csv.name.lower()
    if "meta" in name:
        return measures_root / "REPORTE_COMPREHENSIVO_META_DESDE_ESTADISTICAS.md"
    return measures_root / "REPORTE_COMPREHENSIVO_DESDE_ESTADISTICAS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera reporte comprensivo narrativo desde metricas CSV")
    parser.add_argument(
        "--input-csv",
        default="Measures/metrics_template.csv",
        help="Ruta al CSV consolidado relativa a la raiz del proyecto o absoluta.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Ruta al markdown de salida. Si se omite, se genera en Measures/.",
    )
    parser.add_argument(
        "--experiment-name",
        default="benchmark_contra_condicion_control",
        help="Etiqueta textual del experimento para el encabezado del reporte.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    measures_root = Path(__file__).resolve().parents[1]
    project_root = measures_root.parent

    input_csv = resolve_path(project_root, args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    if "agent_condition" not in df.columns:
        raise ValueError("El CSV no contiene columna agent_condition")

    if "meta_run_id" in df.columns:
        df = df[~df["meta_run_id"].isna()]

    rows_count = int(len(df))

    overall = compute_overall(df)
    by_task = compute_by_task(df)
    by_stratum = compute_by_stratum(df)
    risk = compute_risk(df)

    out_path = (
        resolve_path(project_root, args.output_md)
        if args.output_md.strip()
        else default_output_for_input(input_csv, measures_root)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = render_report(
        experiment_name=args.experiment_name,
        input_csv=input_csv,
        rows_count=rows_count,
        overall=overall,
        by_task=by_task,
        by_stratum=by_stratum,
        risk=risk,
    )
    out_path.write_text(report, encoding="utf-8")

    print(f"Reporte generado: {out_path}")
    print(f"Filas analizadas: {rows_count}")


if __name__ == "__main__":
    main()
