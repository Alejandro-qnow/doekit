"""Deterministic automatic-conclusions for humans, agents, and LLMs."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..report_impl import (
    _DEFAULT_THRESHOLDS,
    _RSM_TEMPLATES,
    _STANDARD_TEMPLATES,
    _executive_summary,
    _recommendations,
    _sig_terms,
    report_summary,
)
from ...shared.serialize import jsonify

DEFAULT_THRESHOLDS = dict(_DEFAULT_THRESHOLDS)


def _quality_level(eff: Mapping[str, Any], kind: str, thr: Mapping[str, float]) -> str:
    if eff.get("rank_deficient"):
        return "rank_deficient"
    d = eff.get("D_efficiency")
    if d is None or not np.isfinite(d):
        return "unknown"
    if d >= thr["d_excellent"]:
        return "excellent"
    if d >= thr["d_ok"]:
        return "acceptable"
    if kind in _RSM_TEMPLATES:
        return "rsm_typical"
    return "low"


def _vif_gate(vif: Optional[Mapping[str, Any]], thr: Mapping[str, float]) -> dict:
    if not vif:
        return {"status": "unknown", "max_vif": None, "threshold": thr["vif_warn"]}
    vals = []
    for v in vif.values():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            vals.append(fv)
    if not vals:
        return {"status": "unknown", "max_vif": None, "threshold": thr["vif_warn"]}
    mx = max(vals)
    return {
        "status": "warn" if mx > thr["vif_warn"] else "ok",
        "max_vif": mx,
        "threshold": thr["vif_warn"],
    }


def _power_gate(power: Optional[Mapping[str, Any]], thr: Mapping[str, float]) -> dict:
    target = thr["power_target"]
    if not power:
        return {"status": "unknown", "target": target, "below_target": []}
    below = []
    finite = []
    for name, pw in power.items():
        if pw is None:
            continue
        try:
            fv = float(pw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        finite.append(fv)
        if fv < target:
            below.append({"term": str(name), "power": fv})
    if not finite:
        return {"status": "unknown", "target": target, "below_target": []}
    status = "ok" if not below else "below_target"
    return {
        "status": status,
        "target": target,
        "mean_power": float(np.mean(finite)),
        "below_target": below,
    }


def _inference_gate(fit) -> dict:
    if fit is None:
        return {"status": "no_response", "significant_terms": []}
    if fit.dof <= 0:
        return {"status": "saturated_no_test", "significant_terms": []}
    sig = _sig_terms(fit)
    if sig:
        return {"status": "has_sig", "significant_terms": sig}
    return {"status": "no_sig_at_0.05", "significant_terms": []}


def _process_gate(
    quality: str,
    inference: Mapping[str, Any],
    comparison: Optional[Mapping[str, Any]],
    unknowns: Sequence[str],
) -> dict:
    """Suggest stop | augment | redesign with required evidence fields."""
    evidence: dict[str, Any] = {}
    if "rank_deficient" == quality:
        return {
            "status": "redesign",
            "evidence": {"quality": quality, "reason": "rank_deficient"},
        }
    if comparison is not None and "worth_it" in comparison:
        evidence["worth_it"] = comparison.get("worth_it")
        evidence["comparison_summary"] = comparison.get("summary")
        if comparison.get("worth_it") is True:
            return {"status": "augment", "evidence": evidence}
    if inference.get("status") == "no_response":
        return {
            "status": "stop",
            "evidence": {**evidence, "reason": "awaiting_response",
                         "unknowns": list(unknowns)},
        }
    if inference.get("status") == "saturated_no_test":
        return {
            "status": "augment",
            "evidence": {**evidence, "reason": "saturated_fit"},
        }
    # Default: stop; agent may still choose augment from worth_it / judgment
    return {
        "status": "stop",
        "evidence": {
            **evidence,
            "quality": quality,
            "inference": inference.get("status"),
            "note": "Default stop; augment only with comparison.worth_it or explicit budget.",
        },
    }


def _collect_rules(
    design,
    eff: Mapping[str, Any],
    fit,
    thr: Mapping[str, float],
    anomalies,
    comparison: Optional[Mapping[str, Any]],
    lang: str,
) -> list[dict]:
    """Tag rule-based recommendations with stable rule ids."""
    kind = design.metadata.get("kind", "")
    rules: list[dict] = []
    texts = _recommendations(design, eff, fit, thr, lang)
    text_i = 0

    def push(rule_id: str, message: Optional[str] = None, **extra):
        nonlocal text_i
        msg = message
        if msg is None and text_i < len(texts):
            msg = texts[text_i]
            text_i += 1
        rules.append({"id": rule_id, "message": msg, **extra})

    if eff.get("rank_deficient"):
        push("rank_deficient")
        return rules

    d = eff.get("D_efficiency")
    if d is not None and np.isfinite(d) and d < thr["d_ok"]:
        if kind in _RSM_TEMPLATES:
            push("d_rsm_typical", d=float(d))
        elif kind not in _STANDARD_TEMPLATES:
            push("d_below_ok_nonstd", d=float(d), ok=thr["d_ok"])

    if fit is not None:
        if fit.dof <= 0:
            push("saturated_fit")
        else:
            sig = _sig_terms(fit)
            if sig:
                push("sig_terms", terms=sig)
            else:
                push("no_sig_at_0.05")
            if kind in {"PlackettBurman", "FractionalFactorial", "DefinitiveScreening"} and sig:
                push("screening_next_rsm")
        if anomalies is not None and len(anomalies):
            push("anomalies_present", n=int(len(anomalies)))

    if comparison is not None and comparison.get("worth_it") is True:
        push(
            "worth_extra_runs",
            message=comparison.get("summary") or "Extra runs appear worthwhile.",
            worth_it=True,
        )
    elif comparison is not None and comparison.get("worth_it") is False:
        push(
            "extra_runs_not_worth_it",
            message=comparison.get("summary") or "Extra runs may not be worth it.",
            worth_it=False,
        )

    # Attach any leftover recommendation strings without inventing new ones
    while text_i < len(texts):
        push("narrative", texts[text_i])
        text_i += 1
    return rules


def build_conclusions(
    design,
    *,
    response=None,
    model=None,
    evaluation=None,
    fit=None,
    thresholds: Optional[Mapping[str, float]] = None,
    comparison: Optional[Mapping[str, Any]] = None,
    lang: str = "en",
    doekit_version: str = "",
    wave_id: str = "",
    project_name: str = "",
) -> dict:
    """Build ``doekit.AutomaticConclusions/1`` from computed facts only."""
    from ..._version import __version__ as _ver  # noqa: PLC0415

    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    model = model or design.model
    summary = report_summary(
        design, response=response, model=model, thresholds=thr, lang=lang,
    )
    # Prefer caller-supplied evaluation/fit when available (avoid drift)
    if evaluation is not None:
        ev_dict = evaluation.to_dict() if hasattr(evaluation, "to_dict") else dict(evaluation)
        quality = ev_dict.get("efficiencies", summary["quality"])
        power = ev_dict.get("power")
        vif = ev_dict.get("vif")
    else:
        from ...assessment.evaluation import evaluate as _evaluate  # noqa: PLC0415
        ev = _evaluate(design, model=model)
        ev_dict = ev.to_dict()
        quality = ev_dict["efficiencies"]
        power = ev_dict["power"]
        vif = ev_dict["vif"]

    fit_obj = fit
    if fit_obj is None and response is not None:
        from ...assessment.analysis import fit_linear_model  # noqa: PLC0415
        fit_obj = fit_linear_model(design, response, model=model)

    anomalies = None
    if fit_obj is not None:
        an = fit_obj.anomalies()
        anomalies = an.to_dict("records") if an is not None and len(an) else []

    kind = design.metadata.get("kind", "")
    q_level = _quality_level(quality, kind, thr)
    vif_g = _vif_gate(vif, thr)
    power_g = _power_gate(power, thr)
    inf_g = _inference_gate(fit_obj)

    unknowns: list[str] = []
    if response is None:
        unknowns.append("no_response")
    if fit_obj is not None and fit_obj.dof <= 0:
        unknowns.append("lof_not_applicable_saturated")
    if power_g["status"] == "unknown":
        unknowns.append("power_unavailable")

    process = _process_gate(q_level, inf_g, comparison, unknowns)
    rules = _collect_rules(
        design, quality, fit_obj, thr, anomalies, comparison, lang,
    )

    exec_bullets = _executive_summary(design, quality, fit_obj, thr, lang)

    payload = {
        "schema": "doekit.AutomaticConclusions/1",
        "provenance": {
            "doekit_version": doekit_version or _ver,
            "project": project_name,
            "wave_id": wave_id,
            "design_kind": kind,
            "n_runs": int(design.n_runs),
            "n_params": (len(model.column_names(design.matrix))
                         if model is not None else None),
            "lang": lang,
            "thresholds": dict(thr),
        },
        "facts": {
            "methodology": summary.get("methodology"),
            "quality": quality,
            "power": power,
            "vif": vif,
            "executive_summary": exec_bullets,
            "recommendations": summary.get("recommendations"),
            "anomalies": anomalies,
            "fit": fit_obj.to_dict() if fit_obj is not None else summary.get("fit"),
            "comparison": comparison,
            "advisor": summary.get("recommendation"),
        },
        "rules": rules,
        "gate_board": {
            "quality": q_level,
            "vif": vif_g,
            "power": power_g,
            "inference": inf_g,
            "process": process,
        },
        "unknowns": unknowns,
        "llm_contract": {
            "allow": "Paraphrase executive_summary, recommendations, and rules[].message only.",
            "forbid": (
                "Do not invent efficiencies, p-values, rankings, or significance claims "
                "absent from facts/rules/gate_board."
            ),
        },
    }
    return jsonify(payload)


def conclusions_to_markdown(conclusions: Mapping[str, Any]) -> str:
    """Render a short markdown view of automatic conclusions."""
    prov = conclusions.get("provenance") or {}
    gates = conclusions.get("gate_board") or {}
    facts = conclusions.get("facts") or {}
    lines = [
        "# Automatic conclusions",
        "",
        f"- Project: `{prov.get('project', '')}`",
        f"- Wave: `{prov.get('wave_id', '')}`",
        f"- Design: `{prov.get('design_kind', '')}` ({prov.get('n_runs', '?')} runs)",
        f"- doekit: `{prov.get('doekit_version', '')}`",
        "",
        "## Gate board",
        "",
        f"- quality: **{gates.get('quality')}**",
        f"- vif: `{gates.get('vif')}`",
        f"- power: `{gates.get('power')}`",
        f"- inference: `{gates.get('inference')}`",
        f"- process: **{(gates.get('process') or {}).get('status')}**",
        "",
        "## Executive summary",
        "",
    ]
    for b in facts.get("executive_summary") or []:
        lines.append(f"- {b}")
    lines += ["", "## Rules", ""]
    for r in conclusions.get("rules") or []:
        msg = r.get("message") or ""
        lines.append(f"- `{r.get('id')}`: {msg}")
    unknowns = conclusions.get("unknowns") or []
    if unknowns:
        lines += ["", "## Unknowns", ""]
        for u in unknowns:
            lines.append(f"- {u}")
    contract = conclusions.get("llm_contract") or {}
    if contract:
        lines += [
            "",
            "## LLM contract",
            "",
            f"- Allow: {contract.get('allow', '')}",
            f"- Forbid: {contract.get('forbid', '')}",
        ]
    lines.append("")
    return "\n".join(lines)
