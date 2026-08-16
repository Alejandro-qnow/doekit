"""Write compact docstring inventory (UTF-8 file only)."""
from __future__ import annotations

import json
from pathlib import Path

data = json.loads(Path(r"d:\projects\doekit\_doc_audit.json").read_text(encoding="utf-8"))
by_file = {e["file"]: e for e in data}


def tops(e):
    return [i for i in e["items"] if i["kind"] in ("func", "class")]


lines = []
agg = {"full_numpy": 0, "brief": 0, "one_line": 0, "missing": 0}
n_files = 0
for e in data:
    t = tops(e)
    if not t:
        continue
    n_files += 1
    c = {"full_numpy": 0, "brief": 0, "one_line": 0, "missing": 0}
    for i in t:
        c[i["style"]] += 1
        agg[i["style"]] += 1
    formulas = [i["name"] for i in t if "Formulas" in i["sections"]]
    examples = [i["name"] for i in t if "Examples" in i["sections"]]
    # incomplete full: has Parameters but no Returns (and is a function)
    incomplete = []
    for i in t:
        if i["style"] != "full_numpy":
            continue
        if i["kind"] == "func" and "Parameters" in i["sections"] and "Returns" not in i["sections"]:
            incomplete.append(i["name"])
        if i["kind"] == "class" and not any(
            s in i["sections"] for s in ("Parameters", "Attributes", "Notes", "Examples")
        ):
            incomplete.append(i["name"])
    lines.append(
        f"{e['file']}\tn={len(t)}\tfull={c['full_numpy']}\tbrief={c['brief']}\tone={c['one_line']}\tmiss={c['missing']}"
        f"\tFormulas={formulas}\tExamples={examples}\tincomplete={incomplete}"
    )
    for i in t:
        lines.append(
            f"  {i['kind']}\t{i['name']}\t{i['style']}\t{i['sections']}\t{i['first'][:90]}"
        )

lines.append("")
lines.append(f"FILES={n_files} AGG={agg} TOTAL={sum(agg.values())}")

# T0 map corrected
T0 = {
    "Factor": "domain/factors/protocols.py",
    "ContinuousFactor": "domain/factors/continuous.py",
    "DiscreteFactor": "domain/factors/discrete.py",
    "CategoricalFactor": "domain/factors/categorical.py",
    "MixtureFactor": "domain/factors/mixture.py",
    "as_factors": "domain/factors/registry.py",
    "factor_from_dict": "domain/factors/registry.py",
    "Model": "domain/model/spec.py",
    "Intercept": "domain/model/terms.py",
    "Main": "domain/model/terms.py",
    "Interaction": "domain/model/terms.py",
    "Power": "domain/model/terms.py",
    "d_criterion": "domain/criteria/functions.py",
    "a_criterion": "domain/criteria/functions.py",
    "t_criterion": "domain/criteria/functions.py",
    "g_criterion": "domain/criteria/functions.py",
    "e_criterion": "domain/criteria/functions.py",
    "i_criterion": "domain/criteria/functions.py",
    "Design": "domain/design/entity.py",
    "Constraints": "domain/constraints.py",
    "coerce_constraints": "domain/constraints.py",
    "Region": "domain/region/base.py",
    "HypercubeRegion": "domain/region/hypercube.py",
    "SimplexRegion": "domain/region/simplex.py",
    "region_from_design": "domain/region/hypercube.py",
    "full_factorial": "generation/catalog/factorial.py",
    "fractional_factorial": "generation/catalog/factorial.py",
    "plackett_burman": "generation/catalog/screening.py",
    "is_plackett_burman": "generation/catalog/screening.py",
    "fold": "generation/catalog/screening.py",
    "box_behnken": "generation/catalog/response_surface.py",
    "central_composite": "generation/catalog/response_surface.py",
    "definitive_screening": "generation/catalog/definitive.py",
    "random_design": "generation/catalog/random_design.py",
    "latin_hypercube": "generation/catalog/random_design.py",
    "optimal_design": "generation/search/optimal.py",
    "kl_exchange": "generation/search/optimal.py",
    "fedorov_exchange": "generation/search/optimal.py",
    "simplex_lattice": "generation/catalog/mixture.py",
    "simplex_centroid": "generation/catalog/mixture.py",
    "split_plot_design": "generation/catalog/split_plot.py",
    "fit_linear_model": "assessment/analysis/ols.py",
    "fit_mixed_model": "assessment/analysis/mixed.py",
    "main_effects": "assessment/analysis/effects.py",
    "half_normal_data": "assessment/analysis/effects.py",
    "anova_table": "assessment/analysis/anova.py",
    "lack_of_fit": "assessment/analysis/lof.py",
    "attach_blocks": "assessment/analysis/helpers.py",
    "FitResult": "assessment/analysis/results.py",
    "MixedFitResult": "assessment/analysis/results.py",
    "evaluate": "assessment/evaluation/metrics.py",
    "efficiencies": "assessment/evaluation/metrics.py",
    "power_analysis": "assessment/evaluation/metrics.py",
    "vif": "assessment/evaluation/metrics.py",
    "alias_matrix": "assessment/evaluation/metrics.py",
    "fds_data": "assessment/evaluation/metrics.py",
    "DesignEvaluation": "assessment/evaluation/metrics.py",
    "report": "presentation/report_impl.py",  # alias of report_html
    "report_summary": "presentation/report_impl.py",
    "interpret": "presentation/narrative/interpret.py",
    "Interpretation": "presentation/narrative/interpret.py",
    "run_sheet": "presentation/export.py",
    "export_csv": "presentation/export.py",
    "export_excel": "presentation/export.py",
    "ExperimentProject": "presentation/workspace/project.py",
    "Wave": "presentation/workspace/project.py",
    "open_project": "presentation/workspace/project.py",
    "project": "presentation/workspace/project.py",
    "build_conclusions": "presentation/workspace/conclusions.py",
    "recommend_design": "orchestration/advise/recommend.py",
    "Recommendation": "orchestration/advise/recommend.py",
    "ExperimentHistory": "orchestration/advise/history.py",
    "ExperimentRecord": "orchestration/advise/history.py",
    "learn_priors": "orchestration/advise/history.py",
    "historical_recommendation": "orchestration/advise/history.py",
    "augment_design": "orchestration/sequential/propose.py",
    "propose_next_runs": "orchestration/sequential/propose.py",
    "compare_designs": "orchestration/sequential/propose.py",
    "NextRunsProposal": "orchestration/sequential/propose.py",
    "DesignComparison": "orchestration/sequential/propose.py",
    "expected_improvement": "orchestration/optimize/acquisition.py",
    "probability_of_improvement": "orchestration/optimize/acquisition.py",
    "upper_confidence_bound": "orchestration/optimize/acquisition.py",
    "expected_hypervolume_improvement": "orchestration/optimize/acquisition.py",
    "get_acquisition": "orchestration/optimize/acquisition.py",
    "pareto_front": "orchestration/optimize/pareto.py",
    "pareto_mask": "orchestration/optimize/pareto.py",
    "dominates": "orchestration/optimize/pareto.py",
    "hypervolume": "orchestration/optimize/pareto.py",
    "Decision": "orchestration/decide/engine.py",
    "DecisionContext": "orchestration/decide/engine.py",
    "DecisionScore": "orchestration/decide/engine.py",
    "decide_next_action": "orchestration/decide/engine.py",
    "context_from_proposal": "orchestration/decide/engine.py",
    "ContinuationScorer": "orchestration/decide/engine.py",
    "ThresholdPolicy": "orchestration/decide/engine.py",
    "RiskAdaptivePolicy": "orchestration/decide/engine.py",
    "BudgetAwarePolicy": "orchestration/decide/engine.py",
    "ConvergenceResult": "orchestration/decide/monitoring.py",
    "check_convergence": "orchestration/decide/monitoring.py",
    "DiagnosticsReport": "orchestration/decide/monitoring.py",
    "diagnose_step": "orchestration/decide/monitoring.py",
    "Experiment": "orchestration/experiment/aggregate.py",
    "experiment": "orchestration/experiment/aggregate.py",
    "desirability_scores": "orchestration/experiment/aggregate.py",
    "candidates_from_bounds": "adapters/bo.py",
    "candidates_from_skopt_space": "adapters/bo.py",
    "Surrogate": "assessment/surrogate/base.py",
    "OLSSurrogate": "assessment/surrogate/ols.py",
    "GPSurrogate": "assessment/surrogate/gp.py",
    "fit_surrogate": "assessment/surrogate/base.py",
    "loo_calibration": "assessment/surrogate/base.py",
}

# Fix report names
REPORT_ALIASES = {"report": "report_html"}

lines.append("\n=== T0 SYMBOL STYLES ===")
by_style = {"full_numpy": [], "brief": [], "one_line": [], "missing": [], "not_found": []}
for sym, fpath in sorted(T0.items(), key=lambda x: x[1]):
    look = REPORT_ALIASES.get(sym, sym)
    entry = by_file.get(fpath)
    if not entry:
        by_style["not_found"].append(f"{sym}@{fpath}")
        continue
    found = next(
        (i for i in entry["items"] if i["kind"] in ("func", "class") and i["name"] == look),
        None,
    )
    if not found:
        by_style["not_found"].append(f"{sym}@{fpath}")
        continue
    by_style[found["style"]].append(
        f"{sym} @ {fpath} sections={found['sections']}"
    )

for style, items in by_style.items():
    lines.append(f"\n{style} ({len(items)})")
    for x in items:
        lines.append(f"  {x}")

# Tier work lists: only non-full for T0, plus NEW files all public
NEW = [
    "adapters/mcp.py",
    "presentation/workspace/project.py",
    "presentation/workspace/conclusions.py",
    "presentation/workspace/paths.py",
    "cli.py",
    "shared/serialize.py",
    "orchestration/decide/monitoring.py",
]

MATH = {
    "d_criterion", "a_criterion", "t_criterion", "g_criterion", "e_criterion", "i_criterion",
    "expected_improvement", "probability_of_improvement", "upper_confidence_bound",
    "expected_hypervolume_improvement", "hypervolume", "dominates", "pareto_front",
}

lines.append("\n=== MATH-HEAVY T0 WITHOUT Formulas ===")
for sym, fpath in sorted(T0.items()):
    if sym not in MATH:
        continue
    look = REPORT_ALIASES.get(sym, sym)
    entry = by_file[fpath]
    found = next(i for i in entry["items"] if i["name"] == look and i["kind"] in ("func", "class"))
    if "Formulas" not in found["sections"]:
        lines.append(f"  {sym} @ {fpath} style={found['style']} sections={found['sections']}")

# Group T0 needing upgrade by package priority
PRIORITY_ORDER = [
    ("orchestration", "orchestration/"),
    ("generation", "generation/"),
    ("assessment", "assessment/"),
    ("presentation", "presentation/"),
    ("adapters", "adapters/"),
    ("domain", "domain/"),
]

lines.append("\n=== T0 NEEDING UPGRADE (not full_numpy) BY PACKAGE ===")
for label, prefix in PRIORITY_ORDER:
    lines.append(f"\n## {label}")
    for sym, fpath in sorted(T0.items(), key=lambda x: x[0]):
        if not fpath.startswith(prefix):
            continue
        look = REPORT_ALIASES.get(sym, sym)
        entry = by_file[fpath]
        found = next(
            (i for i in entry["items"] if i["name"] == look and i["kind"] in ("func", "class")),
            None,
        )
        if found and found["style"] != "full_numpy":
            lines.append(f"  {sym} ({found['style']}) @ {fpath}")

# Also full but incomplete (Params only) among T0
lines.append("\n=== T0 full_numpy BUT incomplete (func Params without Returns) ===")
for sym, fpath in sorted(T0.items()):
    look = REPORT_ALIASES.get(sym, sym)
    entry = by_file[fpath]
    found = next(
        (i for i in entry["items"] if i["name"] == look and i["kind"] in ("func", "class")),
        None,
    )
    if not found or found["style"] != "full_numpy":
        continue
    if found["kind"] == "func" and "Parameters" in found["sections"] and "Returns" not in found["sections"]:
        lines.append(f"  {sym} @ {fpath} sections={found['sections']}")
    # missing Examples for user-facing API - note separately
    if "Examples" not in found["sections"] and found["kind"] == "func":
        pass  # too many

# Files with Examples anywhere
lines.append("\n=== FILES WITH Examples SECTION ===")
for e in data:
    ex = [i["name"] for i in tops(e) if "Examples" in i["sections"]]
    if ex:
        lines.append(f"  {e['file']}: {ex}")

lines.append("\n=== FILES WITH Formulas SECTION ===")
for e in data:
    fr = [i["name"] for i in tops(e) if "Formulas" in i["sections"]]
    if fr:
        lines.append(f"  {e['file']}: {fr}")

# Google Args check
google_hits = []
for e in data:
    for i in e["items"]:
        if i["google"]:
            google_hits.append(f"{e['file']}:{i['name']}")
lines.append(f"\n=== GOOGLE Args HITS: {len(google_hits)} ===")
for h in google_hits:
    lines.append(f"  {h}")

Path(r"d:\projects\doekit\_doc_inventory.txt").write_text("\n".join(lines), encoding="utf-8")
print("wrote", len(lines), "lines")
