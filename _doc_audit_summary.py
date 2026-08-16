"""Summarize docstring audit into priority tiers."""
from __future__ import annotations

import json
from pathlib import Path

data = json.loads(Path(r"d:\projects\doekit\_doc_audit.json").read_text(encoding="utf-8"))
by_file = {e["file"]: e for e in data}

# Top-level exports from doekit/__init__.py (symbols -> defining modules)
# Manually mapped from __init__ imports for priority T0.
T0_EXPORTS = {
    # domain
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
    "region_from_design": "domain/region/__init__.py",
    # generation
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
    # analysis
    "fit_linear_model": "assessment/analysis/ols.py",
    "fit_mixed_model": "assessment/analysis/mixed.py",
    "main_effects": "assessment/analysis/effects.py",
    "half_normal_data": "assessment/analysis/effects.py",
    "anova_table": "assessment/analysis/anova.py",
    "lack_of_fit": "assessment/analysis/lof.py",
    "attach_blocks": "assessment/analysis/helpers.py",
    "FitResult": "assessment/analysis/results.py",
    "MixedFitResult": "assessment/analysis/results.py",
    # evaluation
    "evaluate": "assessment/evaluation/metrics.py",
    "efficiencies": "assessment/evaluation/metrics.py",
    "power_analysis": "assessment/evaluation/metrics.py",
    "vif": "assessment/evaluation/metrics.py",
    "alias_matrix": "assessment/evaluation/metrics.py",
    "fds_data": "assessment/evaluation/metrics.py",
    "DesignEvaluation": "assessment/evaluation/metrics.py",
    # presentation
    "report": "presentation/report.py",
    "report_summary": "presentation/report.py",
    "interpret": "presentation/narrative/interpret.py",
    "Interpretation": "presentation/narrative/interpret.py",
    "run_sheet": "presentation/export.py",
    "export_csv": "presentation/export.py",
    "export_excel": "presentation/export.py",
    "ExperimentProject": "presentation/workspace/project.py",
    "Wave": "presentation/workspace/project.py",
    "open_project": "presentation/workspace/project.py",
    "project": "presentation/workspace/project.py",
    "DEFAULT_THRESHOLDS": "presentation/workspace/project.py",
    "build_conclusions": "presentation/workspace/conclusions.py",
    # orchestration
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
    # adapters / surrogate
    "candidates_from_bounds": "adapters/bo.py",
    "candidates_from_skopt_space": "adapters/bo.py",
    "Surrogate": "assessment/surrogate/base.py",
    "OLSSurrogate": "assessment/surrogate/ols.py",
    "GPSurrogate": "assessment/surrogate/gp.py",
    "fit_surrogate": "assessment/surrogate/base.py",
    "loo_calibration": "assessment/surrogate/base.py",
}

# NEW feature files of interest
NEW_FILES = [
    "adapters/mcp.py",
    "presentation/workspace/project.py",
    "presentation/workspace/conclusions.py",
    "presentation/workspace/paths.py",
    "cli.py",
    "shared/serialize.py",
    "orchestration/decide/monitoring.py",
]

# Priority order of packages for non-T0 files
PACKAGE_TIER = {
    "orchestration": 1,
    "generation": 2,
    "assessment": 3,
    "presentation": 4,
    "adapters": 5,
    "domain": 6,
    "shared": 7,
    "cli.py": 5,
}

out_lines = []


def item_summary(it):
    secs = ",".join(it["sections"]) if it["sections"] else "-"
    return f"{it['name']} ({it['style']}; {secs})"


def file_stats(entry):
    # count only top-level public funcs/classes (not methods) for inventory
    tops = [i for i in entry["items"] if i["kind"] in ("func", "class")]
    methods = [i for i in entry["items"] if i["kind"] == "method"]
    counts = {"full_numpy": 0, "brief": 0, "one_line": 0, "missing": 0}
    for i in tops:
        counts[i["style"]] += 1
    has_formulas = [i["name"] for i in tops if "Formulas" in i["sections"]]
    has_examples = [i["name"] for i in tops if "Examples" in i["sections"]]
    full = [i["name"] for i in tops if i["style"] == "full_numpy"]
    brief = [i["name"] for i in tops if i["style"] == "brief"]
    one = [i["name"] for i in tops if i["style"] == "one_line"]
    miss = [i["name"] for i in tops if i["style"] == "missing"]
    # methods needing work (missing or one_line on important methods)
    miss_methods = [i["name"] for i in methods if i["style"] == "missing"]
    google = [i["name"] for i in entry["items"] if i["google"]]
    return {
        "n_public": len(tops),
        "n_methods": len(methods),
        "counts": counts,
        "full": full,
        "brief": brief,
        "one": one,
        "miss": miss,
        "has_formulas": has_formulas,
        "has_examples": has_examples,
        "miss_methods": miss_methods,
        "google": google,
        "tops": tops,
        "methods": methods,
    }


# Build per-file report for files with public tops
public_files = []
for e in data:
    stats = file_stats(e)
    if stats["n_public"] == 0 and e["file"] not in NEW_FILES:
        continue
    public_files.append((e, stats))

# Print structured inventory
print("PUBLIC MODULES WITH USER-FACING EXPORTS")
print("=" * 80)
for e, stats in public_files:
    print(f"\nFILE: doekit/{e['file']}")
    print(f"  module_doc: {e['module_doc']}")
    print(
        f"  public defs/classes: {stats['n_public']} "
        f"(full_numpy={stats['counts']['full_numpy']}, "
        f"brief={stats['counts']['brief']}, "
        f"one_line={stats['counts']['one_line']}, "
        f"missing={stats['counts']['missing']}); "
        f"public methods scanned: {stats['n_methods']}"
    )
    if stats["has_formulas"]:
        print(f"  HAS Formulas: {', '.join(stats['has_formulas'])}")
    if stats["has_examples"]:
        print(f"  HAS Examples: {', '.join(stats['has_examples'])}")
    if stats["full"]:
        print(f"  full NumPy sections: {', '.join(stats['full'])}")
    if stats["brief"]:
        print(f"  brief (multi-line, no sections): {', '.join(stats['brief'])}")
    if stats["one"]:
        print(f"  one-line only: {', '.join(stats['one'])}")
    if stats["miss"]:
        print(f"  MISSING docs: {', '.join(stats['miss'])}")
    if stats["miss_methods"]:
        print(f"  methods MISSING docs: {', '.join(stats['miss_methods'][:20])}"
              + (" ..." if len(stats["miss_methods"]) > 20 else ""))
    if stats["google"]:
        print(f"  GOOGLE Args: {', '.join(stats['google'])}")

# T0 symbol styles
print("\n\nTIER 0 — top-level __all__ symbols docstring style")
print("=" * 80)
t0_by_style = {"full_numpy": [], "brief": [], "one_line": [], "missing": [], "not_found": []}
for sym, fpath in sorted(T0_EXPORTS.items(), key=lambda x: (x[1], x[0])):
    entry = by_file.get(fpath)
    if not entry:
        t0_by_style["not_found"].append((sym, fpath))
        continue
    found = None
    for it in entry["items"]:
        if it["kind"] in ("func", "class") and it["name"] == sym:
            found = it
            break
    if found is None:
        # maybe constant - skip DEFAULT_THRESHOLDS
        t0_by_style["not_found"].append((sym, fpath))
        continue
    secs = ",".join(found["sections"]) if found["sections"] else "-"
    t0_by_style[found["style"]].append((sym, fpath, secs))

for style, items in t0_by_style.items():
    print(f"\n{style} ({len(items)}):")
    for row in items:
        if style == "not_found":
            print(f"  {row[0]} @ {row[1]}")
        else:
            print(f"  {row[0]} @ doekit/{row[1]} [{row[2]}]")

# NEW files detail
print("\n\nNEW FEATURE FILES DETAIL")
print("=" * 80)
for f in NEW_FILES:
    e = by_file.get(f)
    if not e:
        print(f"MISSING FILE: {f}")
        continue
    stats = file_stats(e)
    print(f"\nkoekit/{f}: public={stats['n_public']} "
          f"full={stats['counts']['full_numpy']} brief={stats['counts']['brief']} "
          f"one={stats['counts']['one_line']} miss={stats['counts']['missing']}")
    for it in stats["tops"]:
        print(f"  {it['kind']} {item_summary(it)}")
    for it in stats["methods"][:30]:
        print(f"  method {item_summary(it)}")

# Aggregate counts
print("\n\nAGGREGATE (top-level public func/class only)")
print("=" * 80)
agg = {"full_numpy": 0, "brief": 0, "one_line": 0, "missing": 0}
files_with_public = 0
for e, stats in public_files:
    if stats["n_public"] == 0:
        continue
    files_with_public += 1
    for k, v in stats["counts"].items():
        agg[k] += v
print(f"files_with_public_defs: {files_with_public}")
print(agg)
print(f"total public tops: {sum(agg.values())}")
print(f"pct full_numpy: {100*agg['full_numpy']/sum(agg.values()):.1f}%")

# List files needing most work (missing + brief + one_line among T0-related and NEW)
print("\n\nWORK QUEUE BY PRIORITY")
print("=" * 80)

# Tier mapping for files
tier_files = {
    0: set(T0_EXPORTS.values()),
    "new": set(NEW_FILES),
}

# For each T0 file, list symbols needing upgrade
need = []
for sym, fpath in T0_EXPORTS.items():
    entry = by_file.get(fpath)
    if not entry:
        continue
    for it in entry["items"]:
        if it["kind"] in ("func", "class") and it["name"] == sym:
            if it["style"] != "full_numpy" or (
                # acquisition/metrics should ideally have Formulas for math-heavy
                False
            ):
                need.append((0, fpath, sym, it["style"], it["sections"]))

# Math-heavy that have full but missing Formulas
MATH_HEAVY = {
    "efficiencies", "power_analysis", "vif", "alias_matrix", "fds_data",
    "d_criterion", "a_criterion", "t_criterion", "g_criterion", "e_criterion",
    "i_criterion", "expected_improvement", "probability_of_improvement",
    "upper_confidence_bound", "expected_hypervolume_improvement", "hypervolume",
    "dominates", "pareto_front",
}

print("\nMath-heavy T0 symbols missing Formulas section:")
for sym, fpath in sorted(T0_EXPORTS.items()):
    if sym not in MATH_HEAVY:
        continue
    entry = by_file.get(fpath)
    if not entry:
        continue
    for it in entry["items"]:
        if it["name"] == sym and "Formulas" not in it["sections"]:
            print(f"  {sym} @ doekit/{fpath} style={it['style']} sections={it['sections']}")

Path(r"d:\projects\doekit\_doc_audit_summary.txt").write_text(
    "see stdout", encoding="utf-8"
)
