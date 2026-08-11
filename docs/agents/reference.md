# doekit Experiment Designer — API reference

Self-contained cheat sheet for the skill. Import: `import doekit as ed`.

## Core loop

| Call | Purpose |
|------|---------|
| `ed.experiment(...)` / `Experiment.from_goal` / `.from_design` | Aggregate: plan → evaluate → ingest → next → report → export → `to_dict()` |
| `exp.plan` / `exp.export_csv` / `exp.export_excel` | Lab run sheet (CSV; Excel needs `doekit[export]`) |
| `exp.ingest({...})` / `exp.multi_response_summary` / `desirability_scores` | Multi-y MVP |
| `ed.recommend_design(goal, factors, budget=..., model_order=..., priorities=..., constraints=..., mixture=..., hard_to_change=..., effect_size=..., sigma=...)` | Advisor → `Recommendation` |
| `ed.evaluate(design, model=..., effect_size=..., sigma=...)` | Quality → `DesignEvaluation` |
| `ed.fit_linear_model(design, y, blocks=..., cov_type=...)` | OLS (+ blocks, HC SE) |
| `ed.fit_mixed_model(design, y, groups=...)` | MixedLM (batches / whole plots) |
| `ed.propose_next_runs(design, response=y, n_add=4, budget=..., criterion="D", candidates=...)` | Next batch + comparison (`priorities=` accepted but not yet used) |
| `ed.augment_design(design, n_add, ...)` | Conditional optimal augmentation |
| `ed.compare_designs(a, b, ...)` | Δ D/A/G, SPV, power, runs |
| `ed.report(design, response=y, lang="es"\|"en", self_contained=...)` | Rule-based HTML |
| `ed.report_summary(design, response=y, ...)` | Same narrative as structured dict (agents) |
| `ed.project(name)` / `ExperimentProject` / `Wave` | Traceable on-disk project → waves |
| `exp.save(project\|wave)` / `Experiment.load(path)` / `exp.conclude(wave)` | Persist + automatic conclusions |
| `ed.build_conclusions(...)` | `doekit.AutomaticConclusions/1` (gates + rules) |
| `ed.candidates_from_bounds(...)` | Candidate points from factor bounds |

## Recommendation

Fields: `method`, `design`, `model`, `rationale`, `table`, `caveats`, `scenario`.
Use `rec.to_dict()`.

- `goal`: `"screening"` \| `"optimization"`
- `model_order`: `"linear"` \| `"interactions"` \| `"quadratic"`
- Default `priorities`: `{runs, precision, prediction}` = `1.0`

## Catalog

Always pair custom builds with `evaluate`.

| Family | Functions |
|--------|-----------|
| Screening | `plackett_burman`, `fractional_factorial`, `fold`, `definitive_screening` |
| Factorial | `full_factorial` |
| RSM | `box_behnken`, `central_composite` |
| Space-filling | `random_design`, `latin_hypercube` |
| Optimal | `optimal_design`, `kl_exchange`, `fedorov_exchange` |
| Mixture | `simplex_lattice`, `simplex_centroid`, `MixtureFactor`, Scheffé models |
| Split-plot | `split_plot_design` + `fit_mixed_model(groups="whole_plot_id")` |

## Constraints / region

```python
ed.Constraints(mixture=..., hard_to_change=..., irregular=..., run_cost=...)
```

Prefer `Constraints(...)` over deprecated `constrained=True` (`irregular=True`).

## Metrics (decision use)

| Metric | Use |
|--------|-----|
| D / A efficiency | Coefficient precision / information |
| G efficiency, SPV, FDS | Prediction variance over the region |
| VIF | Collinearity / unstable effects |
| Power | Detectability given effect_size/sigma |
| Alias matrix / resolution | Confounding (fractionals) |
| Rank deficiency | Model not supported by design |

## Analysis helpers

`main_effects`, `anova_table`, `lack_of_fit`, `half_normal_data`, `attach_blocks`,
`vif`, `alias_matrix`, `fds_data`, `power_analysis`, `efficiencies`.

## Serialization

| Schema / object | Source |
|-----------------|--------|
| `doekit.Experiment/1` | `Experiment.to_dict()` / `Experiment.from_dict` |
| `Design` / `Model` | `Design.from_dict` / `Model.from_dict` (resume) |
| `FitResult` / `MixedFitResult` | `fit.to_dict()` |
| `DesignEvaluation` | `evaluate(...).to_dict()` / `DesignEvaluation.from_dict` |
| `Recommendation` | `recommend_design(...).to_dict()` |
| `doekit.ExperimentProject/1` | `PROJECT.json` |
| `doekit.WaveManifest/1` | `waves/wave_NNN/manifest.json` |
| `doekit.AutomaticConclusions/1` | `automatic-conclusions/conclusions.json` |

## Workspace CLI

```text
doekit project init --name "My Study" --root experiments
doekit project sync --path experiments/experiment_project_my-study --factors 6 --budget 12
doekit project conclude --path experiments/experiment_project_my-study/waves/wave_001
```
