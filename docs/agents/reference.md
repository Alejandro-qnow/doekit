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
| `ed.propose_next_runs(design, response=y, n_add=4, budget=..., criterion="D", candidates=...)` | Next batch + comparison — `intent="learn"` (default) |
| `ed.propose_next_runs(..., intent="optimize", acquisition="ei", goals=..., surrogate="auto")` | Surrogate + acquisition batch (see Optimize section) |
| `ed.augment_design(design, n_add, ...)` | Conditional optimal augmentation |
| `ed.compare_designs(a, b, ...)` | Δ D/A/G, SPV, power, runs |
| `ed.report(design, response=y, lang="es"\|"en", self_contained=...)` | Rule-based HTML |
| `ed.report_summary(design, response=y, ...)` | Same narrative as structured dict (agents) |
| `ed.project(name)` / `ExperimentProject` / `Wave` | Traceable on-disk project → waves |
| `exp.save(project\|wave)` / `Experiment.load(path)` / `exp.conclude(wave)` | Persist + automatic conclusions |
| `ed.build_conclusions(...)` | `doekit.AutomaticConclusions/1` (gates + rules) |
| `ed.candidates_from_bounds(...)` | Candidate points from factor bounds |

## Optimize: surrogate + acquisition

Response optimization (move the result), the complement of learn (sharpen the model).

| Call | Purpose |
|------|---------|
| `ed.fit_surrogate(design, y, kind="auto"\|"ols"\|"gp", model=..., factors=...)` | Fit a `Surrogate` (GP prior-mean = OLS, else OLS) |
| `sur.predict(X)` → `(mean, std)` | Predictive mean + **calibrated** sigma (grows away from data) |
| `sur.calibration(levels=(.5,.8,.95))` | LOO interval coverage vs nominal — audit the box |
| `ed.propose_next_runs(design, response=y, n_add=4, intent="optimize", acquisition=..., goals=..., surrogate="auto", kappa=2.0, xi=0.01)` | Surrogate batch (Kriging-Believer, respects `Region`/simplex) |
| `ed.expected_improvement` / `upper_confidence_bound` / `probability_of_improvement` | Single-objective acquisitions on `(mean, std)` |
| `ed.expected_hypervolume_improvement`, `ed.pareto_front`, `ed.pareto_mask`, `ed.dominates`, `ed.hypervolume` | Multi-objective / Pareto |
| `ed.get_acquisition("ei"\|"ucb"\|"pi"\|"ehvi")` | Acquisition lookup |

- `acquisition`: `"ei"` (default single), `"ucb"`, `"pi"`, `"ehvi"` (default multi).
- `goals`: `{col: "max"\|"min"}`; single-objective also honors `goal="max"\|"min"`.
- `surrogate`: `"auto"` → GP if `doekit[bo]` (scikit-learn) installed, else `"ols"`.
- GP backend needs `doekit[bo]`; `OLSSurrogate` is the dependency-free fallback.

**`NextRunsProposal` (optimize)** adds: `intent`, `acquisition`, `best_so_far`,
`predicted_improvement`, `pareto_front`, `explore_exploit` (`mode`:
exploring/exploiting/balanced), `surrogate`, `acquisition_values`. All in `to_dict()`
(`surrogate` → kind + calibration summary).

## Agentic layer: interpret · decide · monitor · memory · MCP

Structured signals for agents; every fact is composed from doekit results (never invented).

| Call | Purpose |
|------|---------|
| `ed.interpret(result)` → `Interpretation` | Uniform reading of Recommendation/Evaluation/Fit/Proposal/Comparison; `.for_llm()`, `.to_dict()` (`doekit.Interpretation/1`) |
| `exp.decide_next(n_add=…, intent=…, budget=…, history=…)` → `Decision` | Action `stop`/`augment`/`refine`/`redesign` (`doekit.Decision/1`) |
| `ed.decide_next_action(ctx, ...)` / `ed.context_from_proposal(proposal, budget_total=…, budget_spent=…)` | Engine + context factory |
| `ed.ContinuationScorer`, `ed.ThresholdPolicy`/`RiskAdaptivePolicy`/`BudgetAwarePolicy` | Pluggable scoring / policies |
| `ed.check_convergence(history, metric_key="best_so_far")` → `ConvergenceResult` | Marginal-gain stop signal (feeds the engine) |
| `ed.diagnose_step(metrics, budget_remaining=…, uncertainty=…)` → `DiagnosticsReport` | Per-wave warnings (power, G-eff drop, budget, uncertainty) |
| `ed.ExperimentHistory.from_project(proj)` / `ed.learn_priors(...)` / `ed.historical_recommendation(...)` | Meta-learning from past waves (store = workspace) |
| `python -m doekit.adapters.mcp` (extra `[mcp]`) | Serve recommend / evaluate / propose+decide as MCP tools |

- Decision is **intent-aware**: learn scores efficiency/power deltas; optimize
  scores `predicted_improvement` + explore/exploit and does *not* penalize a
  D-efficiency drop.
- Hard gates first: rank-deficient → `redesign`; budget exhausted / convergence → `stop`.
- `gate_board.process.status` (in `AutomaticConclusions`) is produced by this same
  engine — one decision logic, not two.

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

## Plots (`ed.plotting`, needs matplotlib)

Design/analysis: `half_normal_plot`, `effects_plot`, `correlation_plot`,
`fds_plot` (accepts `surrogate=` for sigma(x)), `power_plot`, `alias_heatmap`.
Optimize: `surrogate_surface`, `acquisition_plot`, `convergence_plot`,
`parity_plot`, `calibration_plot` (the moat viz), `pareto_plot`, `slice_plot`.

## Serialization

| Schema / object | Source |
|-----------------|--------|
| `doekit.Experiment/1` | `Experiment.to_dict()` / `Experiment.from_dict` |
| `Design` / `Model` | `Design.from_dict` / `Model.from_dict` (resume) |
| `FitResult` / `MixedFitResult` | `fit.to_dict()` |
| `DesignEvaluation` | `evaluate(...).to_dict()` / `DesignEvaluation.from_dict` |
| `Recommendation` | `recommend_design(...).to_dict()` |
| `doekit.NextRunsProposal/1` | `propose_next_runs(...).to_dict()` (learn + optimize fields) |
| `doekit.DesignComparison/1` | `compare_designs(...).to_dict()` |
| `doekit.Interpretation/1` | `interpret(result).to_dict()` |
| `doekit.Decision/1` | `decide_next_action(...).to_dict()` / `exp.decide_next(...)` |
| `doekit.ConvergenceResult/1` · `doekit.DiagnosticsReport/1` | `check_convergence(...)` · `diagnose_step(...)` |
| `doekit.ExperimentRecord/1` · `doekit.PriorEstimate/1` | history / `learn_priors(...)` |
| `doekit.ExperimentProject/1` | `PROJECT.json` |
| `doekit.WaveManifest/1` | `waves/wave_NNN/manifest.json` |
| `doekit.AutomaticConclusions/1` | `automatic-conclusions/conclusions.json` |

## Workspace CLI

```text
doekit project init --name "My Study" --root experiments
doekit project sync --path experiments/experiment_project_my-study --factors 6 --budget 12
doekit project conclude --path experiments/experiment_project_my-study/waves/wave_001
```
