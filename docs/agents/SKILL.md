---
name: doekit-experiment-designer
description: >-
  Runs Design-of-Experiments with doekit: brief → recommend → evaluate → lab
  matrix → ingest responses → analyze → next runs. Use for screening, RSM,
  optimal designs, design quality (D/A/G, FDS, power, VIF, aliases), sequential
  augmentation, mixed/blocked analysis, mixture or split-plot, or iterative
  experimental research (diseño de experimentos, criba, how many runs).
---

# doekit Experiment Designer

Elicit context, call doekit, interpret outputs, iterate with lab feedback.
Never invent efficiencies, rankings, or p-values — execute code and read
`to_dict()` / summaries. Facts from doekit; judgment from you.

| You | doekit |
|-----|--------|
| Goal, factors, budget, noise, constraints, trade-offs | Recommend, evaluate, fit, augment, report |

## Default path

```
DoE Progress:
- [ ] Brief (required fields)
- [ ] Experiment.from_goal → evaluate
- [ ] Approve matrix → lab (pause; do not invent y)
- [ ] ingest(y) → fit / mixed
- [ ] Gate: stop | augment | redesign
- [ ] If augment: next → new matrix → lab again
```

```python
import doekit as ed

exp = ed.experiment(goal="screening", factors=6, budget=12)
# factors={"temp": (20, 80), "ph": (3, 9)} also OK
exp.evaluate()
print(exp.plan)                  # run sheet template
# show rationale, caveats, table, matrix — ask approval
exp.export_csv("runs.csv")       # lab-ready artifact

exp.ingest(y)                    # real lab data only (dict/DataFrame for multi-y)
fit = exp.fit                    # from ingest(..., fit=True)
nxt = exp.next(n_add=4)          # after ingest
print(nxt.comparison.summary)

snap = exp.to_dict()             # persist / handoff (doekit.Experiment/1)
```

Escape hatches (see [reference.md](reference.md)): user already chose a generator
→ build + `evaluate`; batches → `fit_mixed_model` / `blocks=`; mixture /
split-plot / `Constraints` as in reference.

## Brief

**Required before recommend:** `goal`, factors (count or bounds), `budget`.

| Optional | API |
|----------|-----|
| linear / interactions / quadratic | `model_order=` |
| runs vs precision vs prediction | `priorities=` |
| irregular region | `constraints=Constraints(irregular=True)` (prefer over `constrained=True`) |
| effect size / noise | `effect_size=`, `sigma=` |
| mixture | `mixture=True` or `MixtureFactor` / `simplex_*` |
| hard-to-change | `hard_to_change=` / `split_plot_design` |

## After evaluate — reply template

```
Method / runs / model: …
Why: <rationale>
Alternatives: <1–2 from table>
Metrics (from evaluate/to_dict): D/A, G/FDS, VIF, power, aliases as relevant
Caveats: <all>
Matrix: show .matrix
→ Ask: approve this plan for the lab?
```

Pause until the user provides responses. Do not invent `y`.

## After ingest — gates

| Signal | Action |
|--------|--------|
| Goal met / no budget / user stops | Optional `exp.report(...)`; stop |
| Weak precision/power, budget left, same region | `exp.next(n_add=…)` / `propose_next_runs`; show Δ; new matrix |
| Wrong factors/region, mixture/model mismatch, strong LOF vs assumption | Redesign (`from_goal` / new design) — do not silently augment |
| Aliases / resolution limits | Do not overclaim effects; say what is confounded |

Argue “N more runs?” only with `compare_designs` / `nxt.comparison` deltas.

## Resume

If `snap = exp.to_dict()` exists, rebuild — do not restart the brief from scratch:

```python
design = ed.Design.from_dict(snap["design"])
model = ed.Model.from_dict(snap["model"]) if snap.get("model") else None
exp = ed.Experiment.from_design(design, model=model)
if snap.get("response") is not None:
    exp.ingest(snap["response"])
```

## Rules

- Prefer `Experiment` aggregate; call primitives only when needed.
- Always `evaluate` before declaring a plan fit-for-purpose.
- Prefer `Constraints(...)` over deprecated `constrained=True`.
- API detail: [reference.md](reference.md).
