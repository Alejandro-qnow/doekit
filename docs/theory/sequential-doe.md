# Sequential / adaptive DoE

Classical DoE is often taught as a **one-shot** plan. In practice you run a
wave, look at the data, and decide what to run next. From **doekit 0.5** that
loop is first-class — without abandoning D/A/G-efficiency, SPV or power as the
common language.

## Augment a design

Keep the runs you already have; add points that maximize information of the
**combined** design:

```python
import doekit as ed

base = ed.random_design(
    [ed.ContinuousFactor("x1", -1, 1), ed.ContinuousFactor("x2", -1, 1)],
    n=6, seed=0,
)
base.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")

aug = ed.augment_design(base, n_add=4, criterion="D")
# aug.metadata: n_original, n_added, criterion, kind="AugmentedDesign"
```

## Propose the next batch

```python
nxt = ed.propose_next_runs(base, response=y, n_add=4, budget=16)
print(nxt.rationale)
print(nxt.comparison.summary)   # "worth the extra runs?"
nxt.added.matrix                # run sheet for the lab
nxt.to_dict()                   # schema: doekit.NextRunsProposal/1
```

- Without `response`: information-based augmentation (D/I/…).
- With `response`: residual `sigma_hat`, active terms (p-value cutoff), power
  deltas use the empirical noise.

## Compare designs

```python
cmp = ed.compare_designs(current, augmented)
print(cmp.summary)
cmp.table   # Δ D/A/G, SPV_mean, mean_power, n_runs
```

## Bridge to Bayesian optimization

Doekit does **not** replace Optuna/Ax. It lets you turn a search space into a
candidate set and keep evaluating with the same metrics:

```python
cand = ed.candidates_from_bounds([("lr", 1e-4, 1e-1), ("wd", 1e-6, 1e-2)], n=200)
# or: ed.candidates_from_skopt_space(space)  # requires doekit[bo]
nxt = ed.propose_next_runs(design, n_add=4, candidates=cand)
```

Related shipping surface (doekit ≥ 0.6 / 0.7): [mixture & split-plot](mixture-and-split-plot.md),
and the aggregate `ed.experiment(...)` / `Experiment` loop in the
[agents cheat sheet](../agents/reference.md).
