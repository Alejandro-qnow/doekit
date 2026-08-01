# Optimal design

## Motivation

Standard templates (factorial, Box-Behnken, CCD) assume a nice cubic/spherical region
and a fixed model. When the region is **irregular**, some combinations are
**infeasible**, or the run budget is an arbitrary fixed number, no template fits.
Optimal design selects the best subset of a **candidate set** under a criterion on the
information matrix.

## Theory

For a model matrix $X$, the **information matrix** is $M = X^\top X$; the covariance
of the least-squares estimates is $\sigma^2 M^{-1}$. Each optimality criterion
compresses $M$ into a scalar to maximize (doekit uses the "larger is better"
convention throughout):

| Criterion | Optimizes | Meaning |
|---|---|---|
| **D** | $\det(M)$ | maximal joint information (most used) |
| **A** | $\operatorname{tr}(M^{-1})$ | minimal average coefficient variance |
| **I** | mean prediction variance over the region | best for *prediction* |
| **G** | $\max_x \operatorname{Var}\hat y(x)$ | minimal worst-case prediction variance |
| **E** | $\min \lambda(M)$ | best worst-conditioned direction |
| **T** | $\operatorname{tr}(M)$ | maximal information "magnitude" |

Two exchange algorithms search the candidate set:

- **KL-exchange** — specialized and efficient for **D**-optimality (Atkinson, Donev &
  Tobias); the default for `criterion="D"`. It swaps low-variance design points for
  high-variance candidates using the rank-one determinant update
  $\Delta = (1 - d_k)(1 + d_l) + (x_k^\top M^{-1} x_l)^2$.
- **Fedorov** — a generic exchange that optimizes **any** criterion; the default
  otherwise.

`n_starts > 1` runs several random restarts and keeps the best, escaping local optima.

## In doekit

```python
import doekit as ed

cand = ed.random_design([ed.ContinuousFactor("x1", -1, 1),
                         ed.ContinuousFactor("x2", -1, 1)], n=300, seed=0)
cand.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")

opt = ed.optimal_design(cand, n_runs=12, criterion="D",
                        algorithm="kl", n_starts=5, seed=1)
opt.metadata["criteria"]          # all criteria of the final design
```

## See also

- Theory: [Evaluation metrics](evaluation-metrics.md)
- API: [`optimal_design`, `kl_exchange`, `fedorov_exchange`, criteria](../api/designs.md)
