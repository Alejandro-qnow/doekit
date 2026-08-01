# Blocked and mixed analysis

From **doekit 0.4**, analysis is no longer limited to plain OLS. Blocking,
heteroscedasticity-robust standard errors, lack-of-fit tests and linear mixed
models (REML) are first-class, with `statsmodels` as a **core** dependency.

## Fixed blocks

When a known nuisance factor (day, batch, operator) is controlled by the design,
treat it as a **fixed block**:

```python
import doekit as ed
import numpy as np

d = ed.full_factorial({"A": [-1, 1], "B": [-1, 1]})
# replicate the factorial across two blocks
import pandas as pd
mat = pd.concat([d.matrix, d.matrix], ignore_index=True)
design = ed.Design(matrix=mat, factors=d.factors,
                   model=ed.Model.main_effects(["A", "B"]))
design = ed.attach_blocks(design, blocks=[0, 0, 0, 0, 1, 1, 1, 1])

y = ...  # measured responses
fit = ed.fit_linear_model(design, y, blocks="block")   # or rely on metadata
ed.anova_table(fit)
```

`attach_blocks` writes the column and `metadata["blocking"]`. If that metadata
is present, `fit_linear_model(..., blocks=None)` uses it automatically. Block
columns are **excluded** from the default factor model and enter as drop-first
dummies (`block[...]`).

## Robust standard errors

```python
fit = ed.fit_linear_model(design, y, cov_type="HC3")  # or HC0 / HC1
```

Use HC when residual variance is suspected to change with the factor settings.
Point estimates stay OLS; only the covariance (hence SE / p-values) changes.

## Lack of fit

When the design has **replicate** rows (identical factor levels), pure error can
be separated from lack of fit:

```python
lof = ed.lack_of_fit(design, y)
# sources: lack_of_fit, pure_error, residual
```

Raises if there are no replicates.

## Mixed models (random groups)

Hard-to-change factors / batches that were **not** designed as fixed blocks are
better modelled as random effects:

```python
fit = ed.fit_mixed_model(design, y, groups="batch")          # REML by default
fit = ed.fit_mixed_model(design, y, groups=batch_ids, method="ml")
print(fit.re_var, fit.summary_frame())
```

Typical use: analyse already-collected split-plot / multi-batch data. Generating
split-plot *designs* is still outside the advisor catalog (flagged in caveats);
analysis of grouped data is supported here.

## Agent-friendly serialization

```python
fit.to_dict()     # schema: doekit.FitResult/1
mix.to_dict()     # schema: doekit.MixedFitResult/1
ed.evaluate(d).to_dict()
ed.recommend_design("screening", 5).to_dict()
```

Stable JSON schemas so MCP / notebooks / agents share the same contract.
