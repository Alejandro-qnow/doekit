# Response surface designs

## Motivation

Once screening has found the few factors that matter, the goal shifts from *which*
to *how much*: locate the optimum and map the response near it. This needs a
**quadratic** model, which needs at least three levels per factor. The two workhorses
are Box-Behnken and Central Composite designs.

## Theory

The response-surface model is the full quadratic

$$
y = \beta_0 + \sum_i \beta_i x_i + \sum_{i<j} \beta_{ij} x_i x_j
       + \sum_i \beta_{ii} x_i^2 + \varepsilon .
$$

**Box-Behnken (BBD)** places points at the *edge midpoints* of the cube plus center
points — never at the corners. This avoids extreme factor combinations (often
infeasible or costly) and keeps the run count low; it needs $\ge 3$ factors.

**Central Composite (CCD)** augments a two-level factorial with **star** (axial)
points at distance $\pm\alpha$ and center points. The choice of $\alpha$ sets a
geometric property:

- **rotatable** — constant prediction variance at equal distance from the center,
  $\alpha = (2^n)^{1/4}$;
- **orthogonal** — orthogonal blocking;
- **faced** — $\alpha = 1$ (stars on the faces, only three levels).

## In doekit

```python
import doekit as ed

bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})
cc = ed.central_composite(3, alpha="rotatable")
cc.metadata["alpha_value"]        # star distance α

# both default to a full quadratic model:
fit = ed.fit_linear_model(bb, y)  # y = measured responses
```

## See also

- Theory: [Definitive screening](definitive-screening.md),
  [Evaluation metrics](evaluation-metrics.md)
- API: [`box_behnken`, `central_composite`](../api/designs.md)
