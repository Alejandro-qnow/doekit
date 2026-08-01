# Factors and coding

## Motivation

An experimenter thinks in **natural units** (°C, mol/L, a catalyst brand). The
algebra of DoE — orthogonality, efficiencies, model matrices — only behaves well in
**coded units**, where every factor lives on the same dimensionless scale. Coding is
the bridge between the two, and skipping it silently distorts every efficiency you
compute.

## Theory

For a continuous factor on the natural interval $[\ell, h]$, the standard linear
coding maps the endpoints to $\pm 1$:

$$
x_\text{coded} = 2\,\frac{x - \ell}{h - \ell} - 1,
\qquad
x_\text{natural} = \ell + \frac{x_\text{coded} + 1}{2}\,(h - \ell).
$$

so that $\ell \mapsto -1$, the midpoint $\mapsto 0$ and $h \mapsto +1$. Two
consequences matter:

- **Scale invariance.** Efficiencies (D/A/G) depend on the information matrix
  $M = X^\top X$. In coded units the columns share a common scale, so $M$ is
  comparable across factors; in natural units a factor measured in the hundreds
  would dominate $M$ purely by magnitude.
- **Interpretability.** A coded coefficient is the change in response over *half* the
  factor range, and the classic DoE *effect* is $\text{mean}(+1) - \text{mean}(-1) =
  2\beta$.

Categorical factors cannot be coded to $\pm 1$; they are **dummy-coded** in the model
matrix (first level as reference).

## In doekit

```python
import doekit as ed

temp = ed.ContinuousFactor("temp", low=20, high=80)
temp.encode([20, 50, 80])   # -> [-1, 0, 1]
temp.decode([-1, 0, 1])     # -> [20, 50, 80]

ed.DiscreteFactor("reps", levels=[1, 2, 3])          # numeric, snaps on decode
ed.CategoricalFactor("catalyst", levels=["A", "B"])  # dummy-coded in the model
ed.MixtureFactor("A", lower=0, upper=1)              # simplex component (Σ xᵢ = 1)
```

Design constructors that take factor bounds return the run matrix in **natural**
units and keep the `Factor` objects on the design. The evaluation / analysis layer
re-encodes to ±1 (or the appropriate coding) from those factors before building
the model matrix or any metric — coded columns are not stored as a separate
metadata matrix.

## See also

- Theory: [Mixture and split-plot](mixture-and-split-plot.md)
- API: [`ContinuousFactor`, `DiscreteFactor`, `CategoricalFactor`, `MixtureFactor`](../api/factors-model.md)
