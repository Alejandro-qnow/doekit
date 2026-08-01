# Definitive Screening Designs

## Motivation

Classical two-level screening has a blind spot: it cannot see **curvature**, and its
main effects can be biased by two-factor interactions. Running a separate
response-surface study afterwards doubles the cost. Definitive Screening Designs
(DSDs, Jones & Nachtsheim 2011) collapse both phases into one small, three-level
design — the most important screening advance of the last decade.

## Theory

For $m$ factors, a DSD uses $2m+1$ runs built from a **conference matrix** $C$ of
order $m$ (zero diagonal, $\pm 1$ off-diagonal, $C^\top C = (m-1)I$):

$$
\text{DSD} = \begin{bmatrix} C \\ -C \\ \mathbf{0} \end{bmatrix}.
$$

The fold $C / {-}C$ makes the **linear effects** (odd functions of the coded levels)
orthogonal to the **quadratic effects and two-factor interactions** (even functions).
Consequences:

- main effects are **unbiased** by 2FI and curvature;
- every factor is run at three levels $\{-1, 0, +1\}$, so **quadratic** effects are
  estimable;
- the design **projects** onto a response surface in the few active factors — no
  extra runs needed.

doekit builds the conference matrices by the **Paley** construction ($m = q+1$,
$q$ prime), covering orders $4, 6, 8, 12, 14, \dots$; when the smallest available
order exceeds $m$, the surplus columns become "phantom" factors that further reduce
estimate bias.

## In doekit

```python
import doekit as ed

dsd = ed.definitive_screening({f"x{i+1}": (0, 10) for i in range(6)})  # 13 runs
ed.evaluate(dsd).vif.max()        # ~1.0  (main effects orthogonal)
```

## See also

- Theory: [Response surface](response-surface.md),
  [Evaluation metrics](evaluation-metrics.md)
- API: [`definitive_screening`](../api/designs.md)
