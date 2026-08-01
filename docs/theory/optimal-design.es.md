# Diseño óptimo

## Motivación

Las plantillas estándar (factorial, Box-Behnken, CCD) asumen una región cúbica/esférica
agradable y un modelo fijo. Cuando la región es **irregular**, algunas combinaciones son
**inviables**, o el presupuesto de corridas es un número fijo arbitrario, ninguna
plantilla encaja. El diseño óptimo selecciona el mejor subconjunto de un **candidate
set** según un criterio sobre la matriz de información.

## Teoría

Para una matriz de modelo $X$, la **matriz de información** es $M = X^\top X$; la
covarianza de los estimados por mínimos cuadrados es $\sigma^2 M^{-1}$. Cada criterio de
optimalidad comprime $M$ en un escalar a maximizar (doekit usa la convención "mayor es
mejor" en todo):

| Criterio | Optimiza | Significado |
|---|---|---|
| **D** | $\det(M)$ | información conjunta máxima (el más usado) |
| **A** | $\operatorname{tr}(M^{-1})$ | mínima varianza media de los coeficientes |
| **I** | varianza media de predicción sobre la región | mejor para *predecir* |
| **G** | $\max_x \operatorname{Var}\hat y(x)$ | mínima varianza de predicción en el peor punto |
| **E** | $\min \lambda(M)$ | mejor dirección peor-condicionada |
| **T** | $\operatorname{tr}(M)$ | máxima "magnitud" de la información |

Dos algoritmos de intercambio recorren el candidate set:

- **KL-exchange** — especializado y eficiente para **D**-optimalidad (Atkinson, Donev &
  Tobias); el default para `criterion="D"`. Intercambia puntos de baja varianza por
  candidatos de alta varianza usando la actualización de determinante de rango uno
  $\Delta = (1 - d_k)(1 + d_l) + (x_k^\top M^{-1} x_l)^2$.
- **Fedorov** — intercambio genérico que optimiza **cualquier** criterio; el default en
  el resto de casos.

`n_starts > 1` corre varios arranques aleatorios y se queda con el mejor, escapando de
óptimos locales.

## En doekit

```python
import doekit as ed

cand = ed.random_design([ed.ContinuousFactor("x1", -1, 1),
                         ed.ContinuousFactor("x2", -1, 1)], n=300, seed=0)
cand.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")

opt = ed.optimal_design(cand, n_runs=12, criterion="D",
                        algorithm="kl", n_starts=5, seed=1)
opt.metadata["criteria"]          # todos los criterios del diseño final
```

## Ver también

- Teoría: [Métricas de evaluación](evaluation-metrics.md)
- API: [`optimal_design`, `kl_exchange`, `fedorov_exchange`, criterios](../api/designs.md)
