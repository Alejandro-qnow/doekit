# Diseño óptimo

## Motivación

Las plantillas estándar (factorial, Box-Behnken, CCD) asumen una región cúbica/esférica
agradable y un modelo fijo. Cuando la región es **irregular**, algunas combinaciones son
**inviables**, o el presupuesto de corridas es un número fijo arbitrario, ninguna
plantilla encaja. El diseño óptimo selecciona el mejor subconjunto de un **candidate
set** según un criterio sobre la matriz de información.

## Teoría

Para una matriz de modelo $X$, la **matriz de información** es $M = X^\top X$; la
covarianza de los estimados por mínimos cuadrados es $\sigma^2 M^{-1}$. Clásicamente
unos criterios se *maximizan* y otros se *minimizan*; doekit siempre expone un score
**mayor-es-mejor** (recíprocos cuando hace falta), alineado con `d_criterion` …
`i_criterion`:

| Criterio | Meta clásica | Score doekit (↑ mejor) | Significado |
|---|---|---|---|
| **D** | maximizar $\det(M)$ | $\det(M/N)^{1/p}$ | información conjunta (el más usado) |
| **A** | minimizar $\operatorname{tr}(M^{-1})$ | $p\,/\,(N\,\operatorname{tr}(M^{-1}))$ | precisión media de coeficientes |
| **I** | minimizar var. media de predicción | recíproco de esa media | mejor para *predecir* |
| **G** | minimizar $\max_x \operatorname{Var}\hat y(x)$ | $p\,/\,\max_i H_{ii}$ | predicción en el peor punto |
| **E** | maximizar $\min\lambda(M)$ | $\min\lambda(M)\,/\,N$ | dirección peor-condicionada |
| **T** | maximizar $\operatorname{tr}(M)$ | $\operatorname{tr}(M)\,/\,(N p)$ | "magnitud" de la información |

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
