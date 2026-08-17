# Métricas de evaluación

## Motivación

Este es el diferenciador de doekit. Construir un diseño es fácil; saber si de verdad
responderá tu pregunta —antes de gastar una sola corrida cara— es la parte difícil que
JMP y Design-Expert cobran. La capa de evaluación certifica un diseño contra el óptimo
teórico. Todo se calcula en **unidades codificadas**.

## Teoría

Sea $X$ la matriz de modelo $N \times p$ y $M = X^\top X$.

**Eficiencias relativas** (como porcentajes, 100% = ideal):

$$
D_\text{ef} = 100\,\frac{\det(M)^{1/p}}{N}, \qquad
A_\text{ef} = 100\,\frac{p}{N\,\operatorname{tr}(M^{-1})} .
$$

La **varianza de predicción escalada (SPV)** en un punto $x$ mide con qué precisión el
modelo predice ahí:

$$
\text{SPV}(x) = N\, x^\top M^{-1} x .
$$

El teorema de equivalencia general da $\max_x \text{SPV}(x) \ge p$, así que
$G_\text{ef} = 100\,p / \max_x \text{SPV}(x)$. Ordenar la SPV sobre una muestra de la
región y graficarla contra la fracción acumulada da el **Fraction of Design Space (FDS)
plot** — el estándar de oro: una curva baja y plana significa predicción uniforme y
precisa en todas partes.

La **potencia** del coeficiente $\beta_i$ usa la $t$ no-central con no-centralidad
$\delta_i = \text{tamaño de efecto} / (\sigma\sqrt{(M^{-1})_{ii}})$ y $N-p$ grados de
libertad — la probabilidad de detectar un efecto del tamaño anticipado.

El **VIF** $= 1/(1 - R_j^2)$ señala multicolinealidad ($1$ = ortogonal). La **matriz de
alias** $A = (X_1^\top X_1)^{-1} X_1^\top X_2$ cuantifica el sesgo
$\mathbb{E}[\hat\beta_1] = \beta_1 + A\beta_2$ por términos omitidos $X_2$.

## En doekit

```python
import doekit as ed

bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})
ev = ed.evaluate(bb, effect_size=1.0, sigma=1.0)
print(ev.summary())               # D/A/G, SPV, potencia, VIF

ed.plotting.fds_plot(bb)          # Fraction of Design Space
ed.plotting.power_plot(ev.power)
ed.alias_matrix(bb)               # sesgo por 2FI omitidas
```

## Varianza del diseño vs incertidumbre del surrogate

Las métricas de arriba (SPV, FDS, G) son propiedades **a priori** del *diseño*:
describen la varianza de predicción sobre la región *antes de tener datos*. Cuando
hay respuestas, el [loop de optimización](bayesian-optimization.md) ajusta un
surrogate cuyo $\sigma(x)$ **a posteriori** crece lejos de las corridas observadas y
se audita con la **cobertura leave-one-out**. Usa las métricas del diseño para
*planificar*; usa la calibración del surrogate para decidir si *confiar en un óptimo
predicho*.

## Ver también

- Teoría: [Reporte](reporting.md), [Diseño óptimo](optimal-design.md),
  [Optimización bayesiana](bayesian-optimization.md)
- API: [`evaluate`, `efficiencies`, `power_analysis`, `vif`, `alias_matrix`, `fds_data`](../api/evaluation.md)
