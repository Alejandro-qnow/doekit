# Diseños de superficie de respuesta

## Motivación

Cuando el screening ya encontró los pocos factores que importan, el objetivo cambia de
*cuáles* a *cuánto*: localizar el óptimo y mapear la respuesta cerca de él. Esto necesita
un modelo **cuadrático**, que necesita al menos tres niveles por factor. Los dos
caballos de batalla son Box-Behnken y Central Composite.

## Teoría

El modelo de superficie de respuesta es el cuadrático completo

$$
y = \beta_0 + \sum_i \beta_i x_i + \sum_{i<j} \beta_{ij} x_i x_j
       + \sum_i \beta_{ii} x_i^2 + \varepsilon .
$$

**Box-Behnken (BBD)** coloca puntos en los *puntos medios de las aristas* del cubo más
puntos centrales — nunca en los vértices. Esto evita combinaciones extremas de factores
(a menudo inviables o costosas) y mantiene bajo el número de corridas; requiere
$\ge 3$ factores.

**Central Composite (CCD)** aumenta un factorial de dos niveles con puntos **estrella**
(axiales) a distancia $\pm\alpha$ y puntos centrales. La elección de $\alpha$ fija una
propiedad geométrica:

- **rotable** — varianza de predicción constante a igual distancia del centro,
  $\alpha = (2^n)^{1/4}$;
- **ortogonal** — bloques ortogonales;
- **faced** — $\alpha = 1$ (estrellas sobre las caras, solo tres niveles).

## En doekit

```python
import doekit as ed

bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})
cc = ed.central_composite(3, alpha="rotatable")
cc.metadata["alpha_value"]        # distancia estrella α

# ambos usan por defecto un modelo cuadrático completo:
fit = ed.fit_linear_model(bb, y)  # y = respuestas medidas
```

## Ver también

- Teoría: [Definitive screening](definitive-screening.md),
  [Métricas de evaluación](evaluation-metrics.md)
- API: [`box_behnken`, `central_composite`](../api/designs.md)
