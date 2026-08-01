# Screening: Plackett-Burman

## Motivación

Con muchos factores candidatos y un presupuesto ajustado, la primera pregunta no es
*cuánto* importa cada factor sino *cuáles* importan siquiera. Los diseños de screening
lo responden con el mínimo de corridas, apoyándose en el **principio de esparsidad de
efectos**: solo unos pocos factores dominan la respuesta.

## Teoría

Un diseño de Plackett-Burman (PB) estima $k$ efectos principales de forma
**ortogonal** en $N$ corridas, donde $N$ es múltiplo de 4 con $N > k$. La ortogonalidad
significa que la matriz de diseño $D \in \{-1,+1\}^{N\times(N-1)}$ cumple

$$
D^\top D = N\, I,
$$

de modo que los efectos principales estimados no están correlacionados y cada uno tiene
la menor varianza posible para $N$ corridas. Los PB se construyen a partir de
**matrices de Hadamard** $H$ ($H H^\top = N I$); doekit las genera vía **Sylvester**
(potencias de dos) y **Paley I/II** ($q+1$ y $2(q+1)$ con $q$ primo), cubriendo los
órdenes Hadamard múltiplos de 4 habituales.

El precio de las corridas mínimas: los efectos principales quedan **aliasados con las
interacciones de dos factores** (resolución III). Si hay interacciones activas, sesgan
los estimados de los efectos principales — que es justo lo que resuelven el
[folding](fractional-factorial.md) y el [definitive screening](definitive-screening.md).

## En doekit

```python
import doekit as ed

pb = ed.plackett_burman(6)      # 8 corridas, 6 factores (+ columnas dummy)
ed.is_plackett_burman(pb)       # True: ±1, columnas suma-cero, DᵀD = N·I

# Analizar cuando tengas respuestas y:
effects = ed.main_effects(pb, y, scale="effect")   # efecto clásico = 2·β
ed.plotting.half_normal_plot(effects.to_numpy(), effects.index.tolist())
```

El half-normal plot separa los pocos vitales (puntos que se despegan de la recta) de
los muchos triviales (puntos sobre la recta).

## Ver también

- Teoría: [Factorial fraccional](fractional-factorial.md),
  [Definitive screening](definitive-screening.md)
- API: [`plackett_burman`, `is_plackett_burman`, `fold`](../api/designs.md)
