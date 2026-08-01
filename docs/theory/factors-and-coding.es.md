# Factores y codificación

## Motivación

El experimentador piensa en **unidades naturales** (°C, mol/L, una marca de
catalizador). El álgebra del DoE —ortogonalidad, eficiencias, matrices de modelo—
solo se comporta bien en **unidades codificadas**, donde cada factor vive en la misma
escala adimensional. La codificación es el puente entre ambas, y saltársela distorsiona
en silencio toda eficiencia que calcules.

## Teoría

Para un factor continuo en el intervalo natural $[\ell, h]$, la codificación lineal
estándar mapea los extremos a $\pm 1$:

$$
x_\text{cod} = 2\,\frac{x - \ell}{h - \ell} - 1,
\qquad
x_\text{nat} = \ell + \frac{x_\text{cod} + 1}{2}\,(h - \ell).
$$

de modo que $\ell \mapsto -1$, el punto medio $\mapsto 0$ y $h \mapsto +1$. Dos
consecuencias importan:

- **Invariancia de escala.** Las eficiencias (D/A/G) dependen de la matriz de
  información $M = X^\top X$. En unidades codificadas las columnas comparten escala, así
  que $M$ es comparable entre factores; en unidades naturales un factor medido en
  centenas dominaría $M$ solo por su magnitud.
- **Interpretabilidad.** Un coeficiente codificado es el cambio en la respuesta sobre
  *la mitad* del rango del factor, y el *efecto* clásico del DoE es
  $\text{media}(+1) - \text{media}(-1) = 2\beta$.

Los factores categóricos no se codifican a $\pm 1$; se codifican como **dummies** en la
matriz de modelo (primer nivel como referencia).

## En doekit

```python
import doekit as ed

temp = ed.ContinuousFactor("temp", low=20, high=80)
temp.encode([20, 50, 80])   # -> [-1, 0, 1]
temp.decode([-1, 0, 1])     # -> [20, 50, 80]

ed.DiscreteFactor("reps", levels=[1, 2, 3])          # numérico, ajusta al decodificar
ed.CategoricalFactor("catalyst", levels=["A", "B"])  # dummy en el modelo
```

Cada constructor de diseño acepta factores y devuelve la matriz de corridas en unidades
**naturales**, conservando la codificada como metadato; la capa de evaluación recodifica
automáticamente antes de calcular cualquier métrica.

## Ver también

- API: [`ContinuousFactor`, `DiscreteFactor`, `CategoricalFactor`](../api/factors-model.md)
