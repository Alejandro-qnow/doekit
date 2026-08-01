# doekit

**Diseño de Experimentos (DoE) en Python**: screening, diseños factoriales,
superficie de respuesta y diseño óptimo (D/A/I) — más una **capa de evaluación de
diseños** que la mayoría de las librerías de DoE en Python no tiene.

Solo depende de `numpy`, `pandas` y `scipy`. `matplotlib` es opcional (gráficas y
reportes HTML).

```bash
pip install doekit            # núcleo
pip install "doekit[plot]"    # con gráficas (matplotlib)
pip install "doekit[report]"  # con reportes HTML
```

## El embudo del DoE

El DoE clásico es un embudo: **screening** (¿qué factores importan?) →
**superficie de respuesta** (¿dónde está el óptimo?), con el **diseño óptimo** como
vía alternativa cuando las plantillas estándar no encajan.

```python
import doekit as ed

# 1) Screening — Plackett-Burman para 6 factores en 8 corridas
pb = ed.plackett_burman(6)

# 2) Superficie de respuesta — Box-Behnken en unidades naturales
bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})

# 3) Diseño óptimo — subconjunto D-óptimo desde un candidate set
cand = ed.random_design([ed.ContinuousFactor("x1", -1, 1),
                         ed.ContinuousFactor("x2", -1, 1)], n=200, seed=0)
cand.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")
opt = ed.optimal_design(cand, n_runs=12, criterion="D", n_starts=5, seed=1)
```

## Lo que distingue a doekit: evaluar el diseño, no solo construirlo

La mayoría de las librerías de DoE en Python **solo generan** diseños. `doekit`
además le da a cada uno un **boletín de calidad** reproducible que responde *"¿qué
tan lejos está mi diseño del óptimo teórico?"* — la mitad del trabajo que hacen las
herramientas comerciales (JMP, Design-Expert):

```python
report = ed.evaluate(bb, effect_size=1.0, sigma=1.0)
print(report.summary())
#   D/A/G-eficiencia, distribución de la SPV (FDS),
#   potencia por término, VIF, estructura de alias ...

ed.report(bb, response=y)   # -> carpeta report/: index.html + images/ + data/
```

## Por dónde seguir

- **Teoría** — una página por metodología, cada una con *motivación → teoría (con
  matemática) → un ejemplo con doekit*. Empieza por
  [Factores y codificación](theory/factors-and-coding.md).
- **Referencia de API** — autogenerada desde los docstrings del código.
- **Guía** — los [notebooks](guide/notebooks.md) recorren casos reales por dominio
  (química, ML, ML cuántico) con el patrón *construir → evaluar → benchmarkear*.
