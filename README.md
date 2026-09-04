# doekit

**Design of Experiments (DoE) en Python** con un **enfoque híbrido** — **personas** (laboratorio) y **agentes LLM** que lo consumen **vía MCP** — sobre un mismo motor. doekit no solo *construye* diseños: los **interpreta** y los **evalúa**, con rigor.

Una **capa semántica** (`interpret`/`Interpretation`) convierte cifras en significado — *qué significa, por qué, qué hacer, con qué salvedades* — **sin inventar nada**: *los hechos vienen de doekit; el juicio, del usuario*. Misma verdad, dos superficies: `summary()`/reportes HTML para personas, `to_dict()`/`for_llm()` para agentes. Cubre screening, factoriales, superficie de respuesta, diseño óptimo (D/A/I), mezcla/split-plot, DoE secuencial (aprender u optimizar) y análisis (OLS con bloques/SE robustos, modelos mixtos).

Depende de `numpy`, `pandas`, `scipy` y `statsmodels`. `matplotlib` es opcional (gráficos); `doekit[mcp]` sirve las tools para agentes y `doekit[bo]` añade el surrogate GP.

- **Documentación (EN/ES):** [doekit](https://doekit.vercel.app) — flujo DoE → learn|optimize → capa semántica → decide, teoría, API y agentes/MCP.
- **Fuente de la doc:** [`docs/`](docs/) (MkDocs Material).
- **Notebooks de casos de uso:** [`notebooks/`](notebooks/)

## Instalación

```bash
pip install doekit            # núcleo
pip install "doekit[plot]"    # con gráficos (matplotlib)
pip install "doekit[report]"  # reportes HTML
pip install "doekit[export]"  # Excel (openpyxl)
pip install "doekit[mcp]"     # servidor MCP para agentes
pip install "doekit[bo]"      # surrogate GP (optimize)
```

Desde el repositorio, para desarrollo:

```bash
uv sync --extra dev                        # entorno completo (pytest + matplotlib + build)
# o con pip:
pip install -e ".[dev,plot]"
```

## Uso rápido

```python
import doekit as ed

# --- Screening: Plackett-Burman para 6 factores en 8 corridas ---
pb = ed.plackett_burman(6)
ed.is_plackett_burman(pb)          # True (ortogonal, cada columna suma cero)

# Factorial fraccional REAL 2^(3-1) con estructura de alias
fr = ed.fractional_factorial(3, generators=["C=AB"])
fr.metadata["defining_relation"]   # 'I = ABC'
fr.metadata["resolution"]          # 'III'
fr.metadata["aliases"]             # [['factor1', 'factor2:factor3'], ...]

# --- Superficie de respuesta en unidades naturales ---
bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})
cc = ed.central_composite(3)       # alpha ortogonal ~ 1.826

# --- Diseño D-óptimo desde un candidate set (KL-exchange) ---
cand = ed.random_design([ed.ContinuousFactor("x1", -1, 1),
                         ed.ContinuousFactor("x2", -1, 1)], n=200, seed=0)
cand.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")
opt = ed.optimal_design(cand, n_runs=12, criterion="D", n_starts=5, seed=1)
opt.metadata["criteria"]           # {'D': ..., 'A': ..., 'I': ...}

# --- Análisis: efectos, bloques, robust SE, mixed ---
y = ...                            # respuestas medidas por corrida
effects = ed.main_effects(pb, y, scale="effect")   # efecto clásico = 2·beta
fit = ed.fit_linear_model(pb, y, cov_type="HC3")
fit = ed.fit_linear_model(design, y, blocks="block")
mix = ed.fit_mixed_model(design, y, groups="batch")
fit.summary_frame()                # DataFrame con estimate/std_error/t/p

# --- Bucle end-to-end ---
exp = ed.experiment(goal="screening", factors=6, budget=12)
exp.evaluate()
exp.export_csv("runs.csv")         # plantilla de laboratorio
exp.ingest(y)                      # datos reales
nxt = exp.next(n_add=4)            # learn (aumentación clásica)
# nxt = exp.next(n_add=4, intent="optimize")  # surrogate + adquisición (doekit[bo])
print(ed.interpret(nxt).for_llm()) # capa semántica → contexto para agentes
dec = exp.decide_next()            # stop | augment | refine | redesign
```

Cada constructor devuelve un objeto `Design` con `.matrix` (pandas), `.model`,
`.metadata` y utilidades (`.n_runs`, `.model_matrix()`, `repr` informativo).

## Capacidades


| Categoría                     | Funciones                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| Screening                     | `plackett_burman`, `fractional_factorial`, `fold`, `is_plackett_burman`                           |
| Factoriales                   | `full_factorial` (explícito y perezoso)                                                           |
| Superficie de respuesta       | `box_behnken`, `central_composite`                                                                |
| Aleatorios                    | `random_design`, `latin_hypercube`                                                                |
| Screening moderno             | `definitive_screening` (Definitive Screening Designs, Jones & Nachtsheim)                         |
| Óptimos                       | `optimal_design` (D/A/I vía KL-exchange o Fedorov, multi-arranque)                                |
| Criterios                     | `d/a/t/g/e/i_criterion`                                                                           |
| **Evaluación / benchmarking** | `evaluate`, `efficiencies`, `power_analysis`, `alias_matrix`, `vif`, `fds_data`                   |
| **Capa semántica**            | `interpret` → `Interpretation` (`summary` / `for_llm` / `to_dict`)                                  |
| **Decisión**                  | `decide_next_action`, `Experiment.decide_next` → stop / augment / refine / redesign                 |
| **Optimize (ML/BO)**          | `propose_next_runs(intent="optimize")`, surrogate OLS/GP, EI/PI/UCB/EHVI, Pareto                    |
| **Reporte**                   | `report` (HTML: metodología, calidad, resultados, anomalías, recomendaciones)                       |
| **Asesor**                    | `recommend_design` (reglas + evaluación: recomienda el mejor método para el caso)                 |
| Factores                      | `ContinuousFactor`, `DiscreteFactor`, `CategoricalFactor` (codificación natural↔codificada)       |
| Modelo                        | `Model.parse`, `Model.full_quadratic`, `Model.main_effects`                                       |
| Análisis                      | `fit_linear_model` (blocks, HC), `anova_table`, `lack_of_fit`, `fit_mixed_model`, `main_effects`  |
| **Secuencial**                | `augment_design`, `propose_next_runs`, `compare_designs`, `candidates_from_bounds`                 |
| **Experiment**                | `ed.experiment(...)` — plan → evaluate → ingest → next → report → export CSV/Excel                 |
| **Workspace / MCP**           | `ed.project` / waves; `doekit[mcp]` (recommend · evaluate · propose_and_decide)                     |
| Mezcla / split-plot           | `simplex_lattice`, `simplex_centroid`, `split_plot_design`, `MixtureFactor`, `Constraints`         |
| Gráficos                      | `half_normal_plot`, `effects_plot`, `correlation_plot`, `fds_plot`, `power_plot`, `alias_heatmap` |
| CLI                           | `doekit recommend|evaluate|experiment|project`                                                     |


## Lo que nos distingue: evaluar el diseño, no solo construirlo

La mayoría de las librerías Python de DoE (`pyDOE3`, `dexpy`) **solo generan**
diseños. `doekit` además les da un **boletín de calidad** reproducible que
responde *"¿qué tan lejos está mi diseño del óptimo teórico?"* — lo que en las
herramientas comerciales (JMP, Design-Expert) es la mitad del trabajo:

```python
import doekit as ed

bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})
report = ed.evaluate(bb, effect_size=1.0, sigma=1.0)   # tamaño de efecto anticipado
print(report.summary())
#   D-efficiency : 36.6 %   A-efficiency : 29.4 %   G-efficiency : 51.7 %
#   SPV min/mean/max, power por término, VIF ...

ed.plotting.fds_plot(bb)          # Fraction of Design Space (varianza de predicción)
ed.plotting.power_plot(report.power)
```

Incluye además **Definitive Screening Designs** (Jones & Nachtsheim, 2011), el
diseño de screening moderno que estima efectos principales libres de sesgo de
curvatura e interacciones en `2m+1` corridas:

```python
dsd = ed.definitive_screening({f"x{i+1}": (0, 10) for i in range(6)})  # 13 corridas
ed.evaluate(dsd).vif.max()        # 1.0  (efectos principales ortogonales)
```

### Reporte HTML de una línea

Todo lo anterior se condensa en un único HTML elegante y autocontenido —
metodología, calidad (con semáforo), resultados, valores anómalos y
recomendaciones— pasable como argumento del propio experimento:

```python
ed.report(bb, response=y, output_dir="reports/")      # standalone
ed.fit_linear_model(bb, y, report="reports/")         # o como argumento
```

## Notebooks

Los `[notebooks/](notebooks/)` son cuadernos explicativos con narrativa y gráficas:

- **01–03** — el flujo clásico: screening → superficie de respuesta → diseño óptimo
(con la capa de evaluación integrada).
- **04–07 (por dominio)** — cada uno aplica el patrón **construir → evaluar →
benchmarkear** usando una *función-verdad conocida* para medir qué tan bien el DoE
recupera el óptimo, y comparándolo contra un baseline:

  | Notebook | Dominio          | Hallazgo                                                                   |
  | -------- | ---------------- | -------------------------------------------------------------------------- |
  | 04       | Química          | recupera el óptimo de una reacción con ~1.8× menos error que el azar       |
  | 05       | Optimización     | diseño D-óptimo en región restringida bate al azar en calidad de surrogate |
  | 06       | Machine Learning | supera a *random search* en el 98% de las réplicas, a presupuesto igual    |
  | 07       | Quantum ML       | mejor kernel con menos evaluaciones de circuito (cada una = tiempo de QPU) |


## Tests

```bash
uv run pytest -q       # o: pytest -q
```

## Licencia

MIT — ver `[LICENSE](LICENSE)`.