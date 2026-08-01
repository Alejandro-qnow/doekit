# Changelog

Todas las modificaciones relevantes de `doekit` se documentan aquí.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el
proyecto adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Unreleased]

## [0.5.0] - 2026-08-01

### Añadido — DoE secuencial / adaptativo
- **`augment_design`**: añade corridas D/I-óptimas **condicionadas** a las filas
  ya existentes (greedy + exchange sobre candidate set).
- **`propose_next_runs`** + **`NextRunsProposal`** (`doekit.NextRunsProposal/1`):
  lote siguiente, métricas before/after, rationale, caveats; con `response`
  estima `sigma_hat` y términos activos.
- **`compare_designs`** + **`DesignComparison`** (`doekit.DesignComparison/1`):
  Δ D/A/G, SPV medio, potencia media, n_runs — “¿valen N corridas más?”.
- Bridge BO delgado: **`candidates_from_bounds`**,
  **`candidates_from_skopt_space`** (extra opcional `[bo]` = scikit-optimize).
- Asesor: caveat que apunta a `propose_next_runs` tras la primera oleada.
- Docs: `theory/sequential-doe.md`, notebook `10_sequential_augment.ipynb`,
  `project/ROADMAP.md` (0.6 mixture/SPD, 0.7 `ed.experiment`).

## [0.4.0] - 2026-08-01

### Añadido — análisis experimental serio (`statsmodels` central)
- **`statsmodels>=0.14`** pasa a dependencia **central** del núcleo (ya no solo
  numpy/pandas/scipy).
- **`fit_linear_model`**: parámetros `blocks=` (columna o array; respeta
  `metadata["blocking"]`; `False` fuerza sin bloques) y `cov_type=`
  (`nonrobust` / `HC0` / `HC1` / `HC3`).
- **`attach_blocks`**: añade columna de bloque + metadata.
- **`anova_table`**: tabla partial-F / Wald por término.
- **`lack_of_fit`**: descompone RSS en pure error vs falta de ajuste (requiere réplicas).
- **`fit_mixed_model`** + **`MixedFitResult`**: MixedLM (REML/ML) con `groups=` y
  `re_formula`.
- Serialización agent-first: `FitResult.to_dict/from_dict`
  (`doekit.FitResult/1`), `MixedFitResult.to_dict/from_dict`
  (`doekit.MixedFitResult/1`), `Recommendation.to_dict`
  (`doekit.Recommendation/1`), `DesignEvaluation.to_dict`
  (`doekit.DesignEvaluation/1`).
- `report_summary` expone `fit`, `anova` y opcionalmente `mixed_fit`
  (`blocks`, `cov_type`, `groups`).
- Docs: `theory/blocked-and-mixed.md` (+ ES); notebook
  `09_analisis_bloques_mixed.ipynb`.

### Cambiado
- Asesor (`recommend_design`): strings y columnas de tabla en **inglés**
  (`method`, `runs`, `D-optimal`, `Full factorial`, …); caveats mencionan
  `fit_mixed_model` / `blocks=` para datos agrupados.
- Notas del reporte (`recommendation` block) en inglés por defecto.
- **Código en inglés**: docstrings y comentarios de todo el paquete traducidos a
  inglés en estilo **NumPy** (Parameters/Returns/Notes), con docstrings añadidos a
  las funciones públicas que faltaban y type hints completados en `plotting`. Los
  identificadores públicos ya estaban en inglés — **la API no cambia**.
- **Reporte por folder (nuevo default)**: `doekit.report` ahora escribe una carpeta
  `report/` con `index.html` + `report.css` + `images/*.png` + `data/*.csv`
  (matriz, coeficientes, eficiencias, power, VIF, FDS, anomalías) en vez de un único
  HTML. El modo autocontenido base64 sigue disponible con `self_contained=True`.
- **Reporte bilingüe**: parámetro `lang` (`"en"` por defecto, `"es"`) en `report` y
  `report_summary`. Los estilos se extrajeron a `doekit/assets/report.css` (mejor
  CSS: banda de veredicto, zebra, grid de gráficas, `@media print`).
- **Robustez de gráficas**: si una gráfica falla, el reporte muestra un aviso visible
  en vez de omitirla en silencio.

## [0.3.0] - No publicado

### Añadido — asesor de diseño experimental
- **`doekit.recommend_design`** (`recommend.py`) — asesor **transparente** (reglas +
  evaluación) que recomienda el mejor método para un caso: las reglas acotan el
  shortlist y la evaluación lo rankea por prioridades (`runs`/`precision`/`prediction`)
  con **media geométrica ponderada** (un eje catastrófico hunde el score, sin
  compensación). Devuelve `Recommendation` (método, `Design`, tabla de alternativas,
  justificación y **salvedades**: trade-off multiobjetivo, condicional a `model_order`,
  no cubre mezcla ni split-plot). Robusto a factores categóricos y presupuesto insuficiente.
- El **reporte HTML** y `report_summary` incluyen una recomendación **coherente con lo
  ejecutado**: infieren el escenario del diseño y comparan (coincide / sugieren alternativa,
  siempre con framing informativo). Añadida a los 7 notebooks.
- D-óptimo del asesor construido sobre rejilla de 3 niveles con metadata de factores
  (evaluación en espacio codificado correcta). Tests: `test_recommend.py`.

### Añadido — módulo de reporte y base para el MCP
- **`doekit.report`** — genera un **HTML autocontenido** (CSS embebido, gráficas en
  base64) con resumen ejecutivo, metodología, calidad del diseño (semáforo de
  eficiencias + FDS + power + VIF + alias), resultados del análisis, **valores
  anómalos** y conclusiones/recomendaciones. Narrativa **por reglas** (determinista).
- Argumento `report=` en `evaluate`, `fit_linear_model` y `optimal_design` (carpeta,
  `True` → `./reports/`, o dict de opciones); la ruta queda en `*.report_path`.
- **Diagnósticos de anomalías** en `FitResult`: `leverage`, `studentized_resid`,
  `cooks_distance`, `r_squared_adj`, y `FitResult.anomalies()` (outlier |t|>3, alto
  leverage h>2p/N, influyente Cook's D>1 — cutoff robusto para diseños pequeños).
- **Serialización MCP-friendly**: `Design.to_dict/from_dict` (+ `to_json/from_json`),
  `Model.to_dict/from_dict`, `Factor.to_dict()` + `factor_from_dict`.
- **`report_summary(design, response=None, ...)`** — devuelve la guía semántica del
  reporte (metodología, resumen ejecutivo, recomendaciones, calidad, anomalías) como
  estructura de datos, sin escribir HTML. Útil para mostrarla inline en notebooks y
  para el futuro tool `analyze_results` del MCP.
- Los **7 notebooks** cierran ahora con una sección de reporte (HTML + guía semántica
  inline), validando que la guía es coherente con el análisis.
- Coherencia de la narrativa (detectada al validar los notebooks): (a) eficiencias
  D/A/G clampeadas a ≤100 % (el muestreo de la región inflaba G en diseños saturados);
  (b) ajuste saturado (dof≤0) apunta al half-normal en vez de "nada es significativo";
  (c) la D-eficiencia típica de un RSM estándar (BBD/CCD/DSD) ya no se marca como defecto.
- Extra `[report]` (matplotlib). Ejemplo `examples/reporte_html.py`.
- Planes escritos: `docs/PLAN_REPORTING.md` y `docs/PLAN_MCP.md`.
- Tests: `test_report.py`, `test_serialization.py` (74 en total).

## [0.2.0] - No publicado

### Añadido — factor diferenciador: evaluación y benchmarking de diseños
- **Módulo `doekit.evaluate`** (el "boletín de calidad" de un DoE, ausente en
  pyDOE3/dexpy): `evaluate` devuelve un `DesignEvaluation` con
  - **eficiencias D/A/G (%)** relativas al óptimo teórico — *"qué tan lejos estás
    del diseño correcto"* — computadas en unidades **codificadas**;
  - **FDS plot** (`fds_data`) — varianza de predicción escalada sobre la región;
  - **power analysis** por coeficiente (`power_analysis`, t no-central);
  - **matriz de alias** (`alias_matrix`) y **VIF** (`vif`).
  - Guardia de rango: los diseños saturados/supersaturados se marcan en vez de
    reportar eficiencias sin sentido.
- **Definitive Screening Designs** (`definitive_screening`, Jones & Nachtsheim
  2011) vía matrices de conferencia de Paley; efectos principales ortogonales a
  curvatura e interacciones.
- **Visualizaciones** de evaluación: `plotting.fds_plot`, `power_plot`,
  `alias_heatmap`.
- Paquete **renombrado a `doekit`** (marca DoEKit); `pydoe`/`pyDOE*` ya estaban
  ocupados en PyPI.
- Tests: `test_evaluate.py`, `test_definitive.py` (26 nuevos casos; 61 en total).
- **Notebooks de dominio 04-07** (química, optimización, ML, quantum ML) con el
  patrón *construir → evaluar → benchmarkear*: cada uno usa una función-verdad
  conocida para medir qué tan bien el DoE recupera el óptimo y lo compara contra un
  baseline (aleatorio / grid / random search). Notebooks 01-03 refrescados con la
  capa de evaluación integrada.

### Añadido
- Metadatos completos de empaquetado (`authors`, `license`, `keywords`,
  `classifiers`, `project.urls`), `LICENSE` (MIT) y marcador `py.typed` (PEP 561).
- Versión de fuente única vía `[tool.hatch.version]` (lee `__version__`).
- Guía de conceptos DoE y referencia de API en `docs/`.
- `CONTRIBUTING.md` y este `CHANGELOG.md`.
- `main_effects(..., scale="effect")`: devuelve el efecto clásico de DoE
  (`media(+1) − media(−1) = 2·β`) además del coeficiente de regresión.
- Tests para el chequeo endurecido de Plackett-Burman, la escala de efectos y
  el flujo de factores categóricos a través de `optimal_design`.

### Cambiado (corrección de fidelidad teórica)
- **`is_plackett_burman` endurecido.** Exige entradas `±1`, que **cada** columna
  sume cero y ortogonalidad entrada a entrada (`DᵀD = N·I`), en lugar de sumas
  agregadas que aceptaban matrices no ortogonales.
- Nota de escala documentada en `i_criterion` (la matriz de información no se
  normaliza por `N`, coherente con su uso solo para *ranking*).

## [0.1.0] - 2026-07-24

### Añadido
- Lanzamiento inicial de `doekit`:
  - Screening: `plackett_burman` (Sylvester + Paley I/II), `fold`,
    `is_plackett_burman`.
  - Factoriales: `full_factorial` (explícito y perezoso), `fractional_factorial`
    (fracción real `2^(k-p)` con relación de definición, resolución y alias).
  - Superficie de respuesta: `box_behnken`, `central_composite`.
  - Aleatorios: `random_design`, `latin_hypercube`.
  - Óptimos: `optimal_design` (D/A/I vía KL-exchange y Fedorov, multi-arranque),
    criterios `d/a/t/g/e/i_criterion`.
  - Factores con codificación natural↔codificada, mini-DSL de modelo, capa de
    análisis (`fit_linear_model`, `main_effects`, `half_normal_data`) y gráficos.
