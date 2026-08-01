# Reporte

## Motivación

El reporte es el artefacto de la capa de evaluación: lo que un científico comparte con
colegas o jefatura. Cierra el círculo **construir → evaluar → comunicar**.
`report_summary(...)` expone el mismo contenido como dict estructurado para agentes;
un paquete MCP dedicado está planificado fuera del núcleo (ver `project/PLAN_MCP.md`).

## Qué contiene el reporte

Un reporte HTML por reglas (determinista, sin LLM) con:

1. **Resumen ejecutivo** — veredicto de calidad, factores significativos, recomendación
   principal.
2. **Metodología** — tipo de diseño y *por qué*, factores con sus rangos, el modelo.
3. **Matriz del diseño** — colapsable.
4. **Calidad del diseño** — tarjetas de D/A/G-eficiencia con semáforo, FDS plot, power,
   VIF, mapa de alias, cada una con una glosa en lenguaje llano.
5. **Resultados** (con respuesta) — modelo ajustado, coeficientes con significancia,
   $R^2$, half-normal plot.
6. **Valores anómalos** — outliers (residuo studentizado $|r|>3$), alto leverage
   ($h_{ii} > 2p/N$), influyentes (Cook's $D > 1$).
7. **Conclusiones y recomendaciones** — narrativa por reglas.

## Modos de salida

- **folder** (por defecto): una carpeta `report/` con `index.html`, `report.css`,
  `images/*.png` y `data/*.csv` (matriz del diseño, coeficientes, eficiencias, power,
  VIF, FDS, anomalías) — assets portables que un investigador puede reutilizar.
- **autocontenido**: un único `.html` con CSS embebido y gráficas en base64 — enviable
  por correo.

El reporte es **bilingüe** (`lang="en"` / `"es"`).

## En doekit

```python
import doekit as ed

ed.report(bb, response=y)                        # -> carpeta report/ (por defecto)
ed.report(bb, response=y, self_contained=True)   # -> un único .html
ed.report(bb, response=y, lang="es")             # español

# o como argumento de las funciones del experimento:
ed.fit_linear_model(bb, y, report="report/")     # ruta en fit.report_path
ed.evaluate(bb, report=True)                     # ruta en ev.report_path
```

Requiere el extra `[report]` (matplotlib).

## Ver también

- Teoría: [Métricas de evaluación](evaluation-metrics.md)
- API: [`report`, `report_summary`](../api/analysis-report.md)
