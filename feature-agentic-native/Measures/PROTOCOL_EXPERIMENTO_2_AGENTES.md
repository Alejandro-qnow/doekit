# Protocolo estricto: medicion Con y Sin DoEkit (2 subagentes, data real)

## Objetivo del producto
Medir impacto de DoEkit en tareas reales de investigacion guiada por diseno de experimentos.
Comparar dos condiciones:
- Agent_With: subagente con acceso a DoEkit (API/modulos/MCP permitido).
- Agent_Without: subagente sin acceso a DoEkit (flujo manual, estadistica basica, heuristicas).

## Hipotesis
- H1 (eficiencia): Agent_With reduce tiempo total a recomendacion final al menos 20%.
- H2 (calidad tecnica): Agent_With mejora calidad de decision (score tecnico compuesto) al menos 10%.
- H3 (riesgo): Agent_With reduce errores de planteamiento (diseno/modelo) al menos 25%.

## Unidad experimental
Cada subagente ejecutando la misma tarea del par en sandbox aislado.

## Politica de validez
- Solo data real en resultados finales.
- Prohibido mock/sintetico para metricas consolidadas.
- Cada run debe incluir evidencia auditable de origen de datos y codigo generado.
- Cada corrida debe usar una fuente APPROVED en REAL_DATA_SOURCES.md.

## Control de sesgo
- Misma tarea base.
- Mismo presupuesto maximo de corridas.
- Mismo dataset de entrada y mismo formato de salida.
- Worktree limpio en cada ejecucion.
- Prompt unico por condicion.
- Sin internet durante corrida (si se desea maxima comparabilidad).

## Sandbox seguro por subagente
Reglas recomendadas:
1. El subagente solo puede leer/escribir dentro de su carpeta asignada:
   - Measures/Agent_With
   - Measures/Agent_Without
2. Ejecucion en entorno aislado (venv fijo y dependencias congeladas).
3. Limpiar estado antes de cada run:
   - eliminar artefactos previos del run
   - reiniciar terminal/sesion
4. Prohibir cambios fuera de carpeta objetivo.
5. Registrar todo en log estructurado JSON.

## Catalogo de tareas (minimo 3)
Las corridas se asignan por par de forma balanceada y ciclica:
- Task-01: diseno inicial de optimizacion con presupuesto fijo.
- Task-02: modelado y diagnostico estadistico con datos reales.
- Task-03: iteracion secuencial y decision de siguiente ola.

Todas las tareas requieren:
- razonamiento tecnico,
- generacion de codigo reproducible,
- evidencia de data real.

## Datasets publicos aprobados y por que
Dependencias requeridas:
- scikit-learn
- pandas
- seaborn
- statsmodels

Datasets aprobados (materializados localmente en Measures/data/public):
1. SRC-101 / california_housing.csv (Task-01)
- Por que: dataset tabular continuo, suficiente tamano para evaluar diseno inicial, calidad de diseno y propuesta de siguiente ola.
2. SRC-102 / diabetes.csv (Task-02)
- Por que: dataset estandar para regresion y diagnostico estadistico, ideal para comparar calidad de modelado con y sin DoEkit.
3. SRC-103 / wine.csv (Task-03)
- Por que: dataset compacto y multivariable para iteracion rapida, decision secuencial y trazabilidad de estrategias.

Regla de asignacion por tarea:
- Task-01 usa SRC-101
- Task-02 usa SRC-102
- Task-03 usa SRC-103

### Salidas obligatorias por run
- recommendation.json
- metrics.json
- trace.log (debe incluir REAL_DATA_CONFIRMED=true)
- evidence.json
- al menos 1 artefacto de codigo generado (.py/.ipynb/.md con codigo ejecutable)

## Metricas
### A. Eficiencia
- total_time_sec
- time_to_first_valid_plan_sec
- iterations_count

### B. Calidad tecnica
- d_efficiency
- mean_power
- predicted_gain
- uncertainty_index
- budget_used_ratio

### C. Riesgo/calidad de decision
- invalid_assumptions_count
- budget_violations_count
- model_mismatch_flags
- decision_reversal_count

### D. Resultado UX-operativa (proxy)
- output_completeness_score (0-1)
- format_compliance_score (0-1)

### E. Evidencia y auditabilidad
- real_data_compliance (binaria: 1 si cumple evidencia real)
- code_artifacts_count
- dataset_traceability_score (0-1)

## Score compuesto
ImpactScore =
0.35 * CalidadTecnica +
0.30 * Eficiencia +
0.20 * Riesgo +
0.15 * UX

Normalizar cada bloque en [0, 100].

## Diseno del experimento (Fase 1: piloto ordenado)
- N inicial: 10 repeticiones por condicion (20 runs total).
- Seed por run: 1001..1020.
- Orden por pares balanceados With/Without con misma tarea y seed.
- Asignacion de tarea por indice: Task-01, Task-02, Task-03, y repetir.

## Diseno del experimento (Fase 2: DOE sobre el propio protocolo)
Una vez validado el piloto, aplicar DOE para optimizar el proceso de agentes.
Factores sugeridos (2 niveles):
- F1: acceso DoEkit (No/Si)
- F2: prompt estricto (medio/alto)
- F3: contexto visible (solo tarea/tarea+historial)
- F4: timeout (corto/largo)

Respuesta objetivo:
- ImpactScore
- varianza de ImpactScore

Sugerencia: iniciar con diseno fraccional 2^(4-1) para minimizar corridas.

## Criterio de exito para decidir adopcion
Adoptar DoEkit por defecto si:
1. delta tiempo <= -20%
2. delta calidad tecnica >= +10%
3. delta errores >= -25%
4. sin degradacion de cumplimiento de formato
5. cumplimiento estricto de data real en 100% de runs consolidados

## Gobernanza de resultados
- Consolidar todos los runs en un CSV unico.
- Generar reporte comparativo con media, mediana, p90 y dispersion.
- Si la muestra lo permite, test de hipotesis no parametrico (Mann-Whitney) por metrica clave.
- Ejecutar auditoria por par antes de consolidar resultados finales.
