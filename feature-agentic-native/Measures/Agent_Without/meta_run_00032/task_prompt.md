# Meta Run meta_run_00032

Pair: PAIR_0004

Condition: Agent_Without

Task: Task-01

Seed: 3004

## Meta-factores operativos (debes respetarlos)
- prompt_strictness_level: medium
- context_visibility_level: task_only
- timeout_budget_level: short
- difficulty_stratum: high (index=48.3813)
- source_id: SRC-101
- dataset_path: Measures/data/public/california_housing.csv

## Instrucciones especificas de sandbox
- Solo puedes operar en esta carpeta: C:\Users\alejandro.gl\Documents\_projects\doekit-sggestions\Measures\Agent_Without\meta_run_00032
- Debes escribir: recommendation.json, metrics.json, trace.log
- Debes registrar supuestos y decisiones en trace.log
- Debes completar evidence.json con data real y artefactos de codigo
- Debes respetar la trazabilidad exacta de source_id y dataset_path

---

# Task-01: Diseno inicial de optimizacion (data real)

Objetivo:
Resolver una tarea de diseno de experimentos con datos reales para maximizacion de respuesta con 4 factores y presupuesto total de 24 corridas.

Instrucciones:
1. Entregar un plan inicial de diseno experimental.
2. Evaluar calidad del diseno propuesto.
3. Proponer siguiente ola de corridas.
4. Definir criterio de parada.
5. Respetar estrictamente el presupuesto.
6. Generar codigo ejecutable para reproducir el analisis.
7. Entregar salidas en formato JSON valido.

Entradas:
- factores: X1, X2, X3, X4
- rango por factor: [-1, 1]
- presupuesto total: 24
- objetivo: maximizar respuesta
- modelo objetivo: cuadratico

Restricciones:
- No leer ni escribir fuera de la carpeta sandbox asignada.
- Registrar decisiones clave en trace.log.
- Prohibido usar datos mock/sinteticos para resultados finales.
- Debes escribir en trace.log la linea: REAL_DATA_CONFIRMED=true
- Debes completar evidence.json indicando dataset real y artefactos de codigo generados.

Salidas obligatorias:
- recommendation.json
- metrics.json
- trace.log
- evidence.json
- al menos 1 archivo de codigo (.py/.ipynb/.md con snippets ejecutables) usado en la solucion

