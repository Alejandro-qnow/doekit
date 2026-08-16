# Meta Run meta_run_00055

Pair: PAIR_0009

Condition: Agent_With

Task: Task-02

Seed: 3009

## Meta-factores operativos (debes respetarlos)
- prompt_strictness_level: medium
- context_visibility_level: task_only
- timeout_budget_level: long
- difficulty_stratum: medium (index=46.8639)
- source_id: SRC-102
- dataset_path: Measures/data/public/diabetes.csv

## Instrucciones especificas de sandbox
- Solo puedes operar en esta carpeta: C:\Users\alejandro.gl\Documents\_projects\doekit-sggestions\Measures\Agent_With\meta_run_00055
- Debes escribir: recommendation.json, metrics.json, trace.log
- Debes registrar supuestos y decisiones en trace.log
- Debes completar evidence.json con data real y artefactos de codigo
- Debes respetar la trazabilidad exacta de source_id y dataset_path

---

# Task-02: Modelado y diagnostico de modelo (data real)

Objetivo:
Usar datos reales para ajustar y diagnosticar un modelo de respuesta asociado a un diseno experimental, proponiendo mejoras de diseno y codigo reproducible.

Instrucciones:
1. Cargar un dataset real disponible en el workspace o ruta declarada.
2. Ajustar un modelo acorde al objetivo experimental (al menos lineal con terminos relevantes).
3. Diagnosticar supuestos (residuales, colinealidad, falta de ajuste cuando aplique).
4. Proponer cambios de diseno para mejorar potencia/prediccion.
5. Entregar codigo ejecutable reproducible.

Comparacion esperada:
- Agent_With puede usar DoEkit para apoyo de analisis/diseno.
- Agent_Without no usa DoEkit y resuelve con herramientas generales.

Restricciones:
- No leer ni escribir fuera de la carpeta sandbox asignada.
- Prohibido mock/sintetico en resultados finales.
- Debes escribir en trace.log la linea: REAL_DATA_CONFIRMED=true
- Debes completar evidence.json con:
  - data_source.kind=real
  - data_source.dataset_path
  - mock_data_used=false
  - code_artifacts_generated

Salidas obligatorias:
- recommendation.json
- metrics.json
- trace.log
- evidence.json
- minimo 1 archivo de codigo generado para el modelado

