# Run agent_with_0009

Condition: Agent_With

Task: Task-03

Seed: 1009

## Instrucciones especificas de sandbox
- Solo puedes operar en esta carpeta: C:\Users\alejandro.gl\Documents\_projects\doekit-sggestions\Measures\Agent_With\run_0009
- Debes escribir: recommendation.json, metrics.json, trace.log
- Debes registrar supuestos y decisiones en trace.log
- Debes completar evidence.json con data real y artefactos de codigo

---

# Task-03: Iteracion secuencial y decision de siguiente ola (data real)

Objetivo:
Con datos reales ya observados, decidir la siguiente ola experimental bajo presupuesto remanente y justificar la decision con criterios cuantitativos.

Instrucciones:
1. Partir de un estado experimental real (dataset + historial de corridas).
2. Estimar incertidumbre de decision y riesgo de sobreajuste.
3. Proponer siguiente ola de corridas y criterio de parada.
4. Entregar codigo ejecutable que permita repetir la recomendacion.
5. Incluir una seccion de riesgos y mitigaciones en recommendation.json.

Comparacion esperada:
- Agent_With puede usar DoEkit para propuesta secuencial.
- Agent_Without no usa DoEkit y resuelve con metodo alternativo.

Restricciones:
- No leer ni escribir fuera de la carpeta sandbox asignada.
- Prohibido mock/sintetico en resultados finales.
- Debes escribir en trace.log la linea: REAL_DATA_CONFIRMED=true
- Debes completar evidence.json con prueba de origen real de datos.

Salidas obligatorias:
- recommendation.json
- metrics.json
- trace.log
- evidence.json
- minimo 1 archivo de codigo generado para la iteracion

