# Reporte comprensivo desde estadisticas

Fecha de generacion: 2026-08-14T13:43:37.101029+00:00

Experimento analizado: piloto_real_data_con_vs_sin_doekit

Fuente de metricas: C:\Users\alejandro.gl\Documents\_projects\doekit-sggestions\Measures\metrics_template.csv

Este reporte transforma el consolidado estadistico en una lectura ejecutiva y tecnica para tomar direccion. Se mantiene una postura realista: no solo se observa si hay mejora de score, sino tambien el costo operativo y la estabilidad por tarea y, cuando aplica, por estrato de dificultad.

Se analizaron 20 corridas validas de las condiciones Agent_With y Agent_Without. La muestra todavia es limitada, por lo que la lectura debe tratarse como senal inicial.

En impacto global, Agent_With presenta media 84.58 frente a 82.10 de Agent_Without, con delta 2.48 (3.02%), p=0.031 y d=0.83. En tiempo total, Agent_With marca media 0.1438 vs 0.0272, con delta 0.1167 (429.37%).

En calidad tecnica, mean_power: delta 0.0736 (10.00%), lectura=evidencia estadistica favorable; predicted_gain: delta 0.1000 (23.85%), lectura=evidencia estadistica favorable; uncertainty_index: delta -0.0500 (-20.12%), lectura=evidencia estadistica favorable; d_efficiency: delta -0.0540 (-10.70%), lectura=evidencia estadistica desfavorable.

Por tarea, Task-01 muestra delta de ImpactScore 2.69 (3.36%); Task-02 muestra delta de ImpactScore 2.66 (3.31%); Task-03 muestra delta de ImpactScore 2.02 (2.34%).



En riesgo operativo, invalid_assumptions_count delta 0.000; budget_violations_count delta 0.000; model_mismatch_flags delta 0.000; decision_reversal_count delta 0.000.

La lectura integrada sugiere que la decision de adopcion no debe basarse en una sola metrica. Si el impacto compuesto mejora pero el costo temporal crece, la decision correcta depende del contexto: donde la calidad de decision y la trazabilidad pesan mas que la latencia, el despliegue de DoEkit tiene sentido; donde la latencia domina, la prioridad pasa por optimizar pipeline y configuracion experimental antes de escalar.

## Direcciones sugeridas
1. Escalar la experimentacion priorizando estabilidad del ImpactScore, manteniendo pares balanceados y control de trazabilidad real.
2. Optimizar latencia del flujo con DoEkit separando tiempo de computo y tiempo de decision util, para no confundir costo tecnico con valor analitico.
3. Recalibrar la estrategia de diseno (factors, budget, model_order) por tarea, especialmente donde d_efficiency cae en escenarios de mayor dificultad.
4. Mantener DoEkit en etapas de decision secuencial, donde se observa mejor ganancia esperada y menor incertidumbre residual.

## Nota metodologica
Se reportan estadisticos descriptivos, contraste Mann-Whitney (cuando SciPy esta disponible) y tamano de efecto Cohen d para metricas continuas clave. Esta lectura no reemplaza un modelo mixto completo, pero sirve como tablero de decision recurrente corrida tras corrida.
