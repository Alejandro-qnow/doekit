# Sandbox Agent_With

Este sandbox es exclusivo para ejecuciones del subagente con DoEkit.

Reglas:
1. No leer/escribir fuera de esta carpeta.
2. Usar el prompt definido en ../TASK_01_PROMPT_UNICO.md.
3. Guardar outputs por corrida en subcarpetas run_XXXX.

Por cada run crear:
- recommendation.json
- metrics.json
- trace.log

Nota:
- Se permite acceso a funciones DoEkit y/o MCP configurado.
