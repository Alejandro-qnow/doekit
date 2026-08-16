# Sandbox Agent_Without

Este sandbox es exclusivo para ejecuciones del subagente sin DoEkit.

Reglas:
1. No leer/escribir fuera de esta carpeta.
2. Usar el prompt definido en ../TASK_01_PROMPT_UNICO.md.
3. Guardar outputs por corrida en subcarpetas run_XXXX.

Por cada run crear:
- recommendation.json
- metrics.json
- trace.log

Nota:
- No se permite usar DoEkit ni MCP de DoEkit en esta condicion.
