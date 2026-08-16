# Runner de subagentes (Con vs Sin DoEkit)

Script principal:
- run_experiment.py

## Flujo rapido
Desde la carpeta Measures:

1. Preparar todos los runs y artefactos:

python Utils/run_experiment.py prepare

2. Ver estado:

python Utils/run_experiment.py status

2.1 Seleccionar siguiente par balanceado (With/Without):

python Utils/run_experiment.py next-pair

2.2 Seleccionar y marcar ambos como running:

python Utils/run_experiment.py next-pair --mark-running

3. Iniciar un run (marca running y te muestra el prompt local):

python Utils/run_experiment.py start agent_with_0001

4. Ejecutar el subagente en ese sandbox usando el archivo:

Agent_With/run_0001/task_prompt.md

5. Cuando termine y tengas metrics.json completo, consolidar:

python Utils/run_experiment.py complete agent_with_0001

6. Reporte agregado:

python Utils/run_experiment.py report

7. Auditoria de integridad/sesgo por par:

python Utils/audit_runs.py --pair-index 1

8. Aprobar una fuente real en el registro:

python Utils/approve_source.py SRC-00X --dataset-path <ruta> --version <version> --owner <owner> --provenance <origen> --notes <nota>

9. Preparar datasets publicos recomendados:

python Utils/prepare_public_datasets.py

10. Generar plan meta-experimental robusto (bloqueado por dificultad) con DoEkit:

python Utils/design_meta_experiment_with_doekit.py --target-total-runs 120

11. Preparar y ejecutar el meta-experimento desde el plan:

python Utils/run_meta_experiment.py prepare
python Utils/run_meta_experiment.py status
python Utils/run_meta_experiment.py next-pair --mark-running

12. Generar reporte comprensivo narrativo desde metricas:

python Utils/generate_comprehensive_report_from_stats.py --input-csv Measures/metrics_template.csv
python Utils/generate_comprehensive_report_from_stats.py --input-csv Measures/metrics_meta_template.csv

## Convencion de trabajo con Copilot/subagentes
- Cada run vive en su carpeta run_XXXX.
- El prompt del run ya incluye condicion, seed y ruta de sandbox.
- El subagente solo debe escribir en su carpeta de run.
- Archivos esperados por run:
  - recommendation.json
  - metrics.json
  - trace.log

## Nota
El runner no invoca subagentes automaticamente. Orquesta el experimento, estados y consolidacion para que podamos ejecutar los subagentes contigo de forma controlada y auditable.

Advertencia de validez:
- Para benchmark final "justo" (Con vs Sin), usar solo ejecuciones reales de subagentes en ambos lados.
- El comando complete aplica validacion estricta de data real; solo usar --allow-nonreal en depuracion, nunca en medicion final.
