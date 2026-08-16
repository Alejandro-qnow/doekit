# Playbook operativo del protocolo (estricto data real)

Este playbook define como seguir el protocolo en ciclos reproducibles con subagentes.

## Setup inicial (una vez por workspace)
1. Instalar librerias requeridas:

python -m pip install scikit-learn pandas seaborn statsmodels

2. Preparar datasets publicos en archivos locales:

python Utils/prepare_public_datasets.py

3. Confirmar registro de fuentes APPROVED en ../REAL_DATA_SOURCES.md:
- Task-01 -> SRC-101 (california_housing.csv)
- Task-02 -> SRC-102 (diabetes.csv)
- Task-03 -> SRC-103 (wine.csv)

## Ciclo estandar (1 par)
0. Seleccionar fuente de datos APPROVED en [Measures/REAL_DATA_SOURCES.md](../REAL_DATA_SOURCES.md).

Si una fuente aun no esta aprobada, aprobarla con:

python Utils/approve_source.py SRC-00X --dataset-path <ruta> --version <version> --owner <owner> --provenance <origen> --notes <nota>

1. Seleccionar par balanceado:

python Utils/run_experiment.py next-pair --mark-running

2. Ejecutar subagente de condicion With usando el prompt del run mostrado.
3. Ejecutar subagente de condicion Without usando el prompt del run mostrado.
4. Verificar que ambos generaron:
- recommendation.json
- metrics.json
- trace.log
- evidence.json
- al menos 1 artefacto de codigo por run

5. Auditoria obligatoria del par:

python Utils/audit_runs.py --pair-index X

Condicion para consolidar:
- sin synthetic_flags
- trace con REAL_DATA_CONFIRMED=true
- evidence con data_source.kind=real y mock_data_used=false
- evidence con data_source.source_id APPROVED y dataset_path consistente

6. Consolidar ambos runs:

python Utils/run_experiment.py complete agent_with_000X
python Utils/run_experiment.py complete agent_without_000X

7. Revisar estado y reporte:

python Utils/run_experiment.py status
python Utils/run_experiment.py report

## Cadencia recomendada
- Piloto: 3 pares (6 runs), cubriendo Task-01/Task-02/Task-03 al menos una vez por condicion.
- Si no hay bloqueos: completar 10 pares (20 runs).

## Gate de calidad al run 6
Exigir para continuar:
1. 100% de runs con outputs completos.
2. 0 violaciones de presupuesto.
3. >= 90% de cumplimiento de formato.
4. 100% de compliance de data real (sin flags de synthetic/mock).

## Gate de decision al run 20
Comparar Agent_With vs Agent_Without:
1. Tiempo total medio
2. ImpactScore medio
3. Errores de riesgo agregados

Decision:
- Si With supera umbrales definidos en experiment_config.json, recomendar adopcion operativa.

## Auditoria
- Conservar run_schedule.csv como bitacora de estados.
- Conservar trace.log por run para trazabilidad de decisiones.
- No modificar manualmente metrics_template.csv fuera de la consolidacion automatica, salvo correccion documentada.

## Limpieza y validez
- No usar scripts de ejecucion sintetica para benchmark final.
- Si hubo corridas de bootstrap/simulacion, excluirlas de metrics_template.csv antes de analisis final.
