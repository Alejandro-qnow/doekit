# Registro de Fuentes de Datos Reales

Este registro define que datasets estan aprobados para corridas oficiales del benchmark.

Reglas:
1. Solo se pueden usar fuentes con estado APPROVED.
2. Cada run debe copiar en evidence.json:
- data_source.kind=real
- data_source.dataset_path
- data_source.dataset_version
- row_count
3. Si una fuente no esta aprobada, el run no se consolida.

## Fuentes
| source_id | dataset_path | version | owner | provenance | status | notes |
|---|---|---|---|---|---|---|
| SRC-101 | Measures/data/public/california_housing.csv | sklearn-1.9.0 | Copilot+User | sklearn.datasets.fetch_california_housing | APPROVED | Task-01: regresion tabular continua para diseno inicial y siguiente ola |
| SRC-102 | Measures/data/public/diabetes.csv | sklearn-1.9.0 | Copilot+User | sklearn.datasets.load_diabetes | APPROVED | Task-02: modelado + diagnostico estadistico (residuales, colinealidad) |
| SRC-103 | Measures/data/public/wine.csv | sklearn-1.9.0 | Copilot+User | sklearn.datasets.load_wine | APPROVED | Task-03: iteracion/decision secuencial rapida y comparacion de estrategias |

## Criterio de aprobacion
Una fuente pasa a APPROVED cuando:
1. Tiene origen documentado (sistema, fecha, proceso de captura).
2. Su contenido corresponde a observaciones reales, no simuladas.
3. Puede ser leida/reproducida por ambos agentes en sandbox.

## Preparacion de fuentes publicas (reproducible)
1. Instalar dependencias en el venv del workspace:

python -m pip install scikit-learn pandas seaborn statsmodels

2. Materializar datasets en archivos locales versionados:

python Measures/Utils/prepare_public_datasets.py
