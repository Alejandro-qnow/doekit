# Protocolo Meta-Diseno V2 (industria + research) con DoEkit

Este protocolo extiende el piloto inicial y lo convierte en un benchmark robusto para tomar decisiones de adopcion. El objetivo ya no es solo comparar medias, sino estimar el efecto de DoEkit con control de dificultad de tarea, heterogeneidad de contexto y riesgo de sobreajuste de metricas.

## 1) Como se hace en industria y en research

En industria (A/B y decision systems), la practica madura combina cuatro ideas: predefinir una metrica primaria para evitar p-hacking, bloquear por segmentos que alteran el rendimiento, randomizar orden para mitigar drift operativo y usar analisis secuencial con criterios de parada claros. En research aplicado, el estandar agrega diseno factorial para identificar interacciones, modelos mixtos para separar variacion por tarea/seed/fuente y analisis de sensibilidad para validar que el efecto no dependa de un unico supuesto.

Por eso, para este caso DoEkit vs no DoEkit, la version robusta no es un unico test global. Debe ser un experimento bloqueado, pareado, con meta-factores y con trazabilidad total de data real.

## 2) Hipotesis V2

H1 primaria: Agent_With mejora ImpactScore frente a Agent_Without, ajustando por dificultad de tarea.

H2 de interaccion: la ventaja de Agent_With aumenta en estratos de mayor dificultad.

H3 de costo operativo: Agent_With no debe degradar de forma inaceptable el tiempo util de decision (no solo tiempo de computo bruto).

## 3) Unidad experimental y bloqueo

La unidad experimental sigue siendo una corrida de agente en sandbox. La novedad es que ahora cada corrida vive dentro de un bloque por:

- tarea,
- fuente real aprobada,
- estrato de dificultad,
- celda de meta-diseno.

El bloqueo evita que una condicion gane solo por tocar tareas mas faciles.

## 4) Dificultad de tarea (medida, no supuesta)

Se define un Difficulty Index en [0, 100] por tarea, con componentes:

- tamano de dataset,
- dimensionalidad efectiva,
- colinealidad media,
- proxy de ruido/no-linealidad (1 - R2 lineal).

Luego se asignan estratos low/medium/high por cuantiles y se balancea el plan para que ambas condiciones recorran la misma dificultad.

## 5) Meta-factores experimentales

Se modelan factores operativos, con niveles codificados -1 / +1:

- F1: acceso DoEkit (Agent_Without / Agent_With)
- F2: prompt strictness (medium / high)
- F3: context visibility (task_only / task_plus_history)
- F4: timeout budget (short / long)

Importante: F1 se ejecuta en pares para comparacion justa dentro de la misma celda F2-F4, misma tarea y misma seed.

## 6) Uso de DoEkit para el meta-diseno

Para generar celdas experimentales, usar DoEkit en los factores F2-F4 con presupuesto base de 8 corridas y modelo lineal. Esa base se cruza con F1 en forma pareada para obtener comparaciones directas With/Without por celda.

Script operativo:

python Measures/Utils/design_meta_experiment_with_doekit.py --target-total-runs 120

Salidas:

- Measures/meta_experiment_plan.csv
- Measures/meta_experiment_plan_summary.json

## 7) Tamano muestral recomendado

Regla operativa:

- inicio robusto: 96-120 corridas totales,
- expansion: 144-192 corridas si el efecto cae al ajustar interacciones.

Razon: queremos potencia util para detectar efectos moderados y, sobre todo, para identificar interaccion Condition x Difficulty.

## 8) Analisis recomendado

Primario:

ImpactScore ~ Condition + Difficulty + Condition:Difficulty + (1|Task) + (1|Seed) + (1|Source)

Secundarios:

- log(total_time_sec)
- d_efficiency
- mean_power
- predicted_gain
- uncertainty_index

Para conteos de riesgo, usar Poisson/NegBin y evaluar exceso de ceros.

## 9) Gobernanza y validez

Se mantiene la politica estricta de data real y trazabilidad existente. Ninguna corrida entra a consolidado final sin evidencia valida y source_id aprobado. Tambien se predefine:

- endpoint primario,
- reglas de parada,
- analisis de sensibilidad,
- criterio de adopcion.

## 10) Criterio de adopcion V2

Adoptar DoEkit por defecto solo si se cumplen las tres condiciones:

1. efecto positivo en endpoint primario con estabilidad por estrato,
2. mejora o neutralidad en riesgo,
3. costo temporal aceptable para el contexto de negocio.

Este criterio obliga a una decision realista: calidad de decision contra latencia, con evidencia estadistica y operativa.
