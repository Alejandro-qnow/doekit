# Figuras Mermaid para IMRaD

Este documento propone figuras por etapas metodologicas y una figura global final del flujo completo de informacion.

## Figura 1. Etapa conceptual: de salida numerica a decision para agentes

```mermaid
flowchart LR
    A[DoE numerico clasico\nmetricas: D-eff, power, gain] --> B[Interpretacion semantica\nmeaning + reasoning + warnings]
    B --> C[Politica de decision\ncontinue/stop/refine]
    C --> D[Accion experimental\nnueva ola o cierre]

    E[Contexto operativo\npresupuesto, riesgo, tarea] --> C
    F[Incertidumbre estimada\npenalizacion/confianza] --> C
```

Mensaje metodologico:
- El valor de la extension no es reemplazar DoE, sino traducir evidencia cuantitativa en decisiones operables por agentes.

## Figura 2. Etapa de arquitectura modular de la libreria

```mermaid
flowchart TB
    subgraph S1[Semantic]
      S1A[Interpreters]
      S1B[Templates]
      S1C[Prompt builders]
    end

    subgraph S2[Decision]
      S2A[Policies]
      S2B[Scoring]
      S2C[Uncertainty]
    end

    subgraph S3[Monitoring]
      S3A[Convergence]
      S3B[Diagnostics]
      S3C[Events]
    end

    subgraph S4[Memory]
      S4A[Store]
      S4B[Transfer]
      S4C[Historical recommendations]
    end

    subgraph S5[Integrations]
      S5A[Bayesian optimization adapter]
      S5B[External tools]
    end

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S2
    S2 --> S5
```

Mensaje metodologico:
- La arquitectura separa responsabilidades y permite autonomia progresiva sin acoplamiento rigido.

## Figura 3. Etapa de diseno empirico (With vs Without)

```mermaid
flowchart LR
    P0[Plan experimental\npares balanceados] --> P1[Asignacion de tarea\nTask-01/02/03 ciclico]
    P1 --> P2[Seed compartida por par]
    P2 --> P3A[Condicion A\nAgent_With]
    P2 --> P3B[Condicion B\nAgent_Without]

    P3A --> P4A[Sandbox aislado\nAgent_With]
    P3B --> P4B[Sandbox aislado\nAgent_Without]

    P4A --> P5[Generacion de artefactos\nrecommendation, metrics, trace, evidence, codigo]
    P4B --> P5
```

Mensaje metodologico:
- La comparabilidad se protege por pareo, misma tarea/seed y reglas simetricas de ejecucion.

## Figura 4. Etapa de auditoria y consolidacion de evidencia real

```mermaid
flowchart TD
    A1[Run finalizado] --> A2{Cumple data real?}
    A2 -->|Si| A3[Valida source_id aprobado]
    A2 -->|No| A8[Excluir del consolidado]

    A3 --> A4{REAL_DATA_CONFIRMED=true?}
    A4 -->|Si| A5{Artefacto de codigo presente?}
    A4 -->|No| A8

    A5 -->|Si| A6[Incluir en CSV consolidado]
    A5 -->|No| A8

    A6 --> A7[Analisis estadistico\ndeltas, p-value, effect size]
```

Mensaje metodologico:
- No hay inferencia final sin trazabilidad verificable de evidencia.

## Figura 5. Etapa de lectura de resultados y criterio de adopcion

```mermaid
flowchart LR
    R1[ImpactScore global] --> R4{Trade-off aceptable?}
    R2[Metricas tecnicas\npower, gain, uncertainty, d_eff] --> R4
    R3[Latencia operativa\ntotal_time_sec] --> R4

    R4 -->|Si| R5[Adopcion condicionada\npor contexto]
    R4 -->|No| R6[Optimizar pipeline\ny recalibrar diseno]

    R5 --> R7[Escalado controlado\ncon monitoreo]
    R6 --> R8[Nueva iteracion experimental]
```

Mensaje metodologico:
- La decision de adopcion es multiobjetivo, no monotona en una sola metrica.

## Figura 6. Flujo global completo de informacion (end-to-end)

```mermaid
flowchart TD
    G0[Pregunta de investigacion\nDoE para agentes] --> G1[Diseno metodologico\nprincipios + arquitectura]
    G1 --> G2[Plan empirico\nWith vs Without, tareas, seeds]
    G2 --> G3[Ejecucion en sandbox\npares comparables]

    G3 --> G4[Artefactos por run\njson + logs + codigo]
    G4 --> G5[Auditoria de evidencia real]
    G5 -->|Aprobado| G6[Consolidacion de metricas]
    G5 -->|Rechazado| G11[Depuracion y rerun]

    G6 --> G7[Analisis estadistico\nglobal, por tarea, por dificultad]
    G7 --> G8[Interpretacion semantica\ntrade-offs y riesgos]
    G8 --> G9[Decision de adopcion\ncondicionada al contexto]
    G9 --> G10[Retroalimentacion a la libreria\npoliticas, scoring, latencia]
    G10 --> G2
```

Mensaje metodologico:
- El sistema completo funciona como ciclo de evidencia: diseno, ejecucion, auditoria, inferencia, decision y mejora continua.

## Sugerencia de ubicacion en IMRaD
1. Introduccion: Figura 1.
2. Metodos: Figuras 2, 3 y 4.
3. Resultados/Discusion: Figura 5.
4. Cierre metodologico o apendice: Figura 6.
