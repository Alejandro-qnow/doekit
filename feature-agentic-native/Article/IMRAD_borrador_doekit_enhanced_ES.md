# Borrador IMRaD v0: doekit-enhanced para experimentacion con agentes

## Titulo propuesto
De salidas DoE numericas a decisiones operativas para agentes: propuesta metodologica y evaluacion empirica de doekit-enhanced

## Resumen
Este trabajo presenta doekit-enhanced, una capa de extension sobre doekit para adaptar Design of Experiments (DoE) a flujos autonomos con agentes de IA. La hipotesis central es que las salidas numericas son necesarias pero insuficientes cuando un agente debe justificar decisiones, cuantificar incertidumbre y mantener trazabilidad auditable entre olas experimentales. Proponemos una arquitectura semantic-first con cinco componentes: interpretacion semantica, motor de decision por politicas, manejo de incertidumbre, monitoreo de convergencia y memoria con transferencia. En la evaluacion empirica, comparamos Agent_With y Agent_Without bajo protocolo pareado y datos reales. En el piloto (20 corridas), Agent_With mejora ImpactScore en +3.02% (84.58 vs 82.10, p=0.031, d=0.83), con aumento de tiempo total (+429%). En la extension meta-diseno (126 corridas), la mejora global se mantiene (+3.01%, d=0.86), con ganancias consistentes en mean_power (+10.00%), predicted_gain (+23.67%) y uncertainty_index (-20.18%), mientras d_efficiency cae (-10.93%). El resultado principal es un trade-off calidad-latencia: mejora la robustez de decision y la trazabilidad, pero exige optimizacion operativa y calibracion por tarea.

Palabras clave: design of experiments, agentes autonomos, experimentacion secuencial, incertidumbre, reproducibilidad

## 1. Introduccion
### 1.1 Problema
Las librerias DoE clasicas fueron pensadas para expertos humanos que traducen diagnosticos numericos a decisiones. En sistemas con agentes, hace falta una capa adicional: razonamiento explicito, reusable y auditable por maquina.

### 1.2 Brecha
El tooling DoE actual suele optimizar metricas de diseno, pero subespecifica tres capacidades criticas para autonomia:
1. interpretacion semantica de evidencia cuantitativa;
2. politicas de decision con trade-offs explicitos;
3. memoria experimental y convergencia para operacion multi-wave.

### 1.3 Contribuciones
1. Marco metodologico de DoE orientado a agentes (semantic-first + decisiones gobernadas por politicas).
2. Arquitectura modular de extension (doekit-enhanced) compatible de forma aditiva.
3. Protocolo empirico con datos reales y comparacion With/Without.
4. Lectura critica del trade-off entre calidad de decision y latencia.

### 1.4 Preguntas de investigacion
RQ1: El uso de doekit-enhanced mejora la calidad global de decision (ImpactScore) frente al flujo control?

RQ2: Las mejoras se sostienen por tarea y por estrato de dificultad?

RQ3: Que trade-offs operativos aparecen entre calidad, incertidumbre y latencia?

## 2. Metodos
### 2.1 Espiritu de la libreria
La propuesta sigue cinco principios:
1. Compatibilidad aditiva: no romper uso base de doekit.
2. Semantic-first: toda salida critica puede elevarse a artefacto interpretado.
3. Politicas explicitas: stop/continue/refine con criterios trazables.
4. Auditabilidad: evidencia minima verificable por corrida.
5. Modularidad: semantic, decision, monitoring, memory e integrations operan desacoplados o en pipeline.

### 2.2 Descomposicion arquitectonica
1. semantic: interpreters, builders y templates para conversion numerico-semantica.
2. decision: politicas y scoring multiobjetivo.
3. monitoring: convergencia y diagnosticos.
4. memory: store de experimentos y transferencia de priors.
5. integrations: adaptadores de optimizacion externa.

### 2.3 Diseno experimental
#### 2.3.1 Condiciones
1. Agent_With: acceso a capacidades DoEkit y capa mejorada.
2. Agent_Without: flujo estadistico/manual sin DoEkit.

#### 2.3.2 Tareas y datos
Tareas ciclicas:
1. Task-01 (diseno inicial de optimizacion)
2. Task-02 (modelado y diagnostico)
3. Task-03 (iteracion secuencial)

Fuentes reales aprobadas por tarea: california_housing, diabetes y wine.

#### 2.3.3 Controles de validez
1. pareo por tarea y seed;
2. mismas restricciones de presupuesto/modelo;
3. sandbox por condicion;
4. artefactos obligatorios por run: recommendation.json, metrics.json, trace.log, evidence.json y codigo ejecutable.

#### 2.3.4 Variables y endpoint
Endpoint primario compuesto:
ImpactScore = 0.35*CalidadTecnica + 0.30*Eficiencia + 0.20*Riesgo + 0.15*UX

Secundarias: total_time_sec, d_efficiency, mean_power, predicted_gain, uncertainty_index y contadores de riesgo.

### 2.4 Estrategia en dos fases
Fase 1 (piloto): 20 corridas (10 por condicion).

Fase 2 (meta-diseno V2): 126 corridas validas balanceadas por condicion, tarea y estrato de dificultad.

### 2.5 Estrategia de analisis
1. direccion y magnitud del efecto (delta y %).
2. contraste no parametrico cuando aplica (Mann-Whitney).
3. tamano de efecto (Cohen d).
4. estabilidad por tarea y dificultad.

## 3. Resultados
### 3.1 Piloto (20 corridas)
1. ImpactScore: 84.58 (With) vs 82.10 (Without), delta +2.48 (+3.02%), p=0.031, d=0.83.
2. Tiempo total: 0.1438 vs 0.0272, delta +429.37%.

Metricas tecnicas:
1. mean_power: +10.00%.
2. predicted_gain: +23.85%.
3. uncertainty_index: -20.12%.
4. d_efficiency: -10.70%.

### 3.2 Meta-diseno (126 corridas)
Balance:
1. 63 corridas por condicion.
2. 42 corridas por tarea.
3. 42 corridas por estrato (high/medium/low).

Resultado global:
1. ImpactScore: 84.62 (With) vs 82.15 (Without), delta +2.47 (+3.01%), p<0.001, d=0.86.
2. Tiempo total: 0.1767 vs 0.0313, delta +465.13%.

Metricas tecnicas:
1. mean_power: +10.00%.
2. predicted_gain: +23.67%.
3. uncertainty_index: -20.18%.
4. d_efficiency: -10.93%.

Patron:
1. la mejora de ImpactScore se mantiene en todas las tareas;
2. la mejora persiste en todos los estratos de dificultad;
3. contadores de riesgo con baja variacion en este corte.

## 4. Discusion
### 4.1 Lectura principal
La evidencia muestra un trade-off calidad-latencia:
1. mejora la calidad compuesta y los proxies de robustez de decision;
2. aumenta la latencia operacional en la implementacion actual;
3. d_efficiency cae pese a la mejora global, sugiriendo tension entre proxies de diseno y utilidad final end-to-end.

### 4.2 Implicaciones metodologicas
Para sistemas agenticos no alcanza con optimizar una metrica. Se requiere un marco multiobjetivo con politicas explicitas que balanceen:
1. calidad inferencial,
2. reduccion de incertidumbre,
3. costo computacional,
4. calidad de evidencia reproducible.

### 4.3 Criterio de adopcion realista
Adoptar por defecto solo si:
1. efecto positivo y estable en endpoint primario;
2. riesgo no degradado;
3. costo temporal aceptable para el contexto operativo.

### 4.4 Amenazas a la validez
1. el piloto es pequeno;
2. algunas metricas de riesgo saturan en cero;
3. el tiempo mezcla computo y overhead de orquestacion;
4. falta cerrar modelo mixto completo para inferencia jerarquica.

### 4.5 Siguiente paso analitico
Modelo recomendado:
ImpactScore ~ Condition + Difficulty + Condition:Difficulty + (1|Task) + (1|Seed) + (1|Source)

## 5. Conclusion
La propuesta doekit-enhanced muestra mejora consistente en calidad compuesta de decision con datos reales, junto con menor incertidumbre y mayor ganancia esperada. El costo observable es la latencia y una deuda tecnica en d_efficiency para ciertos escenarios. La conclusion operativa es adopcion condicionada: preservar ganancias semanticas/decisionales, optimizar overhead y calibrar estrategia de diseno por tarea.

## 6. Anexo de reproducibilidad
Paquete minimo por corrida:
1. recommendation.json
2. metrics.json
3. trace.log con REAL_DATA_CONFIRMED=true
4. evidence.json con source_id aprobado
5. artefacto ejecutable (.py/.ipynb/.md)

Paquete sugerido para suplemento del paper:
1. protocolos (piloto y meta-diseno);
2. configuracion experimental;
3. tablas consolidadas de metricas;
4. scripts de reporte;
5. hash de commit y lockfile de entorno.

## 7. Figuras metodologicas (Mermaid)
Las figuras por etapa y el diagrama global del flujo de informacion estan en:

- Article/FIGURAS_MERMAID_IMRAD.md

Orden sugerido de insercion en manuscrito:
1. Introduccion: figura conceptual (de salida numerica a decision).
2. Metodos: arquitectura modular, diseno empirico y auditoria de evidencia.
3. Resultados/Discusion: criterio de adopcion y trade-off.
4. Cierre: flujo global end-to-end.
