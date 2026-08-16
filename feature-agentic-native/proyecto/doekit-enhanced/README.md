# doekit-enhanced

Extensiones para doekit que habilitan experimentación autónoma y razonamiento semántico.

```
[███████████████░░░░░] Fases 1-4 con MVP funcional (75%)
```

---

## Visión

Transformar **doekit** (librería de Design of Experiments) de una herramienta para experimentadores humanos en un **motor de experimentación autónomo** que agentes de IA pueden usar efectivamente.

### Problema que Resolvemos

doekit actual retorna solo **datos numéricos**:
```python
{"D_efficiency": 46.1, "worth_it": True}
```

Los agentes de IA necesitan **interpretación semántica**:
```
"D-efficiency de 46.1% es aceptable dado presupuesto limitado.
Aunque no es óptima, permite estimar efectos principales con precisión razonable.
Recomendación: Proceder, pero considerar diseño D-optimal si presupuesto aumenta."
```

## Características Principales

### ✅ Fase 1: Capa Semántica (En Progreso)

**Bloques Semántico-Numéricos**: Cada resultado incluye interpretación en lenguaje natural.

```python
from doekit_enhanced.semantic import interpret_result
import doekit as ed

# Obtener recomendación estándar
rec = ed.recommend_design(goal="optimization", factors=3, budget=20)

# Agregar interpretación semántica
semantic = interpret_result(rec)

# Usar para razonamiento
print(semantic.interpretation)
# > "Se recomienda D-optimal con 13 corridas experimentales"

print(semantic.reasoning)
# > "D-optimal fue seleccionado porque balancea óptimamente eficiencia y presupuesto..."

print(semantic.prompt_injection)  # Listo para inyectar en LLM
```

**Componentes**:
- `SemanticResult`: Estructura dual numérica-semántica
- Interpretadores especializados por tipo de resultado
- Sistema de registro para auto-detección
- Templates reutilizables para prompts

### ✅ Fase 2: Motor de Decisión Autónomo (MVP Inicial)

Reemplaza decisiones booleanas con scoring cuantificable y políticas configurables.

```python
from decision import build_context, decide_next_action

ctx = build_context(
  budget_total=50,
  budget_spent=12,
  risk_tolerance="moderate",
  metrics={
    "delta_D_efficiency": 9.4,
    "delta_mean_power": 0.05,
    "delta_G_efficiency": -1.2,
    "n_add": 4,
  },
  uncertainty=0.15,
)

decision = decide_next_action(ctx)

print(decision.action)  # "continue" | "stop" | "refine_model"
print(decision.confidence)  # 0.85
print(decision.score.composite)  # 0.72
print(decision.prompt_injection)  # Explicación completa
```

Incluye:
- `DecisionContext`, `Decision`, `DecisionPolicy`
- `ContinuationScorer`, `MultiObjectiveScorer`
- `ThresholdPolicy`, `RiskAdaptivePolicy`, `BudgetAwarePolicy`
- Tests reales en `decision/tests/test_decision_real.py`

### ✅ Fase 3: Gestión de Incertidumbre (MVP Inicial)

Cuantificación de incertidumbre y propagación a través de decisiones.

- Intervalos de confianza en predicciones
- Probability of Improvement (PI)
- Expected Improvement (EI)
- Uncertainty-aware selection

```python
from decision import (
  build_context,
  decide_next_action,
  estimate_uncertainty_from_proposal,
)

# proposal y comparison obtenidos desde doekit
uncertainty = estimate_uncertainty_from_proposal(proposal, comparison)

ctx = build_context(
  budget_total=40,
  budget_spent=18,
  metrics=metrics,
  proposal=proposal,
  comparison=comparison,
)

decision = decide_next_action(ctx, uncertainty_estimate=uncertainty)
print(decision.score.uncertainty_penalty)
```

### ✅ Fase 4 (bloque inicial): Monitoring de convergencia

Deteccion de convergencia por criterio de mejora marginal consecutiva.

```python
from monitoring import DefaultConvergenceChecker

history = [
  {"wave": 1, "metrics": {"delta_D_efficiency": 10.0}},
  {"wave": 2, "metrics": {"delta_D_efficiency": 10.3}},
  {"wave": 3, "metrics": {"delta_D_efficiency": 10.5}},
]

checker = DefaultConvergenceChecker(
  metric_key="delta_D_efficiency",
  marginal_threshold=0.5,
  consecutive_required=2,
)
result = checker.check(history)

print(result.converged, result.should_stop, result.reason)
```

### ✅ Pipeline Opcional Configurable (sin perder modularidad)

El flujo por etapas sigue disponible de forma modular, y adicionalmente existe
un pipeline opcional para casos estandar:

```python
from decision import build_context, run_decision_pipeline, DecisionPipelineConfig

ctx = build_context(budget_total=40, budget_spent=18, metrics=metrics, proposal=proposal, comparison=comparison)

cfg = DecisionPipelineConfig(
  enable_uncertainty=True,
  enable_convergence=True,
  enable_diagnostics=True,
  enable_events=True,
)

pipeline_out = run_decision_pipeline(ctx, history=history, config=cfg)

print(pipeline_out.executed_stages)
print(pipeline_out.decision.action)
```

Esto permite:
- ejecutar todo end-to-end en modo estandar,
- o desactivar etapas segun contexto sin acoplar todo el sistema.

### ✅ Fase 5: Memory (MVP)

Meta-aprendizaje desde historial:
- `ExperimentStore` y `ExperimentRecord`
- `PriorLearner` para transfer de señales históricas
- `HistoricalRecommender` para sugerencias accionables

### ✅ Fase 6: Integrations (MVP)

Integración de Bayesian Optimization en modo pragmático:
- `BayesianOptAdapter`
- fallback determinístico por Expected Improvement
- preparado para aprovechar BoTorch cuando esté disponible

```python
from memory import ExperimentStore, ExperimentRecord, PriorLearner
from integrations import propose_with_bayesian_opt

store = ExperimentStore()
store.add(ExperimentRecord("exp-1", "optimization", ["X1", "X2"], metrics={"delta_D_efficiency": 8.0}))
prior = PriorLearner(store).learn("optimization", ["X1", "X2"])

proposal = propose_with_bayesian_opt(
  candidate_pool=[{"X1": -0.5, "X2": 0.2}, {"X1": 0.4, "X2": -0.1}],
  objective_values=[0.42, 0.47],
  uncertainty_values=[0.08, 0.11],
)
print(prior.expected_delta_d_efficiency, proposal.selected_index)
```

## Politicas de Desarrollo

### Politica de Testing por Impacto

Para mantener velocidad de iteracion sin perder calidad:

- Ejecutar por defecto solo los tests de los modulos impactados por el cambio.
- Ejecutar bateria completa solo cuando sea necesario:
  - cambios transversales en core/API compartida,
  - preparacion de release,
  - o investigacion de regresiones no localizadas.

Ejemplos:

```bash
# Cambio en monitoring
python -m pytest monitoring/tests -q

# Cambio en decision
python -m pytest decision/tests -q

# Solo cuando aplique validacion global
python -m pytest semantic/tests decision/tests monitoring/tests -q
```

---

## Instalación

### Desarrollo (Actual)

```bash
# Clonar repositorio
git clone <repo-url>
cd proyecto/doekit-enhanced

# Instalar en modo desarrollo
pip install -e .

# O solo doekit original por ahora
pip install doekit
```

### Producción (Futuro)

```bash
pip install doekit-enhanced
# o
pip install "doekit[enhanced]"
```

---

## Quick Start

### Ejemplo 1: Interpretación Semántica Manual

```python
from doekit_enhanced.semantic import SemanticResult

result = SemanticResult(
    numerical={"D_efficiency": 46.1, "G_efficiency": 79.7},
    interpretation="Diseño aceptable dado presupuesto limitado",
    reasoning="D-efficiency de 46.1% permite estimar efectos principales...",
    context="Diseño de 13 corridas para modelo cuadrático (10 parámetros)",
    warnings=["D-efficiency relativamente baja - varianzas mayores"],
    recommendations=["Proceder con diseño actual", "Monitorear R² después de primera wave"]
)

print(result.prompt_injection)  # Texto estructurado para LLM
```

### Ejemplo 2: Interpretador Personalizado

```python
from doekit_enhanced.semantic import SemanticInterpreter, SemanticResult

class MyInterpreter(SemanticInterpreter):
    def validate_input(self, result):
        return isinstance(result, dict) and "value" in result
    
    def interpret(self, numerical_result, context=None):
        value = numerical_result["value"]
        return SemanticResult(
            numerical=numerical_result,
            interpretation=f"Valor observado: {value}",
            reasoning="Medición directa del sistema",
            context="Experimento de validación",
            warnings=[],
            recommendations=[]
        )

# Usar
interpreter = MyInterpreter()
semantic = interpreter.interpret({"value": 42})
```

### Ejemplo 3: Integración con doekit (Patrón Adapter)

```python
import doekit as ed
from doekit_enhanced.semantic import interpret_result

# Workflow normal de doekit
rec = ed.recommend_design(goal="optimization", factors=3, budget=20)

# Agregar semántica post-hoc
semantic = interpret_result(rec)

# Usar ambas representaciones
print("Numérico:", semantic.numerical.design.n_runs)
print("Semántico:", semantic.interpretation)
```

---

## Arquitectura

```
doekit-enhanced/
├── semantic/           # Interpretación semántica (FASE 1) ✅ 40%
│   ├── core.py         # SemanticResult, interpretadores base
│   ├── interpreters.py # Interpretadores especializados
│   ├── builders.py     # Construcción de prompts
│   └── templates.py    # Templates reutilizables
│
├── decision/           # Motor de decisión (FASE 2) ⏳
│   ├── policies.py     # Políticas de decisión
│   ├── scoring.py      # Sistemas de scoring
│   └── uncertainty.py  # Cuantificación incertidumbre (FASE 3)
│
├── monitoring/         # Observabilidad (FASE 4) ⏳
│   ├── convergence.py  # Detección convergencia
│   └── diagnostics.py  # Diagnósticos automáticos
│
├── memory/             # Meta-aprendizaje (FASE 5) ⏳
│   ├── store.py        # Almacén experimentos
│   └── transfer.py     # Transfer learning
│
└── integrations/       # Integraciones (FASE 6) ⏳
    └── bayesian_opt.py # BoTorch, etc.
```

**Principio de Diseño**: Extensión **aditiva**, no modificación de doekit.

Ver [ARCHITECTURE.md](./ARCHITECTURE.md) para detalles completos.

---

## Documentación

### Fundamentación Teórica

- [semantic/docs/theory.md](./semantic/docs/theory.md) - Teoría completa del módulo semántico
  - Motivación y gap identificado
  - 7 teorías fundamentales (Explainable AI, dual-process, etc.)
  - Arquitectura y patrones
  - Validación teórica
  - Referencias académicas

### Guías Prácticas

- [semantic/docs/examples.md](./semantic/docs/examples.md) - 8 ejemplos ejecutables
  - Uso básico
  - Interpretadores personalizados
  - Integración con doekit
  - Testing
  - Serialización

### Plan de Desarrollo

- [SUGGESTIONS_DOEKIT.md](../SUGGESTIONS_DOEKIT.md) - Plan completo de mejoras (2100+ líneas)
  - Gap analysis detallado
  - 6 fases de desarrollo
  - Código de ejemplo completo
  - Métricas de validación
  - Evaluación de suficiencia

- [PROGRESS.md](./PROGRESS.md) - Estado actual del proyecto
  - Tracking por fase
  - Métricas de progreso
  - Próximos pasos
  - Decisiones de diseño

---

## Estado del Proyecto

**Versión Actual**: 0.1.0-alpha (Fase 1 en desarrollo)

### Completado ✅

- [x] Estructura de proyecto
- [x] Documentación teórica completa
- [x] Módulo semantic/core.py (estructuras base)
- [x] Ejemplos de uso documentados
- [x] Plan de desarrollo detallado

### En Progreso 🔄

- [ ] Interpretadores específicos (RecommendationInterpreter, etc.)
- [ ] Builders y templates
- [ ] Tests unitarios

### Próximos Pasos

1. Completar interpretadores específicos
2. Crear demo ejecutable end-to-end
3. Tests con cobertura >90%
4. Release 0.1.0 (Fase 1 completa)

Ver [PROGRESS.md](./PROGRESS.md) para tracking detallado.

---

## Casos de Uso

### 1. Agente Autónomo Simple

```python
# Obtener resultado con semántica
semantic = interpret_result(experiment_result)

# Construir prompt para LLM
agent_prompt = f"""
Tarea: Decidir próxima acción experimental.

{semantic.prompt_injection}

¿Cuál es tu decisión?
"""

# LLM procesa y decide
decision = llm.complete(agent_prompt)
```

### 2. Supervisión Humana de Agente

```python
# Agente genera recomendación
semantic = agent.recommend_design(...)

# Humano revisa interpretación semántica
print(semantic.interpretation)
print("\nDetalle:", semantic.reasoning)
for warning in semantic.warnings:
    print(f"⚠️  {warning}")

# Humano aprueba o rechaza
if user_approves():
    execute(semantic.numerical.design)
```

### 3. Auditoría de Decisiones

```python
# Cargar decisión pasada
experiment = load_from_storage("exp_123")

# Revisar razonamiento
print("Decisión tomada:", experiment.metadata["timestamp"])
print("Razonamiento:", experiment.reasoning)
print("Contexto:", experiment.context)
print("Advertencias consideradas:", experiment.warnings)
```

---

## Validación Teórica

### Fundamentación

El diseño está basado en teorías establecidas:

- **Explainable AI** (Miller, 2019): Explicaciones contrastivas, selectivas, sociales
- **Dual-Process Theory** (Kahneman, 2011): Sistema 1 (semántica) + Sistema 2 (numérica)
- **Conversational Maxims** (Grice, 1975): Cantidad, calidad, relevancia, manera
- **Distributed Cognition** (Hutchins, 1995): Artefactos cognitivos para razonamiento

Ver [semantic/docs/theory.md](./semantic/docs/theory.md) para detalles completos.

### Métricas de Calidad

Una interpretación semántica es **buena** si:

1. **Fidelidad**: Representa fielmente resultado numérico
2. **Completitud**: Cubre aspectos críticos para decisión
3. **Concisión**: No más largo de lo necesario (200-400 palabras)
4. **Accionabilidad**: Lleva a acción clara
5. **Rastreabilidad**: Verificable contra datos

---

## Contribuir

El proyecto está en desarrollo activo. Buscamos contribuciones en:

- Interpretadores especializados adicionales
- Templates de prompts optimizados
- Tests y validación
- Documentación de casos de uso
- Integraciones con otras librerías

### Desarrollo

```bash
# Setup
git clone <repo>
cd doekit-enhanced
pip install -e ".[dev]"

# Tests
pytest tests/ --cov=doekit_enhanced

# Lint
black doekit_enhanced/
mypy doekit_enhanced/
```

---

## Comparación con Alternativas

| | doekit + enhanced | scikit-optimize | Optuna | pyDOE2 |
|---|---|---|---|---|
| DoE Clásico | ✅ Completo | ❌ No | ❌ No | ⚠️ Básico |
| Sequential/Adaptativo | ✅ Nativo | ✅ BO | ✅ BO | ❌ No |
| **Interpretación Semántica** | ✅ Sí | ❌ No | ❌ No | ❌ No |
| **Decisión Autónoma** | ✅ Sí (Fase 2) | ⚠️ Limited | ⚠️ Limited | ❌ No |
| Transparencia | ✅ Alta | ⚠️ Media | ⚠️ Media | ✅ Alta |
| Gestión Incertidumbre | ✅ Sí (Fase 3) | ✅ Sí | ✅ Sí | ❌ No |

**Diferenciador clave**: Única librería con **capa semántica nativa** para agentes de IA.

---

## Roadmap

### Q3 2026
- ✅ Fase 0: Infraestructura
- 🔄 Fase 1: Módulo semantic (en progreso)
- ⏳ Release 0.1.0 (MVP)

### Q4 2026
- ⏳ Fase 2: Motor de decisión
- ⏳ Fase 3: Gestión incertidumbre
- ⏳ Release 0.2.0

### Q1 2027
- ⏳ Fase 4: Monitoring
- ⏳ Fase 5: Meta-aprendizaje
- ⏳ Fase 6: Integraciones
- ⏳ Release 1.0.0

---

## Licencia

MIT License (misma que doekit)

---

## Referencias

### Proyecto Base
- [doekit](https://github.com/ropensci/doekit) - Librería original de DoE

### Referencias Académicas
- Grice, H. P. (1975). Logic and conversation.
- Hutchins, E. (1995). Cognition in the Wild.
- Kahneman, D. (2011). Thinking, Fast and Slow.
- Miller, T. (2019). Explanation in artificial intelligence. *Artificial Intelligence*, 267, 1-38.

Ver documentación completa para referencias adicionales.

---

## Contacto

**Autores**: doekit-enhanced contributors  
**Repositorio**: TBD  
**Issues**: TBD  
**Documentación**: Ver `/docs` en este repositorio

---

**Status**: 🚧 Pre-alpha - Desarrollo activo  
**Última actualización**: 2026-08-13
