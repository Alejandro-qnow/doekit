# Arquitectura de doekit-enhanced

## Estructura General

Siguiendo el patrón arquitectónico de doekit, organizamos las mejoras en módulos cohesivos:

```
doekit/                          # Código original
├── domain/                      # Factores, modelos, diseños
├── generation/                  # Generación de diseños
├── assessment/                  # Evaluación y análisis
├── orchestration/               # Orquestación de experimentos
├── presentation/                # Reportes y visualización
├── shared/                      # Utilidades compartidas
└── adapters/                    # Adaptadores externos

doekit-enhanced/                 # NUEVOS MÓDULOS
├── semantic/                    # ⭐ Capa semántica (FASE 1)
│   ├── __init__.py
│   ├── core.py                  # SemanticResult, wrappers base
│   ├── interpreters.py          # Interpretadores por tipo de resultado
│   ├── builders.py              # Construcción de prompts
│   ├── templates.py             # Templates de texto
│   └── docs/
│       ├── theory.md            # Fundamentación teórica
│       └── examples.md          # Ejemplos de uso
│
├── decision/                    # Motor de decisión (FASE 2)
│   ├── __init__.py
│   ├── policies.py              # Políticas de decisión
│   ├── scoring.py               # Sistemas de scoring
│   ├── criteria.py              # Criterios de parada
│   ├── uncertainty.py           # Cuantificación de incertidumbre (FASE 3)
│   └── docs/
│       ├── theory.md
│       └── examples.md
│
├── monitoring/                  # Observabilidad (FASE 4)
│   ├── __init__.py
│   ├── convergence.py           # Detección de convergencia
│   ├── diagnostics.py           # Diagnósticos automáticos
│   ├── events.py                # Sistema de eventos
│   └── docs/
│
├── memory/                      # Meta-aprendizaje (FASE 5)
│   ├── __init__.py
│   ├── store.py                 # Almacén de experimentos
│   ├── transfer.py              # Transfer learning
│   └── docs/
│
└── integrations/                # Integraciones externas (FASE 6)
    ├── __init__.py
    ├── bayesian_opt.py          # BO real (BoTorch, etc.)
    └── docs/
```

## Principios de Diseño

### 1. Compatibilidad con doekit Original

Todas las mejoras son **aditivas**, no modifican código existente:

```python
# Uso original (sin cambios)
rec = ed.recommend_design(goal="optimization", factors=3)

# Uso mejorado (opt-in)
rec = ed.recommend_design(goal="optimization", factors=3, include_semantics=True)
```

### 2. Separación de Concerns

Cada módulo tiene responsabilidad única:
- **semantic**: Interpretación y explicación
- **decision**: Lógica de decisión autónoma
- **monitoring**: Observación y diagnóstico
- **memory**: Persistencia y aprendizaje
- **integrations**: Conexión con externos

### 3. Inyección de Dependencias

Los módulos no se acoplan rígidamente:

```python
# Configurar interpretador personalizado
interpreter = CustomInterpreter()
result = ed.recommend_design(..., interpreter=interpreter)
```

### 4. Semantic-First

Todos los resultados críticos siguen estructura semántica:

```python
@dataclass
class SemanticResult:
    numerical: Any              # Resultado numérico original
    interpretation: str         # Qué significa
    reasoning: str              # Por qué
    context: str                # Contexto necesario
    warnings: list[str]         # Qué considerar
    recommendations: list[str]  # Qué hacer
    confidence_level: str       # Cuánta certeza
    prompt_injection: str       # Listo para LLM
```

## Orden de Implementación

### Fase 1: semantic (Base)
Fundamento para todos los demás módulos. Se integra con funciones existentes.

### Fase 2: decision (Core de Autonomía)
Depende de semantic. Habilita decisiones autónomas.

### Fase 3: decision.uncertainty (Refinamiento)
Extiende decision con cuantificación de incertidumbre.

### Fase 4: monitoring (Robustez)
Depende de semantic y decision. Añade observabilidad.

### Fase 5: memory (Optimización)
Usa semantic para almacenar conocimiento estructurado.

### Fase 6: integrations (Expansión)
Conecta con ecosistema externo (BoTorch, etc.).

## Integración con doekit

### Patrón de Extensión

```python
# En doekit/orchestration/advise/recommend.py (modificación mínima)

from doekit_enhanced.semantic import with_semantics, RecommendationInterpreter

@with_semantics(RecommendationInterpreter)  # Decorador añade semántica
def recommend_design(
    goal: str,
    factors,
    budget: Optional[int] = None,
    include_semantics: bool = False,  # Nuevo parámetro opt-in
    **kwargs
) -> Union[Recommendation, SemanticResult]:
    # Código original sin cambios
    # ...
    return recommendation  # Decorador envuelve si include_semantics=True
```

### Patrón de Composición

```python
# Usar sin modificar doekit
from doekit_enhanced.semantic import SemanticInterpreter

rec = ed.recommend_design(goal="optimization", factors=3)
interpreter = RecommendationInterpreter()
semantic_result = interpreter.interpret(rec)
print(semantic_result.prompt_injection)
```

## Versionado

```
doekit: 0.7.3 (sin modificar)
doekit-enhanced: 0.1.0 (módulos nuevos)
```

Instalación:
```bash
pip install doekit                    # Original
pip install doekit-enhanced           # Mejoras
# o
pip install doekit[enhanced]          # Ambos
```

---

**Siguiente**: Empezaremos con `semantic/` - documentación teórica y código base.
