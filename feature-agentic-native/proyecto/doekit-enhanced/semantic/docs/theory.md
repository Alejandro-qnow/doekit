# Módulo Semantic - Fundamentación Teórica

## 1. Motivación

### 1.1 El Problema: Gap Semántico en DoE Computacional

Los sistemas de Design of Experiments tradicionales retornan **solo estructuras numéricas**:

```python
# doekit actual
efficiency_metrics = {"D_efficiency": 46.1, "G_efficiency": 79.7, "A_efficiency": 26.8}
decision_flag = True  # worth_it
```

**Problema para agentes autónomos**:

Un agente de IA (especialmente LLM-based) que recibe estos datos no puede:

1. **Interpretar magnitudes**: ¿46.1% de D-efficiency es bueno o malo? ¿En qué contexto?
2. **Razonar sobre trade-offs**: ¿Por qué G-eff es 79.7% mientras D-eff es 46.1%? ¿Qué implica?
3. **Tomar decisiones informadas**: `worth_it=True` es binario. ¿Qué tan seguro estamos? ¿Qué riesgos hay?
4. **Comunicar a humanos**: Si supervisa a un agente, el humano necesita entender el razonamiento.

### 1.2 La Solución: Bloques Semántico-Numéricos

Cada resultado crítico debe incluir **dos representaciones paralelas**:

**Representación Numérica** (actual):
- Datos cuantitativos precisos
- Métricas computables
- Estructuras serializables

**Representación Semántica** (nueva):
- Interpretación en lenguaje natural
- Razonamiento explícito
- Contexto necesario
- Advertencias y recomendaciones
- Nivel de confianza cualitativo

**Representación Híbrida** (para LLMs):
- Texto estructurado combinando ambas
- Listo para inyectar en contexto
- Formato que facilita razonamiento

## 2. Fundamentación Teórica

### 2.1 Teoría de la Explicabilidad (Explainable AI)

**Principio de Transparencia Interpretativa** (Miller, 2019):
> "Las explicaciones efectivas deben ser contrastivas (por qué X en lugar de Y), 
> selectivas (enfocadas en causas relevantes), y sociales (adaptadas al receptor)."

Aplicado a DoE:
- **Contrastiva**: "D-optimal fue seleccionado en lugar de Box-Behnken porque..."
- **Selectiva**: No toda la matemática, solo lo relevante para la decisión
- **Social**: Lenguaje apropiado para el agente (técnico pero claro)

### 2.2 Razonamiento Dual-Process (Kahneman, 2011)

Sistemas de pensamiento:
- **Sistema 1**: Rápido, intuitivo, basado en heurísticas (interpretación semántica)
- **Sistema 2**: Lento, analítico, basado en lógica (cálculo numérico)

Un agente efectivo necesita **ambos**:
- Calcular métricas numéricas (Sistema 2)
- Interpretar resultados en contexto (Sistema 1)

### 2.3 Teoría de Marcos de Decisión (Tversky & Kahneman, 1981)

**Efecto de Framing**:
> "Diferentes presentaciones del mismo problema llevan a decisiones diferentes."

Ejemplo en DoE:

**Frame Numérico**:
```
D_efficiency: 46.1%
worth_it: True
```
¿Decisión? Incierta.

**Frame Semántico**:
```
D-efficiency de 46.1% es aceptable dado el presupuesto limitado (20 corridas).
Aunque no es óptima, es suficiente para estimar efectos principales con precisión razonable.
Recomendación: Proceder, pero considerar diseño D-optimal si presupuesto aumenta.
```
¿Decisión? Clara y fundamentada.

### 2.4 Cognición Distribuida (Hutchins, 1995)

El razonamiento no ocurre solo "dentro" del agente, sino en la interacción con artefactos externos.

**Artefactos cognitivos en DoE**:
- Tablas numéricas (artefacto computacional)
- Interpretaciones textuales (artefacto semántico)
- Prompts estructurados (artefacto conversacional)

Un agente autónomo necesita **artefactos que soporten su razonamiento**.

## 3. Arquitectura del Módulo Semantic

### 3.1 Componentes Principales

```
semantic/
├── core.py           # SemanticResult, estructuras base
├── interpreters.py   # Interpretadores especializados
├── builders.py       # Construcción de prompts
└── templates.py      # Templates reutilizables
```

### 3.2 SemanticResult: Estructura Base

```python
@dataclass
class SemanticResult:
    """Wrapper que agrega semántica a cualquier resultado numérico"""
    
    # NUMÉRICO (datos originales)
    numerical: Any
    
    # SEMÁNTICO (interpretación)
    interpretation: str         # Qué significa (1-2 oraciones)
    reasoning: str              # Por qué (justificación técnica)
    context: str                # Información de fondo necesaria
    warnings: list[str]         # Qué tener en cuenta
    recommendations: list[str]  # Qué hacer a continuación
    confidence_level: str       # Alta/Moderada/Baja con explicación
    
    # HÍBRIDO (para LLMs)
    prompt_injection: str       # Texto estructurado combinando todo
    
    # METADATOS
    metadata: dict              # Info adicional (función origen, timestamp, etc.)
```

**Diseño Justificado**:

1. **`numerical` preservado**: Backward compatibility total
2. **Campos semánticos separados**: Facilita procesamiento individual
3. **`prompt_injection` pre-construido**: Optimización - no construir cada vez
4. **Metadatos extensibles**: Permite rastreo y debugging

### 3.3 Interpretadores Especializados

Cada tipo de resultado tiene un interpretador específico:

```python
class SemanticInterpreter(ABC):
    """Interfaz base para interpretadores"""
    
    @abstractmethod
    def interpret(self, numerical_result: Any) -> SemanticResult:
        """Convierte resultado numérico en semántico"""
        pass
    
    @abstractmethod
    def contextualize(self, result: Any, context: dict) -> str:
        """Genera contexto específico"""
        pass
```

**Especializaciones**:
- `RecommendationInterpreter`: Para `Recommendation` de `recommend_design()`
- `EvaluationInterpreter`: Para `DesignEvaluation` de `evaluate()`
- `ProposalInterpreter`: Para `NextRunsProposal` de `propose_next_runs()`
- `FitInterpreter`: Para `FitResult` de `fit_linear_model()`
- `ComparisonInterpreter`: Para `DesignComparison` de `compare_designs()`

**Justificación de Especialización**:
- Cada tipo de resultado tiene contexto semántico diferente
- Especializados permiten interpretaciones precisas y relevantes
- Extensible: nuevos tipos → nuevos interpretadores

### 3.4 Builders: Construcción de Prompts

```python
class PromptBuilder:
    """Construye prompts estructurados para LLMs"""
    
    def build_decision_prompt(
        self,
        interpretation: str,
        reasoning: str,
        context: str,
        warnings: list[str],
        recommendations: list[str],
        numerical_summary: dict
    ) -> str:
        """Construye prompt para decisión de experimentación"""
        pass
    
    def build_analysis_prompt(self, ...) -> str:
        """Construye prompt para análisis de resultados"""
        pass
```

**Principios de Construcción**:

1. **Estructura consistente**: Siempre mismo formato (DECISIÓN → RAZONAMIENTO → CONTEXTO → ADVERTENCIAS → RECOMENDACIONES)
2. **Información suficiente**: Todo lo necesario para razonar, nada superfluo
3. **Formato parseable**: Secciones claramente delimitadas
4. **Densidad semántica**: Máxima información en mínimo espacio

### 3.5 Templates: Reutilización

```python
RECOMMENDATION_TEMPLATE = """
RECOMENDACIÓN DE DISEÑO EXPERIMENTAL:

Diseño recomendado: {method}
Corridas experimentales: {n_runs}
Eficiencia esperada: D={d_eff:.1f}%, G={g_eff:.1f}%

RAZONAMIENTO:
{reasoning}

CONTEXTO:
{context}

{warnings_section}

{recommendations_section}
"""
```

**Ventajas**:
- Consistencia entre interpretaciones
- Fácil modificación global
- Testeable (validar formato)
- Localizable (templates en varios idiomas)

## 4. Patrones de Uso

### 4.1 Patrón Decorador (Extensión No Invasiva)

```python
@with_semantics(RecommendationInterpreter)
def recommend_design(..., include_semantics=False):
    # Código original sin cambios
    return recommendation
```

**Ventajas**:
- Código original intacto
- Opt-in (no rompe nada)
- Fácil de mantener

### 4.2 Patrón Adapter (Envolver Resultados Existentes)

```python
# Sin modificar doekit
rec = ed.recommend_design(goal="optimization", factors=3)

# Aplicar semántica post-hoc
from doekit_enhanced.semantic import RecommendationInterpreter
interpreter = RecommendationInterpreter()
semantic_rec = interpreter.interpret(rec)
```

**Ventajas**:
- Compatible con doekit sin modificar
- Aplicable selectivamente
- Componible

### 4.3 Patrón Factory (Interpretadores Automáticos)

```python
def interpret_result(result: Any, context: dict = None) -> SemanticResult:
    """Detecta tipo de resultado y aplica interpretador apropiado"""
    if isinstance(result, Recommendation):
        return RecommendationInterpreter().interpret(result)
    elif isinstance(result, DesignEvaluation):
        return EvaluationInterpreter().interpret(result)
    # ...
```

**Ventajas**:
- API simple para usuario
- Extensible vía registro
- Type-safe

## 5. Fundamentación Psico-Lingüística

### 5.1 Teoría de Grice: Máximas Conversacionales

Las interpretaciones siguen máximas de Grice (1975):

1. **Cantidad**: Información suficiente, no excesiva
   - Interpretación: 1-2 oraciones clave
   - Reasoning: 2-4 oraciones de justificación
   - No dump completo de matemática

2. **Calidad**: Verdad y evidencia
   - Basado en cálculos reales
   - Advertencias cuando hay incertidumbre
   - No especulación

3. **Relevancia**: Pertinente al objetivo
   - Enfocado en la decisión a tomar
   - No información tangencial

4. **Manera**: Claro y ordenado
   - Estructura consistente
   - Terminología precisa pero accesible
   - Evitar ambigüedad

### 5.2 Teoría de Coherencia (van Dijk & Kintsch, 1983)

El texto debe tener **coherencia local y global**:

**Coherencia Local** (entre oraciones consecutivas):
```
"D-efficiency es 46.1%. ← Afirmación
Esto es aceptable dado el presupuesto limitado. ← Evaluación conectada
Aunque no es óptima... ← Concesión conectada"
```

**Coherencia Global** (estructura del prompt completo):
```
1. Qué (interpretación)
2. Por qué (razonamiento)
3. En qué contexto (situación)
4. Qué considerar (advertencias)
5. Qué hacer (recomendaciones)
```

### 5.3 Carga Cognitiva (Sweller, 1988)

Minimizar **carga extrínseca**, maximizar **carga germane**:

**Evitar**:
- Jerga innecesaria
- Información redundante
- Formato inconsistente

**Priorizar**:
- Conceptos relevantes
- Conexiones causales
- Estructura que guía razonamiento

## 6. Validación Teórica

### 6.1 Criterios de Calidad

Una interpretación semántica es **buena** si:

1. **Fidelidad**: Representa fielmente el resultado numérico
2. **Completitud**: Cubre aspectos críticos para decisión
3. **Concisión**: No más largo de lo necesario
4. **Accionabilidad**: Lleva a acción clara
5. **Rastreabilidad**: Se puede verificar vs. datos

### 6.2 Métricas de Evaluación

**Cuantitativas**:
- Longitud de prompt (target: 200-400 palabras)
- Cobertura de información (checklist de elementos necesarios)
- Consistencia de formato (validación de template)

**Cualitativas** (evaluación humana):
- Claridad (escala 1-5)
- Utilidad para decisión (escala 1-5)
- Correctitud técnica (binario)

## 7. Casos de Uso

### 7.1 Agente Autónomo Simple

```python
rec = ed.recommend_design(goal="optimization", factors=3, include_semantics=True)

# Agente lee prompt_injection
agent_context = f"""
Tarea: Diseñar experimento de optimización.

{rec.prompt_injection}

Decisión requerida: ¿Proceder con este diseño?
"""

# LLM procesa y decide
decision = llm.complete(agent_context)
```

### 7.2 Agente con Supervisión Humana

```python
rec = ed.recommend_design(..., include_semantics=True)

# Mostrar a humano
print(rec.interpretation)
print("\nDetalles:", rec.reasoning)
if rec.warnings:
    print("\n⚠️  Advertencias:")
    for w in rec.warnings:
        print(f"  - {w}")

# Humano decide
if user_approves():
    execute(rec.numerical.design)
```

### 7.3 Debugging y Auditoría

```python
# Auditar decisión pasada
rec = load_experiment("experiment_123")
print(rec.metadata)  # Timestamp, parámetros, etc.
print(rec.reasoning)  # Por qué se tomó esta decisión
print(rec.context)    # Qué info se consideró
```

## 8. Limitaciones y Trabajo Futuro

### 8.1 Limitaciones Actuales

1. **Interpretación estática**: Templates fijos, no adaptativos al usuario
2. **Sin personalización**: Mismo estilo para todos los agentes
3. **Idioma único**: Solo inglés inicialmente
4. **No considera historia**: Cada interpretación es independiente

### 8.2 Extensiones Futuras

1. **Interpretación adaptativa**: Ajustar nivel de detalle según audiencia
2. **Multilingüe**: Templates en varios idiomas
3. **Context-aware**: Considerar decisiones previas
4. **Interactive**: Permitir queries sobre interpretación

## 9. Referencias

- Grice, H. P. (1975). Logic and conversation. In Cole & Morgan (Eds.), Syntax and Semantics.
- Hutchins, E. (1995). Cognition in the Wild. MIT Press.
- Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.
- Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. Artificial Intelligence, 267, 1-38.
- Sweller, J. (1988). Cognitive load during problem solving. Cognitive Science, 12(2), 257-285.
- Tversky, A., & Kahneman, D. (1981). The framing of decisions. Science, 211(4481), 453-458.
- van Dijk, T. A., & Kintsch, W. (1983). Strategies of Discourse Comprehension. Academic Press.

---

**Siguiente**: Implementación en `semantic/core.py`
