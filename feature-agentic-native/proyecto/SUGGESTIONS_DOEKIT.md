# SUGGESTIONS_DOEKIT: Plan de Mejoras para Autonomía de Agentes

**Versión**: 1.0  
**Fecha**: 2026-08-13  
**Objetivo**: Transformar doekit de "asistente para humanos" a "motor de experimentación autónomo"

---

## TABLA DE CONTENIDOS

1. [Gap Analysis: Dimensión Semántica](#gap-analysis-dimensión-semántica)
2. [Arquitectura Propuesta](#arquitectura-propuesta)
3. [Plan de Implementación por Fases](#plan-de-implementación-por-fases)
4. [Branches y Workflow de Desarrollo](#branches-y-workflow-de-desarrollo)
5. [Especificaciones Detalladas por Feature](#especificaciones-detalladas-por-feature)
6. [Métricas de Validación](#métricas-de-validación)
7. [Evaluación de Suficiencia](#evaluación-de-suficiencia)

---

## GAP ANALYSIS: DIMENSIÓN SEMÁNTICA

### Problema Identificado

doekit actual retorna **solo estructuras numéricas**:

```python
# Estado actual
proposal = ed.propose_next_runs(design, response=y, n_add=4)
print(proposal.comparison.worth_it)  # True/False
print(proposal.comparison.delta)     # {'D_efficiency': 12.3, ...}
```

**Problema para agentes**: Un LLM recibiendo esto no puede razonar sobre:
- ¿Por qué este delta de 12.3% es significativo?
- ¿Qué significa "worth_it=True" en el contexto específico?
- ¿Qué factores de riesgo considerar?
- ¿Cuáles son las alternativas y sus trade-offs?

### Solución: Bloques Semántico-Numéricos

**Concepto**: Cada resultado crítico debe incluir:

```python
{
    "numerical": {...},           # Datos numéricos (actual)
    "semantic": {
        "interpretation": "...",  # Interpretación del resultado
        "reasoning": "...",       # Razonamiento que llevó a esta decisión
        "context": "...",         # Contexto necesario para entender
        "warnings": [...],        # Advertencias/caveats
        "recommendations": [...], # Sugerencias accionables
        "confidence": "..."       # Nivel de confianza en lenguaje natural
    },
    "prompt_injection": "..."     # Texto listo para inyectar en contexto LLM
}
```

### Ejemplos Concretos

#### Ejemplo 1: Decisión de Continuar Experimentando

**Actual**:
```python
proposal.comparison.worth_it  # True
```

**Propuesto**:
```python
{
    "numerical": {
        "worth_it": True,
        "delta_D_eff": 12.3,
        "delta_G_eff": 8.7,
        "extra_runs": 4,
        "score": 2.1,
        "threshold": 1.6
    },
    "semantic": {
        "interpretation": "Los 4 experimentos adicionales propuestos mejoran significativamente la calidad del diseño",
        "reasoning": "El incremento de 12.3% en D-efficiency supera el umbral de 5% para justificar 4 corridas adicionales. El score de 2.1 excede el threshold de 1.6 por un margen cómodo (31% superior).",
        "context": "Estamos en wave 2 de 5 planificadas, con 45% del presupuesto consumido. La mejora en G-efficiency (8.7%) indica mejor capacidad de predicción en toda la región experimental.",
        "warnings": [
            "La mejora en D-efficiency es para el modelo cuadrático completo. Si solo algunos términos son activos, la ganancia real será menor.",
            "Asumir sigma=1.5 basado en wave anterior. Si la variabilidad real es mayor, el beneficio de power será menor."
        ],
        "recommendations": [
            "Proceder con los 4 experimentos propuestos.",
            "Monitorear términos activos después de esta wave - si solo 3-4 efectos son significativos, considerar simplificar el modelo para wave siguiente."
        ],
        "confidence": "Alta (score/threshold ratio = 1.31). Decisión robusta incluso bajo incertidumbre moderada."
    },
    "prompt_injection": """
DECISIÓN DE EXPERIMENTACIÓN:
Recomendación: CONTINUAR con 4 experimentos adicionales.
Confianza: Alta (131% sobre umbral mínimo).

Justificación técnica:
- Mejora en precisión de estimación (D-efficiency): +12.3%
- Mejora en capacidad de predicción (G-efficiency): +8.7%
- Costo: 4 corridas (8% del presupuesto total restante)

Contexto operacional:
Wave 2/5, presupuesto usado 45%. Modelo cuadrático con 10 términos.

Factores de decisión:
1. La mejora en D-efficiency (12.3%) supera ampliamente el umbral típico de 5% para justificar corridas adicionales.
2. El score compuesto (2.1) excede el threshold ajustado por costo (1.6) con margen del 31%.
3. Estamos en fase media del estudio - es el momento óptimo para refinar.

Riesgos a considerar:
- Si solo pocos términos del modelo son realmente activos, parte de esta mejora puede ser ilusoria.
- La estimación de sigma asume que la variabilidad se mantiene constante respecto a wave anterior.

Acción recomendada:
Ejecutar los 4 experimentos propuestos. Después de obtener resultados, re-evaluar si el modelo completo sigue siendo apropiado o si simplificación es prudente.
"""
}
```

#### Ejemplo 2: Detección de Convergencia

**Propuesto**:
```python
convergence_check = ed.check_convergence(experiment_history)

{
    "numerical": {
        "converged": True,
        "consecutive_waves_no_improvement": 2,
        "best_r_squared": 0.94,
        "improvement_last_wave": 0.003,
        "improvement_threshold": 0.01
    },
    "semantic": {
        "interpretation": "El sistema ha convergido. Experimentación adicional probablemente no mejorará significativamente el modelo.",
        "reasoning": "Las últimas 2 waves solo mejoraron R² en 0.003 y 0.005, ambas bajo el umbral de 0.01. El modelo actual explica 94% de la variabilidad, y los residuos muestran patrón aleatorio sin estructura aparente.",
        "context": "Completadas 4 waves (32 experimentos totales). Factores X1, X3, X5 consistentemente significativos. Interacción X1:X3 detectada en wave 3 y confirmada en wave 4.",
        "warnings": [
            "La convergencia detectada es para el modelo cuadrático actual. Efectos de orden superior o no-linealidades complejas no han sido explorados.",
            "La región experimental es un hipercubo [-1,1]³. Fuera de esta región las predicciones son extrapolaciones no validadas."
        ],
        "recommendations": [
            "Detener experimentación exploratoria.",
            "Realizar 2-3 corridas de confirmación en el punto óptimo predicho.",
            "Si se requiere optimización adicional, considerar método de escalada (steepest ascent) desde el óptimo actual."
        ],
        "confidence": "Alta. Convergencia confirmada por múltiples indicadores: R², mejora marginal, estabilidad de coeficientes."
    },
    "prompt_injection": """
ESTADO DE CONVERGENCIA:
Estatus: CONVERGIDO - Experimentación adicional no justificada.
Confianza: Alta (múltiples indicadores concordantes).

Evidencia de convergencia:
1. Mejora marginal últimas 2 waves: 0.003 y 0.005 (umbral: 0.01)
2. R² actual: 0.94 (excelente ajuste)
3. Residuos sin estructura - patrón aleatorio
4. Coeficientes estables últimas 2 waves (variación <5%)

Contexto del estudio:
- 4 waves completadas, 32 experimentos totales
- Factores activos identificados: X1, X3, X5
- Interacción significativa: X1:X3
- Modelo: Cuadrático con términos seleccionados

Interpretación:
El modelo ha alcanzado su capacidad explicativa para este diseño y región experimental. Seguir agregando puntos en la misma región solo reduciría error estándar marginalmente, sin descubrir nueva información estructural.

Limitaciones conocidas:
- Convergencia es relativa al modelo cuadrático. Efectos cúbicos o interacciones de orden superior no explorados.
- Validez limitada a región experimental [-1,1]³.

Próximos pasos recomendados:
1. Ejecutar 2-3 réplicas en punto óptimo predicho para confirmar.
2. Si se requiere mejora adicional, usar método de steepest ascent para mover región experimental hacia nuevo óptimo.
3. Documentar modelo final y limitaciones para uso futuro.

Decisión: DETENER experimentación exploratoria.
"""
}
```

### Bloques Semánticos Necesarios por Función

| Función | Bloque Semántico Requerido |
|---------|---------------------------|
| `recommend_design()` | Explicación de por qué este diseño, qué sacrifica, para qué es óptimo |
| `propose_next_runs()` | Razonamiento de decisión, contexto de wave, riesgos |
| `evaluate()` | Interpretación de métricas, qué significan para este contexto |
| `fit_linear_model()` | Diagnóstico de ajuste, supuestos validados/violados, términos significativos explicados |
| `compare_designs()` | Trade-offs explícitos, recomendación contextualizada |
| `check_convergence()` | Evidencia de convergencia, confianza, próximos pasos |

---

## ARQUITECTURA PROPUESTA

### Nueva Estructura de Módulos

```
doekit/
├── core/                    # Existente (renombrado desde domain/)
│   ├── factors.py
│   ├── models.py
│   └── design.py
│
├── generation/              # Existente
│
├── assessment/              # Existente
│
├── orchestration/           # Existente
│
├── decision/                # NUEVO - Motor de decisión autónomo
│   ├── __init__.py
│   ├── policies.py          # Políticas de decisión configurables
│   ├── criteria.py          # Criterios de parada/continuación
│   ├── uncertainty.py       # Cuantificación de incertidumbre
│   └── cost_models.py       # Modelos de costo asimétrico
│
├── semantic/                # NUEVO - Capa semántica
│   ├── __init__.py
│   ├── interpreters.py      # Interpretación de resultados numéricos
│   ├── reasoners.py         # Generación de razonamiento
│   ├── templates.py         # Templates de texto
│   └── prompt_builder.py    # Construcción de prompts
│
├── memory/                  # NUEVO - Meta-aprendizaje
│   ├── __init__.py
│   ├── experiment_store.py  # Almacén de experimentos previos
│   ├── transfer.py          # Transfer learning
│   └── priors.py            # Priors bayesianos aprendidos
│
├── monitoring/              # NUEVO - Observabilidad
│   ├── __init__.py
│   ├── events.py            # Sistema de eventos
│   ├── diagnostics.py       # Diagnósticos automáticos
│   └── checkpointing.py     # Persistencia automática
│
└── integrations/            # NUEVO - Integraciones externas
    ├── __init__.py
    ├── bayesian_opt.py      # Integración real con BO
    ├── causal.py            # Inferencia causal
    └── active_learning.py   # Active learning strategies
```

### Interfaces Clave

#### 1. DecisionPolicy (decision/policies.py)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class DecisionContext:
    """Contexto completo para tomar decisión"""
    current_design: Design
    responses: np.ndarray
    budget_total: int
    budget_spent: int
    wave_number: int
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    objective: Literal["minimize_runs", "maximize_precision", "balanced"]
    confidence_required: float  # 0-1

@dataclass
class Decision:
    """Decisión estructurada con semántica"""
    # Numérico
    action: Literal["continue", "stop", "refine_model", "expand_region"]
    confidence: float  # 0-1
    expected_gain: dict  # Métricas esperadas
    cost: dict  # Costo de la acción
    
    # Semántico
    interpretation: str
    reasoning: str
    context: str
    warnings: list[str]
    recommendations: list[str]
    confidence_explanation: str
    
    # Prompt-ready
    prompt_injection: str
    
    # Alternativas
    alternatives: list['Decision']

class DecisionPolicy(ABC):
    """Política base para decisiones autónomas"""
    
    @abstractmethod
    def decide(self, context: DecisionContext) -> Decision:
        """Tomar decisión basada en contexto"""
        pass
    
    @abstractmethod
    def explain(self, decision: Decision) -> str:
        """Explicar decisión en lenguaje natural detallado"""
        pass
```

#### 2. SemanticInterpreter (semantic/interpreters.py)

```python
from abc import ABC, abstractmethod

class SemanticInterpreter(ABC):
    """Base para interpretación semántica de resultados"""
    
    @abstractmethod
    def interpret_metrics(self, metrics: dict, context: dict) -> dict:
        """
        Convierte métricas numéricas en interpretación semántica
        
        Returns:
            {
                "interpretation": str,
                "context": str,
                "warnings": list[str],
                "recommendations": list[str]
            }
        """
        pass
    
    @abstractmethod
    def build_prompt_injection(self, numerical: dict, semantic: dict) -> str:
        """Construye texto listo para inyectar en contexto LLM"""
        pass

class EfficiencyInterpreter(SemanticInterpreter):
    """Interpreta métricas de eficiencia (D/A/G/I)"""
    
    def interpret_metrics(self, metrics: dict, context: dict) -> dict:
        d_eff = metrics.get("D_efficiency", 0)
        g_eff = metrics.get("G_efficiency", 0)
        
        # Interpretación contextualizada
        if d_eff >= 80:
            d_interp = "excelente - cerca del óptimo teórico"
        elif d_eff >= 60:
            d_interp = "buena - diseño eficiente aunque hay margen de mejora"
        elif d_eff >= 40:
            d_interp = "aceptable - considera diseño D-optimal si presupuesto lo permite"
        else:
            d_interp = "pobre - este diseño no es eficiente para el modelo especificado"
        
        interpretation = (
            f"D-efficiency de {d_eff:.1f}% es {d_interp}. "
            f"G-efficiency de {g_eff:.1f}% indica capacidad de predicción "
            f"{'excelente' if g_eff >= 80 else 'moderada' if g_eff >= 60 else 'limitada'}."
        )
        
        warnings = []
        if d_eff < 50:
            warnings.append(
                "D-efficiency baja sugiere que el diseño no es óptimo para estimar "
                "los parámetros del modelo. Varianzas de coeficientes serán grandes."
            )
        
        if abs(d_eff - g_eff) > 20:
            warnings.append(
                f"Gap significativo entre D-efficiency ({d_eff:.1f}%) y G-efficiency ({g_eff:.1f}%). "
                "Diseño optimiza uno a costa del otro."
            )
        
        recommendations = []
        if d_eff < 60 and context.get("budget_flexible"):
            recommendations.append(
                "Considera aumentar número de corridas o usar diseño D-optimal "
                "para mejorar precisión de estimación."
            )
        
        return {
            "interpretation": interpretation,
            "context": self._build_context(metrics, context),
            "warnings": warnings,
            "recommendations": recommendations
        }
    
    def _build_context(self, metrics: dict, context: dict) -> str:
        n_runs = metrics.get("n_runs", "?")
        n_params = context.get("n_params", "?")
        model_type = context.get("model_type", "desconocido")
        
        return (
            f"Diseño de {n_runs} corridas para modelo {model_type} "
            f"con {n_params} parámetros. "
            f"DOF = {n_runs - n_params}."
        )
    
    def build_prompt_injection(self, numerical: dict, semantic: dict) -> str:
        return f"""
EVALUACIÓN DE CALIDAD DEL DISEÑO EXPERIMENTAL:

Métricas clave:
- D-efficiency: {numerical['D_efficiency']:.1f}% (precisión de estimación)
- G-efficiency: {numerical['G_efficiency']:.1f}% (calidad de predicción)
- A-efficiency: {numerical['A_efficiency']:.1f}% (varianza promedio)

{semantic['interpretation']}

Contexto: {semantic['context']}

{self._format_warnings(semantic['warnings'])}

{self._format_recommendations(semantic['recommendations'])}
"""
    
    def _format_warnings(self, warnings: list[str]) -> str:
        if not warnings:
            return ""
        return "ADVERTENCIAS:\n" + "\n".join(f"⚠ {w}" for w in warnings)
    
    def _format_recommendations(self, recs: list[str]) -> str:
        if not recs:
            return ""
        return "RECOMENDACIONES:\n" + "\n".join(f"→ {r}" for r in recs)
```

#### 3. UncertaintyQuantifier (decision/uncertainty.py)

```python
from dataclasses import dataclass
import numpy as np
from scipy import stats

@dataclass
class UncertaintyEstimate:
    """Estimación de incertidumbre con semántica"""
    # Numérico
    point_estimate: float
    confidence_interval: tuple[float, float]
    confidence_level: float  # 0.95 típicamente
    standard_error: float
    
    # Semántico
    interpretation: str
    reliability: Literal["high", "moderate", "low"]
    
    # Prompt-ready
    summary: str

class UncertaintyQuantifier:
    """Cuantifica y explica incertidumbre"""
    
    def estimate_prediction_uncertainty(
        self,
        fit_result: FitResult,
        prediction_point: np.ndarray,
        confidence_level: float = 0.95
    ) -> UncertaintyEstimate:
        """
        Calcula intervalo de confianza para predicción con interpretación
        """
        # Cálculo numérico (basado en estadística estándar)
        X_pred = prediction_point.reshape(1, -1)
        var_pred = self._prediction_variance(X_pred, fit_result)
        se_pred = np.sqrt(var_pred)
        
        # Grados de libertad
        dof = fit_result.dof
        t_crit = stats.t.ppf((1 + confidence_level) / 2, dof)
        
        y_pred = fit_result.predict(X_pred)[0]
        margin = t_crit * se_pred
        ci = (y_pred - margin, y_pred + margin)
        
        # Interpretación semántica
        ci_width = ci[1] - ci[0]
        relative_width = ci_width / abs(y_pred) if y_pred != 0 else float('inf')
        
        if relative_width < 0.1:
            reliability = "high"
            interp = f"Predicción muy precisa - intervalo de confianza es solo {relative_width*100:.1f}% del valor predicho"
        elif relative_width < 0.3:
            reliability = "moderate"
            interp = f"Predicción razonablemente precisa - intervalo de confianza es {relative_width*100:.1f}% del valor predicho"
        else:
            reliability = "low"
            interp = f"Predicción imprecisa - intervalo de confianza es {relative_width*100:.1f}% del valor predicho. Considera más datos"
        
        summary = (
            f"Predicción: {y_pred:.2f} ± {margin:.2f} "
            f"(IC {confidence_level*100:.0f}%: [{ci[0]:.2f}, {ci[1]:.2f}]). "
            f"{interp}."
        )
        
        return UncertaintyEstimate(
            point_estimate=y_pred,
            confidence_interval=ci,
            confidence_level=confidence_level,
            standard_error=se_pred,
            interpretation=interp,
            reliability=reliability,
            summary=summary
        )
    
    def _prediction_variance(self, X_pred: np.ndarray, fit: FitResult) -> float:
        """Calcula varianza de predicción: sigma^2 * (1 + x' (X'X)^-1 x)"""
        # Implementación estándar
        # var(y_pred) = sigma^2 * (1 + x_0' (X'X)^-1 x_0)
        # donde x_0 es el punto de predicción
        pass  # Implementación real necesita acceso a matriz de diseño
```

---

## PLAN DE IMPLEMENTACIÓN POR FASES

### FASE 0: Infraestructura Base (2 semanas)

**Objetivos**:
- Setup de repositorio y branches
- CI/CD para testing automático
- Estructura de módulos nuevos

**Entregables**:
```
doekit/
├── decision/__init__.py
├── semantic/__init__.py
├── memory/__init__.py
├── monitoring/__init__.py
└── integrations/__init__.py
```

**Branch**: `feature/infrastructure`

**Criterios de aceptación**:
- Todos los tests existentes pasan
- Nuevos módulos importables
- CI/CD configurado

---

### FASE 1: Capa Semántica Core (3-4 semanas)

**Objetivo**: Agregar interpretación semántica a funciones críticas

**Features**:

#### F1.1: SemanticResult wrapper

```python
# semantic/core.py
@dataclass
class SemanticResult:
    """Wrapper que agrega semántica a cualquier resultado"""
    numerical: Any  # Resultado numérico original
    interpretation: str
    reasoning: str
    context: str
    warnings: list[str]
    recommendations: list[str]
    confidence: str
    prompt_injection: str
    metadata: dict
    
    def to_dict(self) -> dict:
        """Serialización completa"""
        return {
            "numerical": self._serialize_numerical(),
            "semantic": {
                "interpretation": self.interpretation,
                "reasoning": self.reasoning,
                "context": self.context,
                "warnings": self.warnings,
                "recommendations": self.recommendations,
                "confidence": self.confidence
            },
            "prompt_injection": self.prompt_injection,
            "metadata": self.metadata
        }
    
    def _serialize_numerical(self) -> dict:
        """Convierte resultado numérico a dict serializable"""
        if hasattr(self.numerical, 'to_dict'):
            return self.numerical.to_dict()
        elif isinstance(self.numerical, dict):
            return self.numerical
        else:
            return {"value": str(self.numerical)}
```

#### F1.2: Interpretadores por función

```python
# semantic/interpreters.py

class RecommendationInterpreter(SemanticInterpreter):
    """Interpreta resultados de recommend_design()"""
    
    def interpret(self, recommendation: Recommendation) -> SemanticResult:
        method = recommendation.method
        n_runs = recommendation.design.n_runs
        d_eff = recommendation.table.iloc[0]['D_eff']
        
        interpretation = (
            f"Se recomienda {method} con {n_runs} corridas experimentales."
        )
        
        reasoning = self._build_reasoning(recommendation)
        context = self._build_context(recommendation)
        warnings = recommendation.caveats
        recommendations = self._extract_recommendations(recommendation)
        
        confidence = self._assess_confidence(recommendation)
        
        prompt = self._build_prompt(
            method, n_runs, d_eff, reasoning, context, warnings, recommendations
        )
        
        return SemanticResult(
            numerical=recommendation,
            interpretation=interpretation,
            reasoning=reasoning,
            context=context,
            warnings=warnings,
            recommendations=recommendations,
            confidence=confidence,
            prompt_injection=prompt,
            metadata={"function": "recommend_design"}
        )
    
    def _build_reasoning(self, rec: Recommendation) -> str:
        # Analizar tabla de alternativas
        table = rec.table
        winner = table.iloc[0]
        
        # Comparar con alternativas
        comparisons = []
        for i, row in table.iterrows():
            if i == 0:
                continue
            if row['runs'] > winner['runs']:
                comparisons.append(
                    f"{row['method']} requiere {row['runs']} corridas "
                    f"({row['runs'] - winner['runs']} más) pero solo mejora "
                    f"D-eff en {row['D_eff'] - winner['D_eff']:.1f}%"
                )
            elif not row['supports_model']:
                comparisons.append(
                    f"{row['method']} no soporta el modelo especificado"
                )
        
        reasoning = (
            f"{rec.method} fue seleccionado porque balancea óptimamente "
            f"eficiencia y presupuesto según las prioridades especificadas. "
        )
        
        if comparisons:
            reasoning += "Alternativas descartadas: " + "; ".join(comparisons[:2]) + "."
        
        return reasoning
    
    def _build_context(self, rec: Recommendation) -> str:
        scenario = rec.scenario
        return (
            f"Objetivo: {scenario['goal']}. "
            f"Factores: {scenario['n_factors']}. "
            f"Presupuesto: {scenario['budget']} corridas. "
            f"Modelo: {scenario['model_order']}."
        )
    
    def _extract_recommendations(self, rec: Recommendation) -> list[str]:
        recs = []
        
        # Basado en caveats, generar recomendaciones accionables
        if any("budget" in c.lower() for c in rec.caveats):
            recs.append(
                "Evaluar si es posible aumentar presupuesto para diseño más eficiente"
            )
        
        if any("sequential" in c.lower() or "wave" in c.lower() for c in rec.caveats):
            recs.append(
                "Después de primera wave, usar propose_next_runs() para refinamiento adaptativo"
            )
        
        return recs
    
    def _assess_confidence(self, rec: Recommendation) -> str:
        table = rec.table
        winner = table.iloc[0]
        
        if not winner.get('supports_model', True):
            return "Baja - el diseño recomendado no soporta completamente el modelo"
        
        # Si hay alternativas muy cercanas en score
        if len(table) > 1:
            second = table.iloc[1]
            if winner.get('D_eff') and second.get('D_eff'):
                gap = winner['D_eff'] - second['D_eff']
                if gap < 5:
                    return f"Moderada - alternativa {second['method']} es casi equivalente (gap {gap:.1f}%)"
        
        return "Alta - clara ventaja sobre alternativas"
    
    def _build_prompt(self, method, n_runs, d_eff, reasoning, context, warnings, recs):
        prompt = f"""
RECOMENDACIÓN DE DISEÑO EXPERIMENTAL:

Diseño recomendado: {method}
Corridas experimentales: {n_runs}
D-efficiency esperada: {d_eff:.1f}%

RAZONAMIENTO:
{reasoning}

CONTEXTO:
{context}

"""
        if warnings:
            prompt += "CONSIDERACIONES IMPORTANTES:\n"
            prompt += "\n".join(f"• {w}" for w in warnings[:3])
            prompt += "\n\n"
        
        if recs:
            prompt += "RECOMENDACIONES:\n"
            prompt += "\n".join(f"→ {r}" for r in recs)
            prompt += "\n"
        
        return prompt
```

#### F1.3: Decorador para agregar semántica automáticamente

```python
# semantic/decorators.py

from functools import wraps

def with_semantics(interpreter_class):
    """
    Decorador que agrega automáticamente capa semántica a función
    
    Usage:
        @with_semantics(RecommendationInterpreter)
        def recommend_design(...):
            # código existente
            return recommendation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ejecutar función original
            result = func(*args, **kwargs)
            
            # Si usuario pidió semántica (nuevo parámetro)
            include_semantics = kwargs.get('include_semantics', False)
            
            if not include_semantics:
                return result  # Backward compatible
            
            # Agregar interpretación semántica
            interpreter = interpreter_class()
            semantic_result = interpreter.interpret(result)
            
            return semantic_result
        
        return wrapper
    return decorator
```

**Ejemplo de uso**:

```python
# orchestration/advise/recommend.py

@with_semantics(RecommendationInterpreter)
def recommend_design(
    goal: str,
    factors,
    budget: Optional[int] = None,
    model_order: Optional[str] = None,
    priorities: Optional[dict] = None,
    include_semantics: bool = False,  # NUEVO parámetro
    **kwargs
) -> Union[Recommendation, SemanticResult]:
    # Código existente sin cambios
    # ...
    return Recommendation(...)  # El decorador lo envuelve si include_semantics=True
```

**Branch**: `feature/semantic-core`

**Tests**:
```python
# tests/test_semantic_interpreters.py

def test_recommendation_interpreter():
    rec = ed.recommend_design(
        goal="optimization",
        factors=3,
        budget=20,
        include_semantics=True
    )
    
    assert isinstance(rec, SemanticResult)
    assert rec.interpretation != ""
    assert rec.reasoning != ""
    assert len(rec.warnings) > 0
    assert rec.prompt_injection != ""
    
    # Validar que prompt es útil para LLM
    assert "RECOMENDACIÓN" in rec.prompt_injection
    assert str(rec.numerical.design.n_runs) in rec.prompt_injection

def test_backward_compatibility():
    """Asegurar que funciona sin include_semantics"""
    rec = ed.recommend_design(
        goal="optimization",
        factors=3,
        budget=20
    )
    
    assert isinstance(rec, Recommendation)
    assert not isinstance(rec, SemanticResult)
```

**Métricas de validación**:
- Cobertura de tests > 90%
- Todos los tests existentes pasan
- Performance overhead < 5%

---

### FASE 2: Motor de Decisión Autónomo (4-5 semanas)

**Objetivo**: Reemplazar decisiones booleanas con sistema de scoring y políticas configurables

#### F2.1: Sistema de scoring cuantificable

```python
# decision/scoring.py

@dataclass
class DecisionScore:
    """Score multidimensional para decisión"""
    # Componentes de score
    information_gain: float  # 0-1, esperado de D/A/G/I improvement
    cost: float              # 0-1, normalizado por presupuesto
    risk: float              # 0-1, basado en incertidumbre
    confidence: float        # 0-1, qué tan seguro estamos del score
    
    # Score compuesto
    composite: float  # Weighted combination
    
    # Semántica
    interpretation: str
    breakdown: dict  # Desglose de cómo se calculó
    
    def __gt__(self, other: 'DecisionScore') -> bool:
        return self.composite > other.composite

class ContinuationScorer:
    """Calcula score de continuar vs detener experimentación"""
    
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "information_gain": 0.4,
            "cost": 0.3,
            "risk": 0.2,
            "confidence": 0.1
        }
    
    def score_continuation(
        self,
        proposal: NextRunsProposal,
        context: DecisionContext
    ) -> DecisionScore:
        """
        Calcula score de continuar con experimentos propuestos
        """
        # Información ganada (basado en métricas de mejora)
        delta = proposal.comparison.delta
        d_gain = max(0, delta.get('D_efficiency', 0)) / 20  # Normalizar a 0-1
        g_gain = max(0, delta.get('G_efficiency', 0)) / 20
        spv_improve = max(0, -delta.get('spv_mean', 0)) / 1.0
        
        info_gain = (d_gain + g_gain + spv_improve) / 3
        
        # Costo (normalizado por presupuesto restante)
        extra_runs = proposal.added.n_runs
        remaining_budget = context.budget_total - context.budget_spent
        cost_normalized = 1 - (extra_runs / remaining_budget) if remaining_budget > 0 else 0
        
        # Riesgo (basado en incertidumbre de mejora)
        # Aquí iría cálculo de incertidumbre real
        # Por ahora, placeholder basado en DOF
        dof = proposal.combined.n_runs - len(proposal.combined.model.terms)
        risk = min(1.0, dof / 10)  # Más DOF = menos riesgo
        
        # Confianza (basado en múltiples indicadores)
        confidence = self._assess_confidence(proposal, context)
        
        # Score compuesto
        composite = (
            self.weights["information_gain"] * info_gain +
            self.weights["cost"] * cost_normalized +
            self.weights["risk"] * risk +
            self.weights["confidence"] * confidence
        )
        
        # Interpretación
        interpretation = self._interpret_score(
            composite, info_gain, cost_normalized, risk, confidence, context
        )
        
        breakdown = {
            "information_gain": {
                "score": info_gain,
                "weight": self.weights["information_gain"],
                "contribution": info_gain * self.weights["information_gain"],
                "details": {"D_gain": d_gain, "G_gain": g_gain, "SPV_improve": spv_improve}
            },
            "cost": {
                "score": cost_normalized,
                "weight": self.weights["cost"],
                "contribution": cost_normalized * self.weights["cost"],
                "details": {"extra_runs": extra_runs, "remaining_budget": remaining_budget}
            },
            "risk": {
                "score": risk,
                "weight": self.weights["risk"],
                "contribution": risk * self.weights["risk"],
                "details": {"dof": dof}
            },
            "confidence": {
                "score": confidence,
                "weight": self.weights["confidence"],
                "contribution": confidence * self.weights["confidence"]
            }
        }
        
        return DecisionScore(
            information_gain=info_gain,
            cost=cost_normalized,
            risk=risk,
            confidence=confidence,
            composite=composite,
            interpretation=interpretation,
            breakdown=breakdown
        )
    
    def _assess_confidence(self, proposal: NextRunsProposal, context: DecisionContext) -> float:
        """Evalúa confianza en la decisión"""
        confidence = 1.0
        
        # Reducir si R² del modelo actual es bajo
        if proposal.sigma_hat:
            current_fit = context.current_design.model  # Necesitaríamos acceso a fit
            # confidence *= current_fit.r_squared
        
        # Reducir si hay pocos DOF
        dof = proposal.combined.n_runs - len(proposal.combined.model.terms)
        if dof < 3:
            confidence *= 0.7
        
        # Aumentar si términos activos son consistentes entre waves
        # (necesitaría histórico)
        
        return max(0.1, min(1.0, confidence))
    
    def _interpret_score(self, composite, info, cost, risk, conf, context) -> str:
        if composite >= 0.7:
            decision = "fuertemente favorable"
        elif composite >= 0.5:
            decision = "favorable"
        elif composite >= 0.3:
            decision = "marginal"
        else:
            decision = "desfavorable"
        
        return (
            f"Score de continuación: {composite:.2f} ({decision}). "
            f"Información ganada ({info:.2f}) vs costo ({1-cost:.2f}) "
            f"con riesgo {1-risk:.2f} y confianza {conf:.2f}."
        )
```

#### F2.2: Políticas de decisión configurables

```python
# decision/policies.py

class DecisionPolicy(ABC):
    """Política base"""
    @abstractmethod
    def decide(self, context: DecisionContext) -> Decision:
        pass

class ThresholdPolicy(DecisionPolicy):
    """Política simple basada en umbral de score"""
    
    def __init__(
        self,
        continuation_threshold: float = 0.5,
        scorer: ContinuationScorer = None
    ):
        self.threshold = continuation_threshold
        self.scorer = scorer or ContinuationScorer()
    
    def decide(self, context: DecisionContext) -> Decision:
        # Generar propuesta
        proposal = ed.propose_next_runs(
            context.current_design,
            response=context.responses,
            n_add=context.get("n_add", 4)
        )
        
        # Calcular score
        score = self.scorer.score_continuation(proposal, context)
        
        # Decisión basada en umbral
        if score.composite >= self.threshold:
            action = "continue"
            reasoning = (
                f"Score de continuación ({score.composite:.2f}) supera umbral "
                f"({self.threshold:.2f}). {score.interpretation}"
            )
        else:
            action = "stop"
            reasoning = (
                f"Score de continuación ({score.composite:.2f}) bajo umbral "
                f"({self.threshold:.2f}). {score.interpretation}"
            )
        
        # Construir decisión completa con semántica
        return Decision(
            action=action,
            confidence=score.confidence,
            expected_gain=proposal.comparison.delta,
            cost={"runs": proposal.added.n_runs},
            interpretation=score.interpretation,
            reasoning=reasoning,
            context=self._build_context(context),
            warnings=self._generate_warnings(score, proposal),
            recommendations=self._generate_recommendations(action, score, proposal),
            confidence_explanation=self._explain_confidence(score),
            prompt_injection=self._build_prompt(action, score, proposal, context),
            alternatives=self._generate_alternatives(score, proposal, context)
        )
    
    def _build_context(self, ctx: DecisionContext) -> str:
        return (
            f"Wave {ctx.wave_number}, "
            f"presupuesto {ctx.budget_spent}/{ctx.budget_total} "
            f"({ctx.budget_spent/ctx.budget_total*100:.0f}% usado), "
            f"tolerancia al riesgo: {ctx.risk_tolerance}."
        )
    
    def _generate_warnings(self, score: DecisionScore, proposal: NextRunsProposal) -> list[str]:
        warnings = []
        
        if score.risk < 0.5:
            warnings.append(
                "Alto riesgo detectado - baja confianza en las mejoras estimadas. "
                "Considera validar supuestos del modelo antes de proceder."
            )
        
        if score.cost < 0.3:
            warnings.append(
                f"Costo significativo - {proposal.added.n_runs} corridas consumen "
                f"gran parte del presupuesto restante."
            )
        
        if len(proposal.active_terms) == 0:
            warnings.append(
                "No se detectaron términos activos significativos en wave anterior. "
                "Posible sobreajuste o modelo mal especificado."
            )
        
        return warnings
    
    def _generate_recommendations(
        self,
        action: str,
        score: DecisionScore,
        proposal: NextRunsProposal
    ) -> list[str]:
        recs = []
        
        if action == "continue":
            recs.append(
                f"Ejecutar {proposal.added.n_runs} experimentos propuestos."
            )
            
            if score.composite < 0.7:
                recs.append(
                    "Score marginal - monitorear de cerca resultados y re-evaluar "
                    "después de esta wave."
                )
        else:  # stop
            recs.append(
                "Detener experimentación exploratoria."
            )
            recs.append(
                "Realizar 2-3 corridas de confirmación en punto óptimo predicho."
            )
            
            if score.information_gain > 0.3:
                recs.append(
                    "Aunque score total es bajo, información ganada potencial es moderada. "
                    "Si presupuesto lo permite, considerar una wave final reducida."
                )
        
        return recs
    
    def _explain_confidence(self, score: DecisionScore) -> str:
        conf = score.confidence
        
        if conf >= 0.8:
            return "Alta confianza - múltiples indicadores concuerdan."
        elif conf >= 0.6:
            return "Confianza moderada - algunos factores de incertidumbre presentes."
        else:
            return "Baja confianza - alta incertidumbre en estimaciones. Proceder con cautela."
    
    def _build_prompt(
        self,
        action: str,
        score: DecisionScore,
        proposal: NextRunsProposal,
        context: DecisionContext
    ) -> str:
        prompt = f"""
DECISIÓN AUTÓNOMA DE EXPERIMENTACIÓN:

Acción recomendada: {action.upper()}
Score de decisión: {score.composite:.2f} (umbral: {self.threshold:.2f})
Confianza: {score.confidence:.2f}

DESGLOSE DE SCORE:
{self._format_score_breakdown(score.breakdown)}

RAZONAMIENTO:
{score.interpretation}

CONTEXTO:
{self._build_context(context)}

"""
        if len(score.breakdown) > 0:
            prompt += self._format_warnings_and_recs(
                self._generate_warnings(score, proposal),
                self._generate_recommendations(action, score, proposal)
            )
        
        return prompt
    
    def _format_score_breakdown(self, breakdown: dict) -> str:
        lines = []
        for component, data in breakdown.items():
            lines.append(
                f"• {component}: {data['score']:.2f} "
                f"(peso {data['weight']:.2f}, contribución {data['contribution']:.2f})"
            )
        return "\n".join(lines)
    
    def _format_warnings_and_recs(self, warnings: list[str], recs: list[str]) -> str:
        text = ""
        if warnings:
            text += "ADVERTENCIAS:\n"
            text += "\n".join(f"⚠ {w}" for w in warnings)
            text += "\n\n"
        if recs:
            text += "RECOMENDACIONES:\n"
            text += "\n".join(f"→ {r}" for r in recs)
            text += "\n"
        return text
    
    def _generate_alternatives(
        self,
        score: DecisionScore,
        proposal: NextRunsProposal,
        context: DecisionContext
    ) -> list[Decision]:
        """Genera decisiones alternativas con diferentes umbrales"""
        alternatives = []
        
        # Alternativa conservadora (umbral más alto)
        if score.composite >= 0.7:
            alt_policy = ThresholdPolicy(continuation_threshold=0.7)
            alt_decision = alt_policy.decide(context)
            alt_decision.interpretation = "Alternativa conservadora (umbral 0.7)"
            alternatives.append(alt_decision)
        
        # Alternativa agresiva (umbral más bajo)
        if score.composite >= 0.3:
            alt_policy = ThresholdPolicy(continuation_threshold=0.3)
            alt_decision = alt_policy.decide(context)
            alt_decision.interpretation = "Alternativa agresiva (umbral 0.3)"
            alternatives.append(alt_decision)
        
        return alternatives
```

#### F2.3: API unificada de decisión

```python
# decision/__init__.py

def decide_next_action(
    design: Design,
    responses: np.ndarray,
    budget_total: int,
    budget_spent: int,
    wave_number: int = 1,
    policy: DecisionPolicy = None,
    risk_tolerance: str = "moderate",
    include_semantics: bool = True
) -> Decision:
    """
    API unificada para decisión autónoma de experimentación
    
    Esta función reemplaza el flujo manual de:
    1. propose_next_runs()
    2. Evaluar proposal.comparison.worth_it manualmente
    3. Decidir qué hacer
    
    Con un sistema automático que:
    1. Calcula scores cuantificables
    2. Aplica política de decisión
    3. Retorna decisión estructurada con semántica completa
    
    Args:
        design: Diseño experimental actual
        responses: Respuestas observadas
        budget_total: Presupuesto total de experimentos
        budget_spent: Experimentos ya ejecutados
        wave_number: Número de wave actual
        policy: Política de decisión (ThresholdPolicy por defecto)
        risk_tolerance: "conservative", "moderate", "aggressive"
        include_semantics: Si incluir capa semántica completa
    
    Returns:
        Decision con action, confidence, semantic fields, prompt_injection
    
    Example:
        >>> decision = ed.decide_next_action(
        ...     design=current_design,
        ...     responses=y,
        ...     budget_total=50,
        ...     budget_spent=12,
        ...     wave_number=2
        ... )
        >>> print(decision.action)  # "continue" or "stop"
        >>> print(decision.prompt_injection)  # Texto listo para LLM
        >>> if decision.action == "continue":
        ...     next_runs = decision.expected_gain["next_runs_matrix"]
        ...     execute_experiments(next_runs)
    """
    # Configurar política por defecto
    if policy is None:
        # Threshold varía según risk_tolerance
        thresholds = {
            "conservative": 0.7,
            "moderate": 0.5,
            "aggressive": 0.3
        }
        policy = ThresholdPolicy(
            continuation_threshold=thresholds.get(risk_tolerance, 0.5)
        )
    
    # Construir contexto
    context = DecisionContext(
        current_design=design,
        responses=responses,
        budget_total=budget_total,
        budget_spent=budget_spent,
        wave_number=wave_number,
        risk_tolerance=risk_tolerance,
        objective="balanced",  # Podría ser parámetro
        confidence_required=0.8  # Podría ser parámetro
    )
    
    # Tomar decisión
    decision = policy.decide(context)
    
    return decision
```

**Branch**: `feature/decision-engine`

**Tests**:
```python
# tests/test_decision_engine.py

def test_decision_with_clear_improvement():
    """Caso donde debe decidir CONTINUE"""
    design = ed.plackett_burman(5)
    y = np.random.randn(design.n_runs) + np.arange(design.n_runs)  # Tendencia clara
    
    decision = ed.decide_next_action(
        design=design,
        responses=y,
        budget_total=50,
        budget_spent=8,
        wave_number=1,
        risk_tolerance="moderate"
    )
    
    assert decision.action == "continue"
    assert decision.confidence > 0.5
    assert "continue" in decision.prompt_injection.lower()
    assert len(decision.recommendations) > 0

def test_decision_with_no_improvement():
    """Caso donde debe decidir STOP"""
    design = ed.plackett_burman(5)
    y = np.random.randn(design.n_runs) * 0.1  # Puro ruido
    
    decision = ed.decide_next_action(
        design=design,
        responses=y,
        budget_total=50,
        budget_spent=40,  # Casi agotado
        wave_number=5,
        risk_tolerance="conservative"
    )
    
    assert decision.action == "stop"
    assert "stop" in decision.prompt_injection.lower()

def test_semantic_output_completeness():
    """Validar que salida semántica está completa"""
    design = ed.plackett_burman(5)
    y = np.random.randn(design.n_runs)
    
    decision = ed.decide_next_action(
        design=design,
        responses=y,
        budget_total=50,
        budget_spent=10
    )
    
    # Validar estructura
    assert decision.action in ["continue", "stop", "refine_model"]
    assert 0 <= decision.confidence <= 1
    assert decision.interpretation != ""
    assert decision.reasoning != ""
    assert decision.context != ""
    assert isinstance(decision.warnings, list)
    assert isinstance(decision.recommendations, list)
    assert decision.prompt_injection != ""
    
    # Validar que prompt es útil para LLM
    assert "DECISIÓN" in decision.prompt_injection
    assert decision.action.upper() in decision.prompt_injection
    assert str(decision.confidence) in decision.prompt_injection
```

**Métricas de validación**:
- API `decide_next_action()` funciona en 100% de casos de test
- Decisiones son reproducibles (mismo input → mismo output)
- Prompt injection contiene toda información relevante
- Performance: < 100ms overhead vs `propose_next_runs()` actual

---

### FASE 3: Gestión de Incertidumbre (3-4 semanas)

**Objetivo**: Cuantificar y propagar incertidumbre a través de decisiones

#### F3.1: Intervalos de confianza en predicciones

```python
# decision/uncertainty.py

class PredictionWithUncertainty:
    """Predicción con intervalo de confianza"""
    
    def predict_with_uncertainty(
        self,
        fit: FitResult,
        X_pred: np.ndarray,
        confidence_level: float = 0.95
    ) -> UncertaintyEstimate:
        """
        Predicción con intervalo de confianza
        
        Retorna predicción puntual + intervalo + interpretación semántica
        """
        # Implementación estadística estándar
        # var(y_pred) = sigma^2 * (1 + x' (X'X)^-1 x)
        
        # ... cálculo numérico ...
        
        # Interpretación semántica
        # ... como en ejemplo anterior ...
        
        return UncertaintyEstimate(...)
```

#### F3.2: Probabilidad de mejora (Probability of Improvement)

```python
# decision/uncertainty.py

def probability_of_improvement(
    current_best: float,
    prediction_with_uncertainty: UncertaintyEstimate,
    minimize: bool = False
) -> dict:
    """
    Calcula probabilidad de que nueva predicción mejore mejor valor actual
    
    Returns:
        {
            "probability": float,  # 0-1
            "interpretation": str,
            "recommendation": str
        }
    """
    pred = prediction_with_uncertainty
    mu = pred.point_estimate
    se = pred.standard_error
    
    if minimize:
        z = (current_best - mu) / se
    else:
        z = (mu - current_best) / se
    
    prob = stats.norm.cdf(z)
    
    if prob > 0.8:
        interpretation = f"Alta probabilidad ({prob:.1%}) de mejora sobre mejor valor actual"
        recommendation = "Muy recomendable explorar este punto"
    elif prob > 0.6:
        interpretation = f"Probabilidad moderada ({prob:.1%}) de mejora"
        recommendation = "Considerar explorar si presupuesto lo permite"
    else:
        interpretation = f"Baja probabilidad ({prob:.1%}) de mejora"
        recommendation = "No prioritario - enfocar en regiones más prometedoras"
    
    return {
        "probability": prob,
        "z_score": z,
        "interpretation": interpretation,
        "recommendation": recommendation
    }
```

#### F3.3: Expected Improvement (EI)

```python
# decision/uncertainty.py

def expected_improvement(
    current_best: float,
    prediction_with_uncertainty: UncertaintyEstimate,
    minimize: bool = False,
    xi: float = 0.01
) -> dict:
    """
    Expected Improvement - criterio estándar en Bayesian Optimization
    
    EI combina:
    - Magnitud de mejora esperada
    - Probabilidad de mejora
    - Exploración (xi parameter)
    
    Args:
        current_best: Mejor valor observado hasta ahora
        prediction_with_uncertainty: Predicción con incertidumbre
        minimize: Si True, minimizar; si False, maximizar
        xi: Parámetro de exploración (0 = explotación pura)
    
    Returns:
        {
            "ei": float,  # Valor de EI
            "interpretation": str,
            "recommendation": str
        }
    """
    pred = prediction_with_uncertainty
    mu = pred.point_estimate
    sigma = pred.standard_error
    
    if sigma == 0:
        return {
            "ei": 0.0,
            "interpretation": "Sin incertidumbre - EI es cero",
            "recommendation": "Este punto no aportará información nueva"
        }
    
    if minimize:
        imp = current_best - mu - xi
    else:
        imp = mu - current_best - xi
    
    Z = imp / sigma
    ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
    
    if minimize:
        ei = max(0, ei)
    else:
        ei = max(0, ei)
    
    # Interpretación
    if ei > 1.0:
        interpretation = f"EI alto ({ei:.2f}) - punto muy prometedor"
        recommendation = "Alta prioridad para explorar"
    elif ei > 0.5:
        interpretation = f"EI moderado ({ei:.2f}) - potencial razonable"
        recommendation = "Considerar incluir en siguiente batch"
    elif ei > 0.1:
        interpretation = f"EI bajo ({ei:.2f}) - mejora marginal esperada"
        recommendation = "Baja prioridad - solo si hay capacidad extra"
    else:
        interpretation = f"EI muy bajo ({ei:.4f}) - mejora insignificante esperada"
        recommendation = "No explorar - enfocar en puntos con mayor EI"
    
    return {
        "ei": ei,
        "z_score": Z,
        "improvement": imp,
        "interpretation": interpretation,
        "recommendation": recommendation
    }
```

#### F3.4: Integración con propose_next_runs()

```python
# orchestration/sequential/propose.py

def propose_next_runs_with_uncertainty(
    design: Design,
    response=None,
    n_add: int = 4,
    model: Optional[Model] = None,
    criterion: str = "D",
    uncertainty_aware: bool = True,  # NUEVO
    acquisition_function: str = "EI",  # NUEVO: "EI", "PI", "UCB"
    **kwargs
) -> NextRunsProposal:
    """
    Versión extendida que considera incertidumbre en selección de puntos
    
    Si uncertainty_aware=True:
    1. Calcula candidatos como antes
    2. Para cada candidato, estima incertidumbre de predicción
    3. Aplica acquisition function (EI, PI, UCB)
    4. Selecciona puntos que maximizan acquisition
    
    Esto combina DoE clásico (candidatos bien distribuidos) con
    BO (selección inteligente considerando incertidumbre)
    """
    # Código existente de propose_next_runs
    # ...
    
    if not uncertainty_aware or response is None:
        # Comportamiento actual
        return original_propose_next_runs(...)
    
    # NUEVO: Selección uncertainty-aware
    from ..decision.uncertainty import (
        PredictionWithUncertainty,
        expected_improvement
    )
    
    predictor = PredictionWithUncertainty()
    fit = fit_linear_model(design, response, model=model)
    current_best = np.max(response) if acquisition_function != "minimize" else np.min(response)
    
    # Para cada candidato, calcular acquisition value
    acquisition_values = []
    for idx, row in candidates.matrix.iterrows():
        X_cand = row.values.reshape(1, -1)
        pred_uncertainty = predictor.predict_with_uncertainty(fit, X_cand)
        
        if acquisition_function == "EI":
            acq = expected_improvement(current_best, pred_uncertainty)
            acquisition_values.append(acq["ei"])
        elif acquisition_function == "PI":
            acq = probability_of_improvement(current_best, pred_uncertainty)
            acquisition_values.append(acq["probability"])
        # ... otros acquisition functions
    
    # Seleccionar top n_add por acquisition value
    top_indices = np.argsort(acquisition_values)[-n_add:]
    
    # Construir diseño aumentado
    # ...
    
    # Agregar información de incertidumbre a proposal
    proposal.uncertainty_info = {
        "acquisition_function": acquisition_function,
        "acquisition_values": [acquisition_values[i] for i in top_indices],
        "current_best": current_best,
        "expected_improvements": [...]
    }
    
    # Agregar semántica sobre incertidumbre
    proposal.semantic.uncertainty_interpretation = (
        f"Puntos seleccionados usando {acquisition_function} que balancea "
        f"explotación (predecir bien) vs exploración (reducir incertidumbre). "
        f"EI promedio de puntos seleccionados: {np.mean([...]):.3f}."
    )
    
    return proposal
```

**Branch**: `feature/uncertainty-quantification`

**Tests**:
```python
def test_prediction_uncertainty():
    design = ed.box_behnken({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)})
    y = np.random.randn(design.n_runs)
    fit = ed.fit_linear_model(design, y)
    
    predictor = PredictionWithUncertainty()
    X_pred = np.array([[0, 0, 0]])  # Centro
    
    unc = predictor.predict_with_uncertainty(fit, X_pred)
    
    assert unc.point_estimate is not None
    assert len(unc.confidence_interval) == 2
    assert unc.confidence_interval[0] < unc.point_estimate < unc.confidence_interval[1]
    assert unc.interpretation != ""

def test_expected_improvement():
    # Mock uncertainty estimate
    pred = UncertaintyEstimate(
        point_estimate=10.0,
        confidence_interval=(8.0, 12.0),
        confidence_level=0.95,
        standard_error=1.0,
        interpretation="...",
        reliability="high",
        summary="..."
    )
    
    current_best = 8.0
    
    ei_result = expected_improvement(current_best, pred, minimize=False)
    
    assert ei_result["ei"] > 0  # Debería tener EI positivo
    assert "interpretation" in ei_result
    assert "recommendation" in ei_result

def test_uncertainty_aware_propose():
    design = ed.plackett_burman(5)
    y = np.random.randn(design.n_runs) + 5
    
    proposal = ed.propose_next_runs_with_uncertainty(
        design,
        response=y,
        n_add=4,
        uncertainty_aware=True,
        acquisition_function="EI"
    )
    
    assert hasattr(proposal, 'uncertainty_info')
    assert len(proposal.uncertainty_info["acquisition_values"]) == 4
    assert proposal.semantic.uncertainty_interpretation != ""
```

---

### FASE 4: Actualización Incremental (3 semanas)

**Objetivo**: Permitir actualización experimento-por-experimento en lugar de batches

#### F4.1: IncrementalExperiment class

```python
# orchestration/incremental.py

class IncrementalExperiment:
    """
    Experimento que se actualiza incrementalmente
    
    En lugar de:
    1. Diseño completo → ejecutar todo → analizar → proponer siguiente batch
    
    Permite:
    1. Diseño inicial → ejecutar 1 → analizar → decidir siguiente 1 → ...
    
    Esto habilita:
    - Early stopping si convergencia detectada prematuramente
    - Adaptación dinámica basada en cada resultado individual
    - Mejor uso de presupuesto
    """
    
    def __init__(
        self,
        initial_design: Design,
        budget_total: int,
        decision_policy: DecisionPolicy = None,
        convergence_checker = None
    ):
        self.design = initial_design
        self.budget_total = budget_total
        self.budget_spent = initial_design.n_runs
        self.policy = decision_policy or ThresholdPolicy()
        self.convergence_checker = convergence_checker or DefaultConvergenceChecker()
        
        # Historia
        self.responses = []
        self.experiment_history = []
        self.decisions_history = []
        
        # Estado
        self.converged = False
        self.convergence_wave = None
    
    def add_result(self, response: float) -> Decision:
        """
        Agrega resultado de un experimento y decide siguiente acción
        
        Returns:
            Decision con action="continue" y next_experiment, o action="stop"
        """
        self.responses.append(response)
        self.budget_spent += 1
        
        # Verificar convergencia
        conv_check = self.convergence_checker.check(
            design=self.design,
            responses=np.array(self.responses)
        )
        
        if conv_check["converged"]:
            self.converged = True
            self.convergence_wave = len(self.responses)
            
            decision = Decision(
                action="stop",
                confidence=conv_check["confidence"],
                expected_gain={},
                cost={},
                interpretation=conv_check["interpretation"],
                reasoning=conv_check["reasoning"],
                context=f"Convergencia detectada en experimento {self.budget_spent}",
                warnings=[],
                recommendations=conv_check["recommendations"],
                confidence_explanation=conv_check["confidence_explanation"],
                prompt_injection=conv_check["prompt_injection"],
                alternatives=[]
            )
            
            self.decisions_history.append(decision)
            return decision
        
        # Si no convergió, decidir siguiente experimento
        context = DecisionContext(
            current_design=self.design,
            responses=np.array(self.responses),
            budget_total=self.budget_total,
            budget_spent=self.budget_spent,
            wave_number=len(self.responses),
            risk_tolerance="moderate",
            objective="balanced",
            confidence_required=0.8
        )
        
        decision = self.policy.decide(context)
        self.decisions_history.append(decision)
        
        if decision.action == "continue":
            # Proponer siguiente experimento individual
            # Usaría uncertainty-aware selection si está disponible
            next_point = self._propose_next_single_point(context)
            decision.next_experiment = next_point
        
        return decision
    
    def _propose_next_single_point(self, context: DecisionContext) -> np.ndarray:
        """Propone siguiente punto individual usando EI u otro criterio"""
        # Podría usar propose_next_runs con n_add=1
        # O implementar selección más sofisticada
        proposal = ed.propose_next_runs_with_uncertainty(
            context.current_design,
            response=context.responses,
            n_add=1,
            uncertainty_aware=True,
            acquisition_function="EI"
        )
        
        return proposal.added.matrix.iloc[0].values
    
    def run_autonomous(
        self,
        experiment_executor: Callable[[np.ndarray], float],
        max_iterations: int = None
    ) -> dict:
        """
        Ejecuta bucle autónomo de experimentación
        
        Args:
            experiment_executor: Función que toma configuración y retorna resultado
            max_iterations: Máximo de iteraciones (None = hasta convergencia o presupuesto)
        
        Returns:
            {
                "converged": bool,
                "total_experiments": int,
                "best_result": float,
                "best_configuration": dict,
                "history": list[Decision]
            }
        """
        max_iter = max_iterations or (self.budget_total - self.budget_spent)
        
        for iteration in range(max_iter):
            if self.budget_spent >= self.budget_total:
                break
            
            if self.converged:
                break
            
            # Decidir siguiente experimento
            if len(self.responses) == 0:
                # Primera iteración - usar diseño inicial
                next_config = self.design.matrix.iloc[0].values
            else:
                # Decidir basándose en resultados previos
                decision = self.decisions_history[-1] if self.decisions_history else None
                if decision and decision.action == "stop":
                    break
                
                if decision and hasattr(decision, 'next_experiment'):
                    next_config = decision.next_experiment
                else:
                    # Fallback
                    context = DecisionContext(
                        current_design=self.design,
                        responses=np.array(self.responses),
                        budget_total=self.budget_total,
                        budget_spent=self.budget_spent,
                        wave_number=iteration,
                        risk_tolerance="moderate",
                        objective="balanced",
                        confidence_required=0.8
                    )
                    next_config = self._propose_next_single_point(context)
            
            # Ejecutar experimento
            result = experiment_executor(next_config)
            
            # Agregar resultado y decidir
            decision = self.add_result(result)
            
            # Logging semántico
            print(f"[Iteration {iteration+1}] {decision.interpretation}")
            
            if decision.action == "stop":
                print(f"Stopping: {decision.reasoning}")
                break
        
        # Resultado final
        best_idx = np.argmax(self.responses)
        
        return {
            "converged": self.converged,
            "convergence_iteration": self.convergence_wave,
            "total_experiments": self.budget_spent,
            "best_result": self.responses[best_idx],
            "best_configuration": self.design.matrix.iloc[best_idx].to_dict(),
            "history": self.decisions_history,
            "final_interpretation": self._generate_final_summary()
        }
    
    def _generate_final_summary(self) -> str:
        """Genera resumen semántico final del experimento"""
        if self.converged:
            summary = (
                f"Experimentación convergió en iteración {self.convergence_wave} "
                f"de {self.budget_total} presupuestadas. "
            )
        else:
            summary = (
                f"Experimentación completó {self.budget_spent} iteraciones "
                f"sin convergencia formal. "
            )
        
        best_idx = np.argmax(self.responses)
        summary += (
            f"Mejor resultado: {self.responses[best_idx]:.3f} "
            f"en iteración {best_idx+1}."
        )
        
        return summary
```

#### F4.2: Convergence Checker

```python
# monitoring/convergence.py

class ConvergenceChecker(ABC):
    @abstractmethod
    def check(self, design: Design, responses: np.ndarray) -> dict:
        """
        Verifica si experimentación ha convergido
        
        Returns:
            {
                "converged": bool,
                "confidence": float,
                "interpretation": str,
                "reasoning": str,
                "recommendations": list[str],
                "confidence_explanation": str,
                "prompt_injection": str
            }
        """
        pass

class DefaultConvergenceChecker(ConvergenceChecker):
    """Convergencia basada en múltiples criterios"""
    
    def __init__(
        self,
        min_iterations: int = 5,
        improvement_threshold: float = 0.01,
        consecutive_no_improve: int = 3,
        r_squared_threshold: float = 0.90
    ):
        self.min_iter = min_iterations
        self.improvement_threshold = improvement_threshold
        self.consecutive_threshold = consecutive_no_improve
        self.r_squared_threshold = r_squared_threshold
    
    def check(self, design: Design, responses: np.ndarray) -> dict:
        n = len(responses)
        
        # Criterio 1: Mínimo de iteraciones
        if n < self.min_iter:
            return self._not_converged(
                "Insuficientes iteraciones",
                f"Solo {n}/{self.min_iter} mínimas completadas"
            )
        
        # Criterio 2: Mejora marginal en últimas iteraciones
        recent_improvements = self._calculate_recent_improvements(responses)
        consecutive_low = sum(
            1 for imp in recent_improvements[-self.consecutive_threshold:]
            if imp < self.improvement_threshold
        )
        
        marginal_improve = consecutive_low >= self.consecutive_threshold
        
        # Criterio 3: R² del modelo
        try:
            fit = ed.fit_linear_model(design, responses)
            high_r_squared = fit.r_squared >= self.r_squared_threshold
        except:
            high_r_squared = False
        
        # Criterio 4: Estabilidad de coeficientes (si hay suficientes datos)
        stable_coefficients = False
        if n >= 10:
            stable_coefficients = self._check_coefficient_stability(design, responses)
        
        # Decisión
        converged = marginal_improve and (high_r_squared or stable_coefficients)
        
        if converged:
            return self._converged(
                recent_improvements=recent_improvements,
                r_squared=fit.r_squared if high_r_squared else None,
                n_iterations=n
            )
        else:
            return self._not_converged(
                "Convergencia no alcanzada",
                self._explain_why_not_converged(
                    marginal_improve,
                    high_r_squared,
                    stable_coefficients,
                    recent_improvements
                )
            )
    
    def _calculate_recent_improvements(self, responses: np.ndarray) -> list[float]:
        """Calcula mejora relativa entre iteraciones consecutivas"""
        improvements = []
        for i in range(1, len(responses)):
            best_so_far = np.max(responses[:i])
            current = responses[i]
            improvement = (current - best_so_far) / abs(best_so_far) if best_so_far != 0 else 0
            improvements.append(max(0, improvement))  # Solo mejoras positivas
        return improvements
    
    def _check_coefficient_stability(self, design: Design, responses: np.ndarray) -> bool:
        """Verifica si coeficientes son estables en ventana deslizante"""
        # Comparar coeficientes de últimas N iteraciones vs N-k anteriores
        # Si variación < threshold, considerado estable
        # ... implementación ...
        return False  # Placeholder
    
    def _converged(self, recent_improvements, r_squared, n_iterations) -> dict:
        interpretation = f"Sistema convergido después de {n_iterations} iteraciones"
        
        reasoning = (
            f"Últimas {self.consecutive_threshold} iteraciones muestran mejora "
            f"marginal < {self.improvement_threshold*100:.1f}%. "
        )
        
        if r_squared:
            reasoning += f"R² actual ({r_squared:.3f}) supera umbral ({self.r_squared_threshold})."
        
        recommendations = [
            "Detener experimentación exploratoria",
            "Ejecutar 2-3 réplicas en punto óptimo para confirmar",
            "Documentar modelo final y limitaciones"
        ]
        
        prompt = f"""
CONVERGENCIA DETECTADA:

Estado: CONVERGIDO
Iteraciones: {n_iterations}
Confianza: Alta

Criterios satisfechos:
✓ Mejora marginal últimas {self.consecutive_threshold} iteraciones < {self.improvement_threshold*100:.1f}%
{'✓ R² = ' + f'{r_squared:.3f}' + f' > {self.r_squared_threshold}' if r_squared else ''}

RAZONAMIENTO:
{reasoning}

RECOMENDACIONES:
{chr(10).join('→ ' + r for r in recommendations)}
"""
        
        return {
            "converged": True,
            "confidence": 0.9,
            "interpretation": interpretation,
            "reasoning": reasoning,
            "recommendations": recommendations,
            "confidence_explanation": "Alta - múltiples criterios concuerdan",
            "prompt_injection": prompt
        }
    
    def _not_converged(self, reason: str, details: str) -> dict:
        return {
            "converged": False,
            "confidence": 0.0,
            "interpretation": f"No convergido: {reason}",
            "reasoning": details,
            "recommendations": ["Continuar experimentación"],
            "confidence_explanation": "",
            "prompt_injection": f"NO CONVERGIDO: {reason}. {details}"
        }
    
    def _explain_why_not_converged(
        self,
        marginal_improve: bool,
        high_r_squared: bool,
        stable_coef: bool,
        improvements: list[float]
    ) -> str:
        reasons = []
        
        if not marginal_improve:
            recent_avg = np.mean(improvements[-3:]) if len(improvements) >= 3 else 0
            reasons.append(
                f"Mejoras recientes ({recent_avg*100:.2f}%) aún superan umbral "
                f"({self.improvement_threshold*100:.1f}%)"
            )
        
        if not high_r_squared:
            reasons.append("R² del modelo aún bajo")
        
        if not stable_coef:
            reasons.append("Coeficientes aún inestables")
        
        return "; ".join(reasons)
```

**Branch**: `feature/incremental-updates`

**Tests**:
```python
def test_incremental_experiment():
    initial_design = ed.plackett_burman(5)
    
    exp = IncrementalExperiment(
        initial_design=initial_design,
        budget_total=50
    )
    
    # Simular adición de resultados
    for i in range(10):
        result = np.random.randn() + 5
        decision = exp.add_result(result)
        
        if decision.action == "stop":
            break
    
    assert len(exp.responses) == len(exp.decisions_history)
    assert exp.budget_spent == initial_design.n_runs + len(exp.responses)

def test_autonomous_run():
    initial_design = ed.plackett_burman(3)
    
    def mock_executor(config):
        """Función objetivo simulada"""
        return -np.sum(config**2) + np.random.randn() * 0.1
    
    exp = IncrementalExperiment(
        initial_design=initial_design,
        budget_total=30
    )
    
    results = exp.run_autonomous(
        experiment_executor=mock_executor,
        max_iterations=20
    )
    
    assert results["converged"] or results["total_experiments"] == 20
    assert "best_result" in results
    assert "final_interpretation" in results
    assert results["final_interpretation"] != ""
```

---

### FASE 5: Meta-Aprendizaje y Memoria (4 semanas)

**Objetivo**: Aprovechar experimentos previos para informar decisiones futuras

#### F5.1: Experiment Store

```python
# memory/experiment_store.py

from dataclasses import dataclass, asdict
from typing import List, Optional
import json
from pathlib import Path

@dataclass
class ExperimentRecord:
    """Registro completo de un experimento"""
    id: str
    timestamp: str
    domain: str  # e.g., "chemical_process", "ml_hyperparams"
    factors: dict
    design_method: str
    n_runs: int
    responses: dict  # Múltiples respuestas
    model_type: str
    final_r_squared: dict  # Por respuesta
    best_configuration: dict
    best_response_values: dict
    convergence_info: dict
    metadata: dict
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentRecord':
        return cls(**data)

class ExperimentStore:
    """Almacén persistente de experimentos previos"""
    
    def __init__(self, storage_path: str = "~/.doekit/experiments"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, record: ExperimentRecord):
        """Guardar experimento"""
        file_path = self.storage_path / f"{record.id}.json"
        with open(file_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
    
    def load(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Cargar experimento por ID"""
        file_path = self.storage_path / f"{experiment_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return ExperimentRecord.from_dict(data)
    
    def search(
        self,
        domain: Optional[str] = None,
        factor_names: Optional[List[str]] = None,
        design_method: Optional[str] = None,
        min_r_squared: Optional[float] = None,
        limit: int = 10
    ) -> List[ExperimentRecord]:
        """
        Buscar experimentos similares
        
        Args:
            domain: Filtrar por dominio
            factor_names: Filtrar por factores similares
            design_method: Filtrar por método de diseño
            min_r_squared: R² mínimo
            limit: Máximo de resultados
        
        Returns:
            Lista de experimentos ordenados por similaridad/calidad
        """
        results = []
        
        for file_path in self.storage_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            record = ExperimentRecord.from_dict(data)
            
            # Filtros
            if domain and record.domain != domain:
                continue
            
            if factor_names:
                record_factors = set(record.factors.keys())
                query_factors = set(factor_names)
                similarity = len(record_factors & query_factors) / len(query_factors)
                if similarity < 0.5:  # Al menos 50% de factores en común
                    continue
            
            if design_method and record.design_method != design_method:
                continue
            
            if min_r_squared:
                avg_r2 = np.mean(list(record.final_r_squared.values()))
                if avg_r2 < min_r_squared:
                    continue
            
            results.append(record)
        
        # Ordenar por calidad (R² promedio)
        results.sort(
            key=lambda r: np.mean(list(r.final_r_squared.values())),
            reverse=True
        )
        
        return results[:limit]
    
    def get_statistics(self, domain: Optional[str] = None) -> dict:
        """Estadísticas del almacén"""
        records = self.search(domain=domain, limit=1000)
        
        if not records:
            return {"total_experiments": 0}
        
        return {
            "total_experiments": len(records),
            "domains": list(set(r.domain for r in records)),
            "design_methods": list(set(r.design_method for r in records)),
            "avg_r_squared": np.mean([
                np.mean(list(r.final_r_squared.values()))
                for r in records
            ]),
            "avg_runs": np.mean([r.n_runs for r in records]),
            "most_common_factors": self._most_common_factors(records)
        }
    
    def _most_common_factors(self, records: List[ExperimentRecord]) -> dict:
        """Factores más comunes en experimentos"""
        factor_counts = {}
        for record in records:
            for factor in record.factors.keys():
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return dict(sorted(
            factor_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
```

#### F5.2: Transfer Learning para Priors

```python
# memory/transfer.py

class PriorLearner:
    """Aprende priors de experimentos históricos"""
    
    def __init__(self, experiment_store: ExperimentStore):
        self.store = experiment_store
    
    def learn_priors(
        self,
        domain: str,
        factor_names: List[str],
        response_name: str
    ) -> dict:
        """
        Aprende distribuciones prior de experimentos similares
        
        Returns:
            {
                "factor_effects": dict,  # Prior sobre magnitud de efectos
                "sigma_prior": dict,     # Prior sobre varianza residual
                "interaction_likelihood": dict,  # Probabilidad de interacciones
                "interpretation": str
            }
        """
        # Buscar experimentos similares
        similar = self.store.search(
            domain=domain,
            factor_names=factor_names,
            min_r_squared=0.7,
            limit=20
        )
        
        if len(similar) < 3:
            return {
                "factor_effects": {},
                "sigma_prior": {},
                "interaction_likelihood": {},
                "interpretation": (
                    "Insuficientes experimentos previos similares. "
                    "Usando priors no informativos."
                )
            }
        
        # Extraer efectos de factores
        factor_effects = {}
        for factor in factor_names:
            effects = []
            for exp in similar:
                if factor in exp.factors:
                    # Extraer coeficiente de este factor
                    # (necesitaríamos guardarlo en ExperimentRecord)
                    pass
            
            if effects:
                factor_effects[factor] = {
                    "mean": np.mean(effects),
                    "std": np.std(effects),
                    "distribution": "normal"
                }
        
        # Sigma prior
        sigmas = []
        for exp in similar:
            if response_name in exp.final_r_squared:
                # Estimar sigma de R²
                pass
        
        sigma_prior = {
            "mean": np.mean(sigmas) if sigmas else 1.0,
            "std": np.std(sigmas) if sigmas else 0.5
        }
        
        # Interacciones
        # Contar qué % de experimentos anteriores encontraron interacciones significativas
        interaction_count = {}
        for exp in similar:
            # Analizar qué interacciones fueron significativas
            pass
        
        interpretation = (
            f"Priors aprendidos de {len(similar)} experimentos similares en dominio '{domain}'. "
            f"Factores con priors informativos: {list(factor_effects.keys())}. "
        )
        
        return {
            "factor_effects": factor_effects,
            "sigma_prior": sigma_prior,
            "interaction_likelihood": interaction_count,
            "interpretation": interpretation,
            "n_similar_experiments": len(similar)
        }
```

#### F5.3: Recomendaciones basadas en historia

```python
# memory/recommendations.py

class HistoricalRecommender:
    """Genera recomendaciones basadas en experimentos similares previos"""
    
    def __init__(self, experiment_store: ExperimentStore):
        self.store = experiment_store
    
    def recommend_design_from_history(
        self,
        domain: str,
        factor_names: List[str],
        budget: int
    ) -> dict:
        """
        Recomienda diseño basándose en qué funcionó antes
        
        Returns:
            {
                "recommended_method": str,
                "confidence": float,
                "reasoning": str,
                "supporting_experiments": list[str],  # IDs
                "caveats": list[str]
            }
        """
        similar = self.store.search(
            domain=domain,
            factor_names=factor_names,
            limit=50
        )
        
        if len(similar) < 5:
            return {
                "recommended_method": None,
                "confidence": 0.0,
                "reasoning": "Insuficiente historia de experimentos similares",
                "supporting_experiments": [],
                "caveats": ["Usar recommend_design() estándar"]
            }
        
        # Analizar qué métodos funcionaron mejor
        method_performance = {}
        for exp in similar:
            method = exp.design_method
            avg_r2 = np.mean(list(exp.final_r_squared.values()))
            
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(avg_r2)
        
        # Encontrar mejor método
        best_method = None
        best_avg_r2 = 0
        for method, r2_values in method_performance.items():
            avg = np.mean(r2_values)
            if avg > best_avg_r2:
                best_avg_r2 = avg
                best_method = method
        
        # Confianza basada en consistencia
        best_std = np.std(method_performance[best_method])
        confidence = max(0.3, 1.0 - best_std)  # Menor std = mayor confianza
        
        reasoning = (
            f"{best_method} mostró mejor performance promedio (R²={best_avg_r2:.3f}) "
            f"en {len(method_performance[best_method])} experimentos similares previos. "
        )
        
        supporting_ids = [
            exp.id for exp in similar
            if exp.design_method == best_method
        ][:5]
        
        caveats = []
        if len(similar) < 20:
            caveats.append(
                f"Basado en solo {len(similar)} experimentos - confianza limitada"
            )
        
        if best_std > 0.1:
            caveats.append(
                f"Alta variabilidad en resultados previos (std={best_std:.3f})"
            )
        
        return {
            "recommended_method": best_method,
            "confidence": confidence,
            "reasoning": reasoning,
            "supporting_experiments": supporting_ids,
            "caveats": caveats
        }
    
    def suggest_factors_to_investigate(
        self,
        domain: str,
        current_factors: List[str]
    ) -> dict:
        """
        Sugiere factores adicionales basándose en experimentos exitosos previos
        
        Returns:
            {
                "suggested_factors": list[str],
                "reasoning": str,
                "supporting_evidence": dict
            }
        """
        similar = self.store.search(domain=domain, limit=100)
        
        # Encontrar factores que aparecen frecuentemente en experimentos exitosos
        successful_experiments = [
            exp for exp in similar
            if np.mean(list(exp.final_r_squared.values())) >= 0.8
        ]
        
        factor_frequency = {}
        for exp in successful_experiments:
            for factor in exp.factors.keys():
                if factor not in current_factors:
                    factor_frequency[factor] = factor_frequency.get(factor, 0) + 1
        
        # Ordenar por frecuencia
        suggested = sorted(
            factor_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if not suggested:
            return {
                "suggested_factors": [],
                "reasoning": "No hay factores adicionales recurrentes en experimentos exitosos",
                "supporting_evidence": {}
            }
        
        reasoning = (
            f"Los siguientes factores aparecen frecuentemente en experimentos exitosos "
            f"similares y podrían ser relevantes: "
            f"{', '.join(f for f, _ in suggested)}"
        )
        
        evidence = {
            factor: {
                "frequency": count,
                "percentage": count / len(successful_experiments) * 100
            }
            for factor, count in suggested
        }
        
        return {
            "suggested_factors": [f for f, _ in suggested],
            "reasoning": reasoning,
            "supporting_evidence": evidence
        }
```

**Branch**: `feature/meta-learning`

---

### FASE 6: Integración Real con BO (3-4 semanas)

**Objetivo**: Integración nativa con frameworks de Bayesian Optimization

#### F6.1: Integración con BoTorch

```python
# integrations/bayesian_opt.py

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False

class BoTorchIntegration:
    """Integración real con BoTorch para BO avanzado"""
    
    def __init__(self):
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch no instalado. pip install botorch")
    
    def fit_gp_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """Ajusta modelo Gaussian Process"""
        X_torch = torch.tensor(X_train, dtype=torch.float64)
        y_torch = torch.tensor(y_train, dtype=torch.float64).unsqueeze(-1)
        
        gp = SingleTaskGP(X_torch, y_torch)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        
        # Optimizar hiperparámetros
        from botorch.fit import fit_gpytorch_model
        fit_gpytorch_model(mll)
        
        return gp
    
    def propose_next_with_ei(
        self,
        gp_model,
        bounds: np.ndarray,
        current_best: float,
        n_candidates: int = 1
    ) -> np.ndarray:
        """Propone siguientes puntos usando Expected Improvement"""
        EI = ExpectedImprovement(model=gp_model, best_f=current_best)
        
        bounds_torch = torch.tensor(bounds, dtype=torch.float64).T
        
        candidates, acq_value = optimize_acqf(
            acq_function=EI,
            bounds=bounds_torch,
            q=n_candidates,
            num_restarts=10,
            raw_samples=512
        )
        
        return candidates.detach().numpy()
```

**Branch**: `feature/botorch-integration`

---

## BRANCHES Y WORKFLOW DE DESARROLLO

### Estrategia de Branches

```
main
├── develop
│   ├── feature/infrastructure (Fase 0)
│   ├── feature/semantic-core (Fase 1)
│   ├── feature/decision-engine (Fase 2)
│   ├── feature/uncertainty-quantification (Fase 3)
│   ├── feature/incremental-updates (Fase 4)
│   ├── feature/meta-learning (Fase 5)
│   └── feature/botorch-integration (Fase 6)
└── hotfix/*
```

### Workflow

1. **Feature Branch**: Desarrollo de feature en branch dedicado
2. **PR a develop**: Code review + tests automáticos
3. **Integration testing**: Tests de integración en develop
4. **Release branch**: Cuando varias features están listas
5. **PR a main**: Release final

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,test]"
      
      - name: Run tests
        run: |
          pytest tests/ --cov=doekit --cov-report=xml
      
      - name: Check coverage
        run: |
          coverage report --fail-under=90
      
      - name: Type checking
        run: |
          mypy doekit/
      
      - name: Lint
        run: |
          flake8 doekit/
          black --check doekit/
```

---

## MÉTRICAS DE VALIDACIÓN

### Por Fase

#### Fase 1: Semántica
- [ ] 100% de funciones críticas tienen interpretación semántica
- [ ] Prompt injection contiene toda info relevante (validación manual)
- [ ] Backward compatibility: 0 tests rotos
- [ ] Performance overhead < 5%

#### Fase 2: Decisión
- [ ] API `decide_next_action()` funciona en 100% casos de test
- [ ] Decisiones reproducibles (deterministic dado seed)
- [ ] Score compuesto predice "worth_it" con accuracy > 85%
- [ ] Interpretaciones semánticas son claras (evaluación humana)

#### Fase 3: Incertidumbre
- [ ] Intervalos de confianza cubren verdadero valor en 95% de casos simulados
- [ ] EI/PI correlacionan con mejora real (validación empírica)
- [ ] Uncertainty-aware selection supera D-optimal en simulaciones

#### Fase 4: Incremental
- [ ] Early stopping detecta convergencia correctamente (< 5% false positives)
- [ ] Uso de presupuesto es eficiente (vs batch baseline)
- [ ] Autonomous run completa sin intervención

#### Fase 5: Meta-learning
- [ ] Priors aprendidos mejoran convergencia (medido en simulaciones)
- [ ] Recomendaciones históricas > 70% accuracy
- [ ] Transfer learning reduce runs necesarios en > 20%

#### Fase 6: BO Integration
- [ ] Resultados iguales/mejores que BoTorch standalone
- [ ] API unificada funciona con/sin BoTorch instalado

### Métricas Globales

- **Cobertura de tests**: > 90%
- **Documentación**: 100% de funciones públicas documentadas
- **Performance**: Overhead total < 10% vs doekit actual
- **Usabilidad**: Agente autónomo completa 5/5 tareas de benchmark sin intervención

---

## EVALUACIÓN DE SUFICIENCIA

### ¿Es Suficiente con Estas Mejoras?

**Para un agente semi-autónomo**: **SÍ**

Con estas mejoras, un agente puede:
- ✅ Diseñar experimentos automáticamente
- ✅ Ejecutar bucle adaptativo con mínima supervisión
- ✅ Tomar decisiones cuantificables con explicaciones
- ✅ Manejar incertidumbre de forma robusta
- ✅ Aprender de experimentos previos
- ✅ Converger automáticamente

**Para un agente completamente autónomo**: **CASI**

Falta aún:

1. **Diagnóstico automático de problemas experimentales**
   - Detección de outliers
   - Identificación de errores de medición
   - Validación de supuestos del modelo

2. **Capacidad de reformular problemas**
   - Si modelo cuadrático no ajusta, intentar transformaciones
   - Si experimentos fallan sistemáticamente, revisar restricciones

3. **Integración con sistemas de ejecución**
   - Orquestación con laboratorios automatizados
   - Manejo de fallos de hardware
   - Queue management para experimentos en paralelo

4. **Razonamiento causal**
   - Inferir mecanismos, no solo correlaciones
   - Proponer experimentos para disambiguar causalidad

### Recomendación Final

**Implementar Fases 0-4 primero** (12-15 semanas):
- Son críticas para autonomía básica
- Habilitan uso inmediato por agentes
- ROI más alto

**Fases 5-6 como mejoras posteriores** (7-8 semanas):
- Agregan capacidades avanzadas
- Menos críticas para MVP
- Pueden desarrollarse en paralelo con adopción de Fases 1-4

**Total estimado**: 19-23 semanas (4.5-5.5 meses) para implementación completa.

---

## APÉNDICE: Ejemplo End-to-End

### Agente Autónomo Usando doekit Mejorado

```python
import doekit as ed

class AutonomousExperimentAgent:
    """Agente completamente autónomo de experimentación"""
    
    def optimize(
        self,
        domain: str,
        factors: dict,
        objective_function: Callable,
        budget: int = 50
    ):
        # 1. Consultar memoria de experimentos similares
        recommender = ed.memory.HistoricalRecommender()
        historical_rec = recommender.recommend_design_from_history(
            domain=domain,
            factor_names=list(factors.keys()),
            budget=budget
        )
        
        print(f"[Memoria] {historical_rec['reasoning']}")
        
        # 2. Diseño inicial informado por historia
        if historical_rec['confidence'] > 0.7:
            design_method = historical_rec['recommended_method']
        else:
            design_method = None  # Dejar que recommend_design decida
        
        rec = ed.recommend_design(
            goal="optimization",
            factors=factors,
            budget=min(budget // 3, 20),  # Usar 1/3 del presupuesto para diseño inicial
            include_semantics=True
        )
        
        # Imprimir razonamiento
        print(f"\n[Diseño Inicial]\n{rec.prompt_injection}")
        
        # 3. Experimentación incremental con decisión autónoma
        exp = ed.IncrementalExperiment(
            initial_design=rec.design,
            budget_total=budget,
            decision_policy=ed.ThresholdPolicy(continuation_threshold=0.5)
        )
        
        results = exp.run_autonomous(
            experiment_executor=objective_function,
            max_iterations=budget
        )
        
        print(f"\n[Resultado Final]\n{results['final_interpretation']}")
        
        # 4. Guardar en memoria para futuros experimentos
        record = ed.memory.ExperimentRecord(
            id=f"{domain}_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            domain=domain,
            factors=factors,
            design_method=rec.method,
            n_runs=results['total_experiments'],
            responses={"objective": results['best_result']},
            model_type="quadratic",
            final_r_squared={"objective": 0.95},  # Placeholder
            best_configuration=results['best_configuration'],
            best_response_values={"objective": results['best_result']},
            convergence_info={
                "converged": results['converged'],
                "iteration": results.get('convergence_iteration')
            },
            metadata={}
        )
        
        store = ed.memory.ExperimentStore()
        store.save(record)
        print("\n[Memoria] Experimento guardado para referencia futura")
        
        return results

# Uso
agent = AutonomousExperimentAgent()

def chemical_process(config):
    """Función objetivo simulada"""
    temp, press, ph = config
    return -(temp - 75)**2 - (press - 3)**2 - (ph - 7)**2 + np.random.randn() * 2

result = agent.optimize(
    domain="chemical_process",
    factors={"temperature": (50, 100), "pressure": (1, 5), "pH": (3, 9)},
    objective_function=chemical_process,
    budget=50
)

print(f"\n✓ Optimización completa: mejor resultado = {result['best_result']:.2f}")
print(f"✓ Configuración óptima: {result['best_configuration']}")
```

**Salida esperada** (con semántica completa):

```
[Memoria] Box-Behnken mostró mejor performance promedio (R²=0.923) en 12 experimentos similares previos.

[Diseño Inicial]
RECOMENDACIÓN DE DISEÑO EXPERIMENTAL:

Diseño recomendado: Box-Behnken
Corridas experimentales: 15
D-efficiency esperada: 88.2%

RAZONAMIENTO:
Box-Behnken fue seleccionado porque balancea óptimamente eficiencia y presupuesto según las prioridades especificadas. Alternativas descartadas: Central Composite requiere 20 corridas (5 más) pero solo mejora D-eff en 8.3%.

CONTEXTO:
Objetivo: optimization. Factores: 3. Presupuesto: 16 corridas. Modelo: quadratic.

[Iteration 1] Score de continuación: 0.72 (fuertemente favorable). Información ganada (0.65) vs costo (0.92) con riesgo 0.15 y confianza 0.82.
[Iteration 2] Score de continuación: 0.68 (favorable). Información ganada (0.61) vs costo (0.90) con riesgo 0.18 y confianza 0.85.
...
[Iteration 18] Stopping: Score de continuación (0.42) bajo umbral (0.50). Convergencia detectada - mejoras marginales últimas 3 iteraciones.

[Resultado Final]
Experimentación convergió en iteración 18 de 50 presupuestadas. Mejor resultado: 45.23 en iteración 16.

[Memoria] Experimento guardado para referencia futura

✓ Optimización completa: mejor resultado = 45.23
✓ Configuración óptima: {'temperature': 76.2, 'pressure': 2.9, 'pH': 7.1}
```

---

**FIN DEL DOCUMENTO**

Este plan cubre todas las mejoras identificadas más la dimensión semántica crítica. La implementación por fases permite validación incremental y adopción temprana de capacidades core.
