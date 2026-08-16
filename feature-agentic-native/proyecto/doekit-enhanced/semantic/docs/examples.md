# Ejemplos de Uso del Módulo Semantic

## 1. Uso Básico: Crear SemanticResult Manualmente

```python
from doekit_enhanced.semantic import SemanticResult

# Resultado numérico simple
numerical_data = {
    "D_efficiency": 46.1,
    "G_efficiency": 79.7,
    "A_efficiency": 26.8,
    "n_runs": 13
}

# Crear representación semántica
result = SemanticResult(
    numerical=numerical_data,
    interpretation="Diseño aceptable dado presupuesto limitado de 20 corridas",
    reasoning=(
        "D-efficiency de 46.1% permite estimar efectos principales con precisión razonable. "
        "G-efficiency de 79.7% indica buena capacidad de predicción en la región experimental. "
        "Aunque no es óptimo, balancea costo y calidad apropiadamente."
    ),
    context="Diseño de 13 corridas para modelo cuadrático (10 parámetros). DOF=3.",
    warnings=[
        "D-efficiency relativamente baja - varianzas de coeficientes serán mayores que diseño óptimo",
        "DOF limitado (3) - modelo saturado, poco poder para detectar lack of fit"
    ],
    recommendations=[
        "Proceder con diseño actual si presupuesto no puede aumentarse",
        "Considerar diseño D-optimal (16-18 corridas) si presupuesto flexible",
        "Monitorear R² después de primera wave - si bajo (<0.7), agregar corridas"
    ],
    confidence_level="Moderada - basada en trade-off entre costo (13 runs) y precisión (46% D-eff)",
    metadata={"function": "evaluate", "design_method": "D-optimal"}
)

# Usar resultado
print(result.interpretation)
# > "Diseño aceptable dado presupuesto limitado de 20 corridas"

print(result.context_addition)
# > Texto estructurado completo listo para LLM

# Serializar
import json
print(json.dumps(result.to_dict(), indent=2))
```

## 2. Crear Interpretador Personalizado

```python
from doekit_enhanced.semantic import SemanticInterpreter, SemanticResult
from typing import Any, Dict, Optional, List

class MyCustomInterpreter(SemanticInterpreter):
    """Interpretador para resultados personalizados"""

    def validate_input(self, result: Any) -> bool:
        """Validar que es del tipo esperado"""
        return isinstance(result, dict) and "metric_value" in result

    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        """Generar interpretación semántica"""

        if not self.validate_input(numerical_result):
            raise ValueError("Resultado no válido para este interpretador")

        # Extraer datos
        metric = numerical_result["metric_value"]
        threshold = context.get("threshold", 0.5) if context else 0.5

        # Construir componentes semánticos
        interpretation = self._build_interpretation(metric, threshold)
        reasoning = self._build_reasoning(metric, threshold)
        ctx = self._build_context(numerical_result, context)
        warnings = self._extract_warnings(metric, threshold)
        recs = self._extract_recommendations(metric, threshold)
        confidence = self._assess_confidence(metric)

        return SemanticResult(
            numerical=numerical_result,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recs,
            confidence_level=confidence,
            metadata={"interpreter": "MyCustomInterpreter"}
        )

    def _build_interpretation(self, metric: float, threshold: float) -> str:
        if metric >= threshold:
            return f"Métrica ({metric:.2f}) supera umbral requerido ({threshold:.2f})"
        else:
            return f"Métrica ({metric:.2f}) bajo umbral requerido ({threshold:.2f})"

    def _build_reasoning(self, metric: float, threshold: float) -> str:
        gap = metric - threshold
        gap_pct = (gap / threshold * 100) if threshold != 0 else 0

        if gap >= 0:
            return (
                f"El valor observado ({metric:.2f}) excede el umbral ({threshold:.2f}) "
                f"por {gap:.2f} ({gap_pct:+.1f}%). Esto indica performance satisfactoria."
            )
        else:
            return (
                f"El valor observado ({metric:.2f}) está {abs(gap):.2f} por debajo del "
                f"umbral ({threshold:.2f}), representando un déficit de {abs(gap_pct):.1f}%."
            )

    def _build_context(self, result: dict, context: Optional[dict]) -> str:
        return f"Evaluación de métrica con threshold={context.get('threshold', 0.5) if context else 0.5}"

    def _extract_warnings(self, metric: float, threshold: float) -> List[str]:
        warnings = []
        if metric < threshold:
            warnings.append("Performance bajo umbral - acción correctiva requerida")
        if abs(metric - threshold) < 0.1:
            warnings.append("Métrica cercana a umbral - monitorear de cerca")
        return warnings

    def _extract_recommendations(self, metric: float, threshold: float) -> List[str]:
        if metric >= threshold:
            return ["Mantener condiciones actuales", "Monitorear para detectar degradación"]
        else:
            return [
                "Investigar causas de bajo performance",
                "Considerar ajustes a parámetros operacionales"
            ]

    def _assess_confidence(self, metric: float) -> str:
        # Placeholder - en caso real consideraría incertidumbre
        return "Alta - basada en medición directa"

# Uso
interpreter = MyCustomInterpreter()
result_data = {"metric_value": 0.75, "metadata": {}}
semantic = interpreter.interpret(result_data, context={"threshold": 0.6})

print(semantic.interpretation)
# > "Métrica (0.75) supera umbral requerido (0.60)"
```

## 3. Usar Registro Global

```python
from doekit_enhanced.semantic import (
    register_interpreter,
    interpret_result,
    SemanticInterpreter,
    SemanticResult
)

# Definir interpretador
class CustomResultInterpreter(SemanticInterpreter):
    def validate_input(self, result):
        return hasattr(result, 'value')

    def interpret(self, numerical_result, context=None):
        return SemanticResult(
            numerical=numerical_result,
            interpretation=f"Valor: {numerical_result.value}",
            reasoning="Resultado directo de medición",
            context="Experimento de prueba",
            warnings=[],
            recommendations=[]
        )

# Registrar
register_interpreter("CustomResult", CustomResultInterpreter)

# Usar automáticamente
class CustomResult:
    def __init__(self, value):
        self.value = value

result = CustomResult(42)
semantic = interpret_result(result)  # Auto-detecta interpretador

print(semantic.interpretation)
# > "Valor: 42"
```

## 4. Validación de Calidad

```python
from doekit_enhanced.semantic import SemanticResult, validate_semantic_result

# Resultado bien formado
good_result = SemanticResult(
    numerical={"value": 1.0},
    interpretation="Interpretación concisa y clara",
    reasoning="Razonamiento detallado pero no excesivo. Explica el por qué sin saturar.",
    context="Contexto necesario",
    warnings=["Advertencia específica con suficiente detalle"],
    recommendations=["Recomendación accionable con pasos claros"]
)

assert validate_semantic_result(good_result) == True

# Resultado mal formado (interpretation muy larga)
bad_result = SemanticResult(
    numerical={"value": 1.0},
    interpretation="X" * 400,  # > 300 caracteres
    reasoning="OK",
    context="OK"
)

assert validate_semantic_result(bad_result) == False
```

## 5. Integración con doekit (Patrón Adapter)

```python
import doekit as ed
from doekit_enhanced.semantic import SemanticInterpreter, SemanticResult

# Este es un ejemplo de cómo se integraría con doekit real
# (requiere que los interpretadores específicos estén implementados)

class RecommendationInterpreter(SemanticInterpreter):
    """Interpreta resultados de recommend_design()"""

    def validate_input(self, result) -> bool:
        # Verificar que es objeto Recommendation de doekit
        return hasattr(result, 'method') and hasattr(result, 'design')

    def interpret(self, numerical_result, context=None) -> SemanticResult:
        rec = numerical_result
        
        # Extraer información
        method = rec.method
        n_runs = rec.design.n_runs
        table = rec.table
        
        # Construir interpretación
        interpretation = f"Se recomienda {method} con {n_runs} corridas experimentales"
        
        # Razonamiento basado en tabla de alternativas
        reasoning = self._analyze_alternatives(table, method)
        
        # Contexto del escenario
        ctx = f"Objetivo: {rec.scenario.get('goal')}. Factores: {rec.scenario.get('n_factors')}."
        
        # Warnings de los caveats
        warnings = rec.caveats[:3]  # Primeros 3 caveats
        
        # Recomendaciones
        recs = [
            f"Ejecutar {n_runs} experimentos usando método {method}",
            "Después de primera wave, usar propose_next_runs() para refinamiento"
        ]
        
        # Construir prompt
        prompt = self._build_prompt(method, n_runs, reasoning, ctx, warnings, recs)
        
        return SemanticResult(
            numerical=rec,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recs,
            confidence_level=self._assess_confidence(table),
            context_addition=prompt,
            metadata={"function": "recommend_design", "method": method}
        )
    
    def _analyze_alternatives(self, table, winner_method):
        # Comparar ganador con alternativas
        winner = table[table['method'] == winner_method].iloc[0]
        
        comparisons = []
        for _, row in table.iterrows():
            if row['method'] == winner_method:
                continue
            if row['runs'] > winner['runs']:
                comparisons.append(
                    f"{row['method']} requiere {row['runs'] - winner['runs']} "
                    f"corridas más pero solo mejora eficiencia marginalmente"
                )
        
        if comparisons:
            return f"{winner_method} seleccionado porque " + "; ".join(comparisons[:2])
        else:
            return f"{winner_method} es la mejor opción disponible"
    
    def _assess_confidence(self, table):
        # Si ganador es claro, alta confianza
        if len(table) > 1:
            winner = table.iloc[0]
            second = table.iloc[1]
            if winner.get('D_eff', 0) - second.get('D_eff', 0) > 10:
                return "Alta - clara ventaja sobre alternativas"
        return "Moderada"
    
    def _build_prompt(self, method, n_runs, reasoning, ctx, warnings, recs):
        return f"""
RECOMENDACIÓN DE DISEÑO EXPERIMENTAL:

Diseño recomendado: {method}
Corridas experimentales: {n_runs}

RAZONAMIENTO:
{reasoning}

CONTEXTO:
{ctx}

ADVERTENCIAS:
{chr(10).join('⚠ ' + w for w in warnings)}

RECOMENDACIONES:
{chr(10).join('→ ' + r for r in recs)}
"""

# Uso
# rec = ed.recommend_design(goal="optimization", factors=3, budget=20)
# interpreter = RecommendationInterpreter()
# semantic = interpreter.interpret(rec)
# print(semantic.context_addition)
```

## 6. Serialización y Persistencia

```python
from doekit_enhanced.semantic import SemanticResult
import json

# Crear resultado
result = SemanticResult(
    numerical={"metric": 0.85},
    interpretation="Resultado excelente",
    reasoning="Supera expectativas",
    context="Experimento 123"
)

# Serializar a JSON
json_str = result.to_json()
print(json_str)

# Guardar a archivo
with open("semantic_result.json", "w") as f:
    f.write(json_str)

# Cargar desde archivo
with open("semantic_result.json", "r") as f:
    data = json.load(f)

# Reconstruir objeto
loaded = SemanticResult.from_dict(data)
print(loaded.interpretation)
# > "Resultado excelente"
```

## 7. Uso con LLMs (Patrón de Inyección)

```python
from doekit_enhanced.semantic import SemanticResult

# Obtener resultado semántico (de cualquier fuente)
semantic_result = SemanticResult(
    numerical={"decision": "continue", "score": 0.72},
    interpretation="Recomendado continuar experimentación",
    reasoning="Score de 0.72 supera umbral de 0.50 con margen suficiente",
    context="Wave 2 de 5, presupuesto 45% consumido",
    warnings=["Riesgo moderado basado en DOF limitado"],
    recommendations=["Ejecutar 4 experimentos propuestos", "Re-evaluar después de wave"]
)

# Construir prompt para LLM
agent_prompt = f"""
Tarea: Decidir si continuar experimentación.

{semantic_result.context_addition}

Basándote en la información anterior, ¿cuál es tu decisión y por qué?
Responde en formato: DECISIÓN: [continuar/detener] | JUSTIFICACIÓN: [tu razonamiento]
"""

# Enviar a LLM
# response = llm.complete(agent_prompt)
# decision = parse_decision(response)
```

## 8. Testing de Interpretadores

```python
import pytest
from doekit_enhanced.semantic import SemanticInterpreter, SemanticResult

class TestCustomInterpreter:
    """Tests para interpretador personalizado"""
    
    def test_valid_input(self):
        interpreter = MyCustomInterpreter()
        result = {"metric_value": 0.75}
        assert interpreter.validate_input(result) == True
    
    def test_invalid_input(self):
        interpreter = MyCustomInterpreter()
        result = {"wrong_key": 0.75}
        assert interpreter.validate_input(result) == False
    
    def test_interpretation_above_threshold(self):
        interpreter = MyCustomInterpreter()
        result = {"metric_value": 0.75}
        semantic = interpreter.interpret(result, context={"threshold": 0.6})
        
        assert "supera umbral" in semantic.interpretation.lower()
        assert semantic.confidence_level != ""
    
    def test_interpretation_below_threshold(self):
        interpreter = MyCustomInterpreter()
        result = {"metric_value": 0.45}
        semantic = interpreter.interpret(result, context={"threshold": 0.6})
        
        assert "bajo umbral" in semantic.interpretation.lower()
        assert len(semantic.warnings) > 0
    
    def test_semantic_result_structure(self):
        interpreter = MyCustomInterpreter()
        result = {"metric_value": 0.75}
        semantic = interpreter.interpret(result)
        
        # Validar estructura
        assert isinstance(semantic, SemanticResult)
        assert semantic.interpretation != ""
        assert semantic.reasoning != ""
        assert isinstance(semantic.warnings, list)
        assert isinstance(semantic.recommendations, list)
        assert semantic.context_addition != ""
    
    def test_serialization(self):
        interpreter = MyCustomInterpreter()
        result = {"metric_value": 0.75}
        semantic = interpreter.interpret(result)
        
        # Serializar y deserializar
        data = semantic.to_dict()
        reconstructed = SemanticResult.from_dict(data)
        
        assert reconstructed.interpretation == semantic.interpretation
        assert reconstructed.reasoning == semantic.reasoning
```

---

## Patrones de Uso Recomendados

### Patrón 1: Wrapper Transparente
Agregar semántica sin modificar código existente.

```python
result = existing_function()  # doekit original
semantic = interpret_result(result)  # agregar semántica
use_semantic(semantic)  # usar versión enriquecida
```

### Patrón 2: Opt-in
Parámetro para activar semántica.

```python
def enhanced_function(..., include_semantics=False):
    result = compute_numerical()
    if include_semantics:
        return interpret_result(result)
    return result
```

### Patrón 3: Siempre Semántico
Retornar siempre SemanticResult, acceder a numérico via `.numerical`.

```python
def new_function(...) -> SemanticResult:
    numerical = compute()
    return interpret_result(numerical)

# Uso
result = new_function()
numbers = result.numerical  # Si se necesita
prompt = result.context_addition  # Para LLM
```

---

Siguiente: Implementación de interpretadores específicos en `interpreters.py`
