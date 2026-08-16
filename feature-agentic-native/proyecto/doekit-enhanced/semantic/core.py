"""
semantic.core - Estructuras base para representación semántica de resultados

Este módulo provee las abstracciones fundamentales para agregar interpretación
semántica a resultados numéricos de doekit.

Teoría:
    Basado en principios de Explainable AI y cognición distribuida. Ver docs/theory.md

Author: doekit-enhanced contributors
License: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime
import json
from functools import wraps


@dataclass
class SemanticResult:
    """
    Wrapper que agrega interpretación semántica a resultados numéricos.

    Provee representación dual:
    - Numérica: datos cuantitativos originales
    - Semántica: interpretación en lenguaje natural
    - Híbrida: prompt estructurado para LLMs

    Attributes:
        numerical: Resultado numérico original (sin modificar)
        interpretation: Resumen de qué significa (1-2 oraciones)
        reasoning: Justificación técnica del resultado (2-4 oraciones)
        context: Información de fondo necesaria para entender
        warnings: Lista de advertencias o caveats importantes
        recommendations: Lista de acciones recomendadas
        confidence_level: Nivel de confianza cualitativo con explicación
        context_addition: Texto estructurado listo para inyectar en contexto LLM
        metadata: Información adicional (timestamp, función origen, etc.)

    Example:
        >>> from doekit_enhanced.semantic import SemanticResult
        >>> result = SemanticResult(
        ...     numerical={"D_efficiency": 46.1, "G_efficiency": 79.7},
        ...     interpretation="Diseño aceptable dado presupuesto limitado",
        ...     reasoning="D-efficiency de 46.1% permite estimar efectos principales...",
        ...     context="Diseño de 13 corridas para modelo cuadrático con 10 parámetros",
        ...     warnings=["Saturación cercana - considerar más corridas si posible"],
        ...     recommendations=["Proceder con diseño actual", "Re-evaluar después de primera wave"],
        ...     confidence_level="Moderada - basada en trade-off entre costo y precisión",
        ...     context_addition="...",
        ...     metadata={"function": "recommend_design", "timestamp": "2026-08-13T..."}
        ... )

    Theory:
        Diseño basado en:
        - Teoría de Grice (máximas conversacionales): cantidad, calidad, relevancia, manera
        - Dual-process theory (Kahneman): Sistema 1 (semántica) + Sistema 2 (numérica)
        - Explainable AI: contrastiva, selectiva, social
    """

    # NUMÉRICO
    numerical: Any

    # SEMÁNTICO
    interpretation: str
    reasoning: str
    context: str = ""  # Opcional con default vacío
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_level: str = ""

    # HÍBRIDO
    context_addition: str = ""

    # METADATOS
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validación y auto-completado"""
        # Agregar timestamp si no existe
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()

        # Validar campos requeridos
        if not self.interpretation:
            raise ValueError("interpretation es requerido")
        if not self.reasoning:
            raise ValueError("reasoning es requerido")

        # Auto-generar context_addition si está vacío
        if not self.context_addition:
            self.context_addition = self._auto_generate_prompt()

    def _auto_generate_prompt(self) -> str:
        """Genera prompt básico si no se provee uno personalizado"""
        sections = []

        # Interpretación principal
        sections.append(f"INTERPRETACIÓN:\n{self.interpretation}")

        # Razonamiento
        sections.append(f"\nRAZONAMIENTO:\n{self.reasoning}")

        # Contexto
        if self.context:
            sections.append(f"\nCONTEXTO:\n{self.context}")

        # Advertencias
        if self.warnings:
            warnings_text = "\n".join(f"⚠ {w}" for w in self.warnings)
            sections.append(f"\nADVERTENCIAS:\n{warnings_text}")

        # Recomendaciones
        if self.recommendations:
            recs_text = "\n".join(f"→ {r}" for r in self.recommendations)
            sections.append(f"\nRECOMENDACIONES:\n{recs_text}")

        # Confianza
        if self.confidence_level:
            sections.append(f"\nNIVEL DE CONFIANZA:\n{self.confidence_level}")

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización completa a diccionario.

        Returns:
            Diccionario con todas las representaciones
        """
        return {
            "numerical": self._serialize_numerical(),
            "semantic": {
                "interpretation": self.interpretation,
                "reasoning": self.reasoning,
                "context": self.context,
                "warnings": list(self.warnings),
                "recommendations": list(self.recommendations),
                "confidence_level": self.confidence_level
            },
            "context_addition": self.context_addition,
            "metadata": dict(self.metadata)
        }

    def _serialize_numerical(self) -> Any:
        """Serializa resultado numérico (maneja objetos doekit)"""
        if hasattr(self.numerical, 'to_dict'):
            return self.numerical.to_dict()
        elif isinstance(self.numerical, dict):
            return self.numerical
        elif isinstance(self.numerical, (int, float, str, bool, type(None))):
            return self.numerical
        else:
            # Fallback: convertir a string
            return str(self.numerical)

    def to_json(self, indent: int = 2) -> str:
        """Serialización a JSON"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticResult':
        """Reconstruye desde diccionario"""
        semantic = data.get("semantic", {})
        return cls(
            numerical=data.get("numerical"),
            interpretation=semantic.get("interpretation", ""),
            reasoning=semantic.get("reasoning", ""),
            context=semantic.get("context", ""),
            warnings=semantic.get("warnings", []),
            recommendations=semantic.get("recommendations", []),
            confidence_level=semantic.get("confidence_level", ""),
            context_addition=data.get("context_addition", ""),
            metadata=data.get("metadata", {})
        )

    def __str__(self) -> str:
        """Representación legible"""
        return self.context_addition

    def __repr__(self) -> str:
        """Representación para debugging"""
        return (
            f"SemanticResult(interpretation='{self.interpretation[:50]}...', "
            f"confidence='{self.confidence_level}')"
        )


class SemanticInterpreter(ABC):
    """
    Interfaz base para interpretadores semánticos.

    Los interpretadores convierten resultados numéricos específicos
    (Recommendation, DesignEvaluation, etc.) en SemanticResult.

    Cada tipo de resultado debe tener su interpretador especializado
    para generar semántica contextualmente apropiada.

    Theory:
        Patrón Strategy - diferentes estrategias de interpretación para
        diferentes tipos de resultados.
    """

    @abstractmethod
    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        """
        Interpreta resultado numérico y genera representación semántica.

        Args:
            numerical_result: Resultado numérico de doekit
            context: Contexto adicional (opcional)

        Returns:
            SemanticResult con interpretación completa

        Raises:
            ValueError: Si resultado no es del tipo esperado
        """
        pass

    @abstractmethod
    def validate_input(self, result: Any) -> bool:
        """
        Valida que el resultado es del tipo esperado por este interpretador.

        Args:
            result: Resultado a validar

        Returns:
            True si es válido, False si no
        """
        pass

    def _build_interpretation(self, **kwargs) -> str:
        """Helper para construir interpretación (override en subclases)"""
        return ""

    def _build_reasoning(self, **kwargs) -> str:
        """Helper para construir razonamiento (override en subclases)"""
        return ""

    def _build_context(self, **kwargs) -> str:
        """Helper para construir contexto (override en subclases)"""
        return ""

    def _extract_warnings(self, **kwargs) -> List[str]:
        """Helper para extraer advertencias (override en subclases)"""
        return []

    def _extract_recommendations(self, **kwargs) -> List[str]:
        """Helper para extraer recomendaciones (override en subclases)"""
        return []

    def _assess_confidence(self, **kwargs) -> str:
        """Helper para evaluar confianza (override en subclases)"""
        return "Moderada"


class SemanticRegistry:
    """
    Registro global de interpretadores por tipo de resultado.

    Permite auto-detección de interpretador apropiado basándose en
    tipo de resultado numérico.

    Example:
        >>> from doekit_enhanced.semantic import SemanticRegistry
        >>> registry = SemanticRegistry()
        >>> registry.register("Recommendation", RecommendationInterpreter)
        >>> interpreter = registry.get_interpreter(recommendation_obj)
        >>> semantic = interpreter.interpret(recommendation_obj)
    """

    def __init__(self):
        self._interpreters: Dict[str, type] = {}
        self._instances: Dict[str, SemanticInterpreter] = {}

    def register(
        self,
        result_type_name: str,
        interpreter_class: type
    ):
        """
        Registra interpretador para un tipo de resultado.

        Args:
            result_type_name: Nombre del tipo (e.g., "Recommendation")
            interpreter_class: Clase del interpretador
        """
        if not issubclass(interpreter_class, SemanticInterpreter):
            raise TypeError(
                f"{interpreter_class} debe heredar de SemanticInterpreter"
            )
        self._interpreters[result_type_name] = interpreter_class

    def get_interpreter(
        self,
        result: Any,
        create_if_not_exists: bool = True
    ) -> Optional[SemanticInterpreter]:
        """
        Obtiene interpretador apropiado para un resultado.

        Args:
            result: Resultado numérico
            create_if_not_exists: Si True, crea instancia si no existe

        Returns:
            Interpretador apropiado o None si no se encuentra
        """
        type_name = type(result).__name__

        if type_name not in self._interpreters:
            return None

        if type_name not in self._instances and create_if_not_exists:
            interpreter_class = self._interpreters[type_name]
            self._instances[type_name] = interpreter_class()

        return self._instances.get(type_name)

    def interpret(
        self,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[SemanticResult, Any]:
        """
        Interpreta resultado usando interpretador registrado.

        Si no hay interpretador registrado, retorna resultado original.

        Args:
            result: Resultado a interpretar
            context: Contexto opcional

        Returns:
            SemanticResult si hay interpretador, resultado original si no
        """
        interpreter = self.get_interpreter(result)

        if interpreter is None:
            return result  # Sin interpretador, retornar original

        return interpreter.interpret(result, context)


# Registro global (singleton)
_global_registry = SemanticRegistry()


def register_interpreter(result_type_name: str, interpreter_class: type):
    """
    Registra interpretador en el registro global.

    Args:
        result_type_name: Nombre del tipo de resultado
        interpreter_class: Clase del interpretador

    Example:
        >>> @register_interpreter("CustomResult", CustomInterpreter)
        >>> class CustomInterpreter(SemanticInterpreter):
        ...     def interpret(self, result, context=None):
        ...         return SemanticResult(...)
    """
    _global_registry.register(result_type_name, interpreter_class)


def interpret_result(
    result: Any,
    context: Optional[Dict[str, Any]] = None
) -> Union[SemanticResult, Any]:
    """
    Función convenience para interpretar cualquier resultado.

    Usa registro global para encontrar interpretador apropiado.

    Args:
        result: Resultado a interpretar
        context: Contexto opcional

    Returns:
        SemanticResult si hay interpretador, resultado original si no

    Example:
        >>> from doekit_enhanced.semantic import interpret_result
        >>> rec = ed.recommend_design(goal="optimization", factors=3)
        >>> semantic = interpret_result(rec)
        >>> print(semantic.context_addition)
    """
    return _global_registry.interpret(result, context)


def get_registry() -> SemanticRegistry:
    """Obtiene registro global de interpretadores"""
    return _global_registry


def with_semantics(
    interpreter_class: Optional[type] = None,
    context_builder: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    on_error: str = "return_original"
) -> Callable:
    """
    Decorador opt-in para agregar capa semántica sin modificar APIs existentes.

    El decorador consume `include_semantics` desde kwargs para no romper
    funciones originales que no conocen ese parámetro.

    Args:
        interpreter_class: Interpretador específico opcional. Si es None,
            usa auto-detección vía registro global.
        context_builder: Callable opcional para construir contexto semántico
            a partir de args/kwargs y resultado.
        on_error: Política ante error de interpretación.
            - "return_original": retorna resultado numérico original
            - "raise": propaga la excepción
    """
    if on_error not in {"return_original", "raise"}:
        raise ValueError("on_error debe ser 'return_original' o 'raise'")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            include_semantics = bool(kwargs.pop("include_semantics", False))

            result = func(*args, **kwargs)
            if not include_semantics:
                return result

            try:
                semantic_context: Optional[Dict[str, Any]] = None
                if context_builder is not None:
                    semantic_context = context_builder(*args, **kwargs, result=result)

                if interpreter_class is not None:
                    interpreter = interpreter_class()
                    return interpreter.interpret(result, context=semantic_context)

                return interpret_result(result, context=semantic_context)
            except Exception:
                if on_error == "raise":
                    raise
                return result

        return wrapper

    return decorator


# Funciones helper para validación

def validate_semantic_result(result: SemanticResult) -> bool:
    """
    Valida que un SemanticResult cumple con criterios de calidad.

    Criterios:
    - Interpretation no vacío, < 300 caracteres
    - Reasoning no vacío, < 1000 caracteres
    - Context < 500 caracteres
    - Warnings son específicas (> 10 caracteres cada una)
    - Recommendations son accionables (> 15 caracteres cada una)

    Args:
        result: SemanticResult a validar

    Returns:
        True si cumple criterios, False si no
    """
    if not result.interpretation or len(result.interpretation) > 300:
        return False

    if not result.reasoning or len(result.reasoning) > 1000:
        return False

    if len(result.context) > 500:
        return False

    for warning in result.warnings:
        if len(warning) < 10:
            return False

    for rec in result.recommendations:
        if len(rec) < 15:
            return False

    return True


__all__ = [
    "SemanticResult",
    "SemanticInterpreter",
    "SemanticRegistry",
    "register_interpreter",
    "interpret_result",
    "get_registry",
    "validate_semantic_result",
    "with_semantics"
]
