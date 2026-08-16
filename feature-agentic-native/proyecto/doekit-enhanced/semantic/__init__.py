"""
doekit_enhanced.semantic - Interpretación semántica de resultados de DoE

Provee capa semántica para resultados numéricos de doekit, permitiendo
que agentes autónomos y humanos razonen efectivamente sobre experimentos.

Módulos:
    core: Estructuras base (SemanticResult, SemanticInterpreter)
    interpreters: Interpretadores especializados por tipo de resultado
    builders: Construcción de prompts estructurados
    templates: Templates reutilizables

Example:
    >>> import doekit as ed
    >>> from doekit_enhanced.semantic import interpret_result
    >>>
    >>> # Obtener recomendación con doekit
    >>> rec = ed.recommend_design(goal="optimization", factors=3, budget=20)
    >>>
    >>> # Agregar interpretación semántica
    >>> semantic = interpret_result(rec)
    >>>
    >>> # Usar para razonamiento
    >>> print(semantic.interpretation)
    >>> print(semantic.prompt_injection)  # Listo para LLM

Theory:
    Basado en principios de:
    - Explainable AI (Miller, 2019)
    - Dual-process theory (Kahneman, 2011)
    - Conversational maxims (Grice, 1975)
    - Distributed cognition (Hutchins, 1995)

    Ver docs/theory.md para fundamentación completa.

Author: doekit-enhanced contributors
License: MIT
"""

__version__ = "0.1.0"

# Core exports
from .core import (
    SemanticResult,
    SemanticInterpreter,
    SemanticRegistry,
    register_interpreter,
    interpret_result,
    get_registry,
    validate_semantic_result,
    with_semantics
)

from .builders import PromptBuilder

# Interpretadores específicos
from .interpreters import (
    RecommendationInterpreter,
    EvaluationInterpreter,
    FitInterpreter,
    ProposalInterpreter,
    ComparisonInterpreter
)

# Los interpretadores se auto-registran al importarse
# (ver registro automático al final de interpreters.py)

__all__ = [
    # Core
    "SemanticResult",
    "SemanticInterpreter",
    "SemanticRegistry",
    "register_interpreter",
    "interpret_result",
    "get_registry",
    "validate_semantic_result",
    "with_semantics",

    # Builders
    "PromptBuilder",

    # Interpretadores específicos
    "RecommendationInterpreter",
    "EvaluationInterpreter",
    "FitInterpreter",
    "ProposalInterpreter",
    "ComparisonInterpreter",

    # Version
    "__version__"
]
