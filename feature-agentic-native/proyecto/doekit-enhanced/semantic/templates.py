"""
semantic.templates - Templates reutilizables para prompts semánticos.

Define estructuras de texto consistentes para decisiones y análisis de resultados.
"""

from typing import Dict


SECTION_TITLES: Dict[str, str] = {
    "decision": "DECISION EXPERIMENTAL",
    "analysis": "ANALISIS DE RESULTADOS",
    "interpretation": "INTERPRETACION",
    "reasoning": "RAZONAMIENTO",
    "context": "CONTEXTO",
    "warnings": "ADVERTENCIAS",
    "recommendations": "RECOMENDACIONES",
    "numerical": "RESUMEN NUMERICO",
    "confidence": "NIVEL DE CONFIANZA",
}


DECISION_PROMPT_TEMPLATE = """{header}\n\n{interpretation_section}\n\n{reasoning_section}{context_block}{numerical_block}{warnings_block}{recommendations_block}{confidence_block}"""


ANALYSIS_PROMPT_TEMPLATE = """{header}\n\n{interpretation_section}\n\n{reasoning_section}{context_block}{numerical_block}{warnings_block}{recommendations_block}{confidence_block}"""


def format_bullets(items: list[str], prefix: str = "- ") -> str:
    """Convierte una lista de textos en bloque con viñetas."""
    return "\n".join(f"{prefix}{item}" for item in items)
