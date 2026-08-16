"""
semantic.builders - Construccion de prompts estructurados para LLMs.

Toma resultados semanticos y genera prompts consistentes y de alta densidad
informativa para apoyar decisiones de experimentacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from semantic.core import SemanticResult
from semantic.templates import (
    SECTION_TITLES,
    DECISION_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    format_bullets,
)


@dataclass
class PromptBuilder:
    """Builder para prompts semanticos de decision y analisis."""

    max_numerical_items: int = 8

    def build_decision_prompt(
        self,
        interpretation: str,
        reasoning: str,
        context: str = "",
        warnings: Optional[list[str]] = None,
        recommendations: Optional[list[str]] = None,
        numerical_summary: Optional[Dict[str, Any]] = None,
        confidence_level: str = "",
    ) -> str:
        """Construye prompt orientado a decision experimental."""
        return self._render_prompt(
            template=DECISION_PROMPT_TEMPLATE,
            header=SECTION_TITLES["decision"],
            interpretation=interpretation,
            reasoning=reasoning,
            context=context,
            warnings=warnings or [],
            recommendations=recommendations or [],
            numerical_summary=numerical_summary or {},
            confidence_level=confidence_level,
        )

    def build_analysis_prompt(
        self,
        interpretation: str,
        reasoning: str,
        context: str = "",
        warnings: Optional[list[str]] = None,
        recommendations: Optional[list[str]] = None,
        numerical_summary: Optional[Dict[str, Any]] = None,
        confidence_level: str = "",
    ) -> str:
        """Construye prompt orientado a analisis tecnico de resultados."""
        return self._render_prompt(
            template=ANALYSIS_PROMPT_TEMPLATE,
            header=SECTION_TITLES["analysis"],
            interpretation=interpretation,
            reasoning=reasoning,
            context=context,
            warnings=warnings or [],
            recommendations=recommendations or [],
            numerical_summary=numerical_summary or {},
            confidence_level=confidence_level,
        )

    def build_from_semantic_result(
        self,
        semantic_result: SemanticResult,
        mode: str = "analysis",
    ) -> str:
        """
        Construye prompt directamente desde SemanticResult.

        mode:
            - "analysis": enfocado en explicacion tecnica
            - "decision": enfocado en accion siguiente
        """
        numerical_summary = self._summarize_numerical(semantic_result.numerical)

        if mode == "decision":
            return self.build_decision_prompt(
                interpretation=semantic_result.interpretation,
                reasoning=semantic_result.reasoning,
                context=semantic_result.context,
                warnings=semantic_result.warnings,
                recommendations=semantic_result.recommendations,
                numerical_summary=numerical_summary,
                confidence_level=semantic_result.confidence_level,
            )

        return self.build_analysis_prompt(
            interpretation=semantic_result.interpretation,
            reasoning=semantic_result.reasoning,
            context=semantic_result.context,
            warnings=semantic_result.warnings,
            recommendations=semantic_result.recommendations,
            numerical_summary=numerical_summary,
            confidence_level=semantic_result.confidence_level,
        )

    def _render_prompt(
        self,
        template: str,
        header: str,
        interpretation: str,
        reasoning: str,
        context: str,
        warnings: list[str],
        recommendations: list[str],
        numerical_summary: Dict[str, Any],
        confidence_level: str,
    ) -> str:
        interpretation_section = (
            f"{SECTION_TITLES['interpretation']}:\n{interpretation}"
        )
        reasoning_section = f"{SECTION_TITLES['reasoning']}:\n{reasoning}"

        context_block = ""
        if context:
            context_block = f"\n\n{SECTION_TITLES['context']}:\n{context}"

        numerical_block = ""
        if numerical_summary:
            numerical_rows = [f"- {k}: {v}" for k, v in numerical_summary.items()]
            numerical_block = (
                f"\n\n{SECTION_TITLES['numerical']}:\n"
                + "\n".join(numerical_rows)
            )

        warnings_block = ""
        if warnings:
            warnings_block = (
                f"\n\n{SECTION_TITLES['warnings']}:\n"
                + format_bullets(warnings)
            )

        recommendations_block = ""
        if recommendations:
            recommendations_block = (
                f"\n\n{SECTION_TITLES['recommendations']}:\n"
                + format_bullets(recommendations)
            )

        confidence_block = ""
        if confidence_level:
            confidence_block = (
                f"\n\n{SECTION_TITLES['confidence']}:\n{confidence_level}"
            )

        return template.format(
            header=header,
            interpretation_section=interpretation_section,
            reasoning_section=reasoning_section,
            context_block=context_block,
            numerical_block=numerical_block,
            warnings_block=warnings_block,
            recommendations_block=recommendations_block,
            confidence_block=confidence_block,
        )

    def _summarize_numerical(self, numerical: Any) -> Dict[str, Any]:
        """Resume estructuras numericas complejas en un dict compacto."""
        if isinstance(numerical, dict):
            items = list(numerical.items())[: self.max_numerical_items]
            return {k: self._safe_render(v) for k, v in items}

        if hasattr(numerical, "to_dict"):
            try:
                as_dict = numerical.to_dict()
                if isinstance(as_dict, dict):
                    items = list(as_dict.items())[: self.max_numerical_items]
                    return {k: self._safe_render(v) for k, v in items}
            except Exception:
                return {"type": type(numerical).__name__}

        return {"value": self._safe_render(numerical)}

    def _safe_render(self, value: Any) -> Any:
        """Renderiza valores complejos de forma estable para texto."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value

        if isinstance(value, list):
            return [self._safe_render(v) for v in value[:5]]

        if isinstance(value, dict):
            reduced = list(value.items())[:5]
            return {k: self._safe_render(v) for k, v in reduced}

        if hasattr(value, "shape"):
            return f"<{type(value).__name__} shape={getattr(value, 'shape', None)}>"

        return str(value)
