"""
Tests reales para builders y decorador with_semantics.
"""

import sys
import os

# Agregar paths para importar (mismo patrón que tests existentes)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import numpy as np
import doekit as ed

from semantic.core import SemanticResult, with_semantics
from semantic.builders import PromptBuilder


class TestPromptBuilder:
    def test_build_analysis_prompt_from_semantic_result(self):
        semantic = SemanticResult(
            numerical={"D_efficiency": 46.1, "G_efficiency": 79.7},
            interpretation="Diseno aceptable para etapa exploratoria",
            reasoning="Balancea costo y precision en etapa temprana",
            context="Budget limitado con 3 factores",
            warnings=["La eficiencia D puede mejorar con mas corridas"],
            recommendations=["Proceder y re-evaluar tras primera wave"],
            confidence_level="Moderada"
        )

        builder = PromptBuilder()
        prompt = builder.build_from_semantic_result(semantic, mode="analysis")

        assert "ANALISIS DE RESULTADOS" in prompt
        assert "INTERPRETACION" in prompt
        assert "RAZONAMIENTO" in prompt
        assert "RESUMEN NUMERICO" in prompt
        assert "D_efficiency" in prompt

    def test_build_decision_prompt(self):
        builder = PromptBuilder(max_numerical_items=3)
        prompt = builder.build_decision_prompt(
            interpretation="Se recomienda continuar",
            reasoning="La mejora supera el umbral esperado",
            context="Wave 2 de 5",
            warnings=["Revisar varianza por posible heterogeneidad"],
            recommendations=["Ejecutar 4 corridas adicionales"],
            numerical_summary={"delta_D_eff": 12.3, "score": 2.1, "threshold": 1.6},
            confidence_level="Alta"
        )

        assert "DECISION EXPERIMENTAL" in prompt
        assert "ADVERTENCIAS" in prompt
        assert "RECOMENDACIONES" in prompt
        assert "delta_D_eff" in prompt
        assert "NIVEL DE CONFIANZA" in prompt


class TestWithSemanticsDecorator:
    def test_with_semantics_returns_original_when_opt_out(self):
        @with_semantics()
        def build_payload(value):
            return {"value": value, "status": "ok"}

        result = build_payload(10)
        assert isinstance(result, dict)
        assert result["value"] == 10

    def test_with_semantics_uses_registered_interpreter(self):
        @with_semantics()
        def get_recommendation():
            return ed.recommend_design(goal="screening", factors=4, budget=12)

        semantic = get_recommendation(include_semantics=True)

        assert isinstance(semantic, SemanticResult)
        assert "RecommendationInterpreter" == semantic.metadata.get("interpreter")
        assert "corridas" in semantic.interpretation.lower()

    def test_with_semantics_with_context_builder(self):
        def build_context(goal, result=None):
            return {
                "goal": goal,
                "method": getattr(result, "method", "unknown")
            }

        @with_semantics(context_builder=build_context)
        def run_recommendation(goal):
            return ed.recommend_design(goal=goal, factors=3, budget=15)

        semantic = run_recommendation("optimization", include_semantics=True)
        assert isinstance(semantic, SemanticResult)
        assert "goal: optimization" in semantic.context.lower()

    def test_with_semantics_error_policy_return_original(self):
        class BrokenInterpreter:
            def interpret(self, result, context=None):
                raise RuntimeError("boom")

        @with_semantics(interpreter_class=BrokenInterpreter, on_error="return_original")
        def build_array():
            np.random.seed(42)
            return np.random.randn(3)

        result = build_array(include_semantics=True)
        assert hasattr(result, "shape")
        assert result.shape[0] == 3
