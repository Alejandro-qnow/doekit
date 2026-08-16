"""
Tests REALES del módulo semantic.core sin mockear datos.

Principio: Test-Driven Validation
- Usar doekit real
- Datos reales de experimentos
- Sin mocks
- Validar contrato completo
"""

import sys
import os

# Agregar paths para importar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../venv/Lib/site-packages'))

import pytest
import numpy as np
import doekit as ed
from semantic.core import (
    SemanticResult,
    SemanticInterpreter,
    SemanticRegistry,
    interpret_result,
    validate_semantic_result
)


class TestSemanticResultBasic:
    """Tests básicos de SemanticResult sin dependencias externas"""

    def test_semantic_result_creation(self):
        """Test: Crear SemanticResult básico funciona"""
        result = SemanticResult(
            numerical={"value": 42},
            interpretation="Test interpretation",
            reasoning="Test reasoning",
            context="Test context"
        )

        assert result.numerical["value"] == 42
        assert result.interpretation == "Test interpretation"
        assert result.reasoning == "Test reasoning"
        assert result.context == "Test context"
        assert isinstance(result.warnings, list)
        assert isinstance(result.recommendations, list)
        assert result.context_addition != ""  # Auto-generado
        assert "timestamp" in result.metadata  # Auto-agregado

    def test_semantic_result_validation_fails_without_required(self):
        """Test: SemanticResult requiere campos obligatorios"""
        with pytest.raises(ValueError, match="interpretation"):
            SemanticResult(
                numerical={"value": 42},
                interpretation="",  # Vacío - debe fallar
                reasoning="Test"
            )

        with pytest.raises(ValueError, match="reasoning"):
            SemanticResult(
                numerical={"value": 42},
                interpretation="Test",
                reasoning=""  # Vacío - debe fallar
            )

    def test_semantic_result_auto_prompt_generation(self):
        """Test: Prompt se auto-genera con estructura correcta"""
        result = SemanticResult(
            numerical={"value": 42},
            interpretation="Test interpretation",
            reasoning="Test reasoning",
            context="Test context",
            warnings=["Warning 1", "Warning 2"],
            recommendations=["Rec 1", "Rec 2"],
            confidence_level="High"
        )

        prompt = result.context_addition

        # Validar estructura
        assert "INTERPRETACIÓN:" in prompt
        assert "Test interpretation" in prompt
        assert "RAZONAMIENTO:" in prompt
        assert "Test reasoning" in prompt
        assert "CONTEXTO:" in prompt
        assert "Test context" in prompt
        assert "ADVERTENCIAS:" in prompt
        assert "Warning 1" in prompt
        assert "RECOMENDACIONES:" in prompt
        assert "Rec 1" in prompt
        assert "NIVEL DE CONFIANZA:" in prompt
        assert "High" in prompt

    def test_semantic_result_serialization(self):
        """Test: Serialización y deserialización funcionan"""
        original = SemanticResult(
            numerical={"value": 42, "nested": {"key": "value"}},
            interpretation="Test interpretation",
            reasoning="Test reasoning",
            context="Test context",
            warnings=["W1"],
            recommendations=["R1"],
            confidence_level="High"
        )

        # Serializar
        data = original.to_dict()

        # Validar estructura
        assert "numerical" in data
        assert "semantic" in data
        assert "context_addition" in data
        assert "metadata" in data

        # Deserializar
        reconstructed = SemanticResult.from_dict(data)

        # Validar igualdad
        assert reconstructed.interpretation == original.interpretation
        assert reconstructed.reasoning == original.reasoning
        assert reconstructed.context == original.context
        assert reconstructed.warnings == original.warnings
        assert reconstructed.recommendations == original.recommendations

    def test_semantic_result_to_json(self):
        """Test: Conversión a JSON funciona"""
        result = SemanticResult(
            numerical={"value": 42},
            interpretation="Test",
            reasoning="Test reasoning",
            context="Test context"
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "Test" in json_str
        assert "value" in json_str

        # Validar que es JSON válido
        import json
        parsed = json.loads(json_str)
        assert parsed["semantic"]["interpretation"] == "Test"


class TestSemanticResultWithDoekitReal:
    """Tests usando datos REALES de doekit (sin mocks)"""

    def test_semantic_result_with_doekit_dict(self):
        """Test: SemanticResult puede envolver dict de doekit"""
        # Usar efficiency metrics reales de doekit
        design = ed.box_behnken({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)})
        model = ed.Model.full_quadratic(["X1", "X2", "X3"])

        # Obtener métricas reales
        effs = ed.efficiencies(design, model=model)

        # Envolver en SemanticResult
        result = SemanticResult(
            numerical=effs,
            interpretation=f"D-efficiency: {effs['D_efficiency']:.1f}%",
            reasoning="Calculated using doekit efficiencies function",
            context=f"Box-Behnken design with {design.n_runs} runs"
        )

        # Validar que datos numéricos se preservan
        assert result.numerical["D_efficiency"] == effs["D_efficiency"]
        assert result.numerical["G_efficiency"] == effs["G_efficiency"]
        assert result.numerical["A_efficiency"] == effs["A_efficiency"]

        # Validar que semántica se genera
        assert "D-efficiency" in result.interpretation
        assert str(design.n_runs) in result.context

    def test_semantic_result_with_doekit_recommendation(self):
        """Test: SemanticResult puede envolver Recommendation de doekit"""
        # Obtener recomendación real
        rec = ed.recommend_design(
            goal="optimization",
            factors={"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)},
            budget=20,
            model_order="quadratic"
        )

        # Envolver en SemanticResult
        result = SemanticResult(
            numerical=rec,
            interpretation=f"Recommended: {rec.method} with {rec.design.n_runs} runs",
            reasoning=rec.rationale,
            context=f"Goal: {rec.scenario['goal']}, Budget: {rec.scenario['budget']}",
            warnings=rec.caveats[:2],
            recommendations=[f"Execute {rec.design.n_runs} experiments"]
        )

        # Validar preservación de objeto doekit
        assert result.numerical.method == rec.method
        assert result.numerical.design.n_runs == rec.design.n_runs

        # Validar que tiene serialización (to_dict existe)
        data = result.to_dict()
        assert "method" in data["numerical"]
        assert "design" in data["numerical"]

    def test_semantic_result_with_doekit_fit(self):
        """Test: SemanticResult puede envolver FitResult de doekit"""
        # Crear diseño y datos reales
        design = ed.plackett_burman(5)
        y = np.random.randn(design.n_runs) * 2 + 10  # Datos simulados pero realistas

        # Ajustar modelo real
        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        # Envolver en SemanticResult
        result = SemanticResult(
            numerical=fit,
            interpretation=f"R² = {fit.r_squared:.3f}",
            reasoning=f"Model fitted with {fit.dof} degrees of freedom",
            context=f"Plackett-Burman design with {design.n_runs} runs",
            warnings=["Check residuals for normality"] if fit.r_squared < 0.7 else []
        )

        # Validar preservación
        assert result.numerical.r_squared == fit.r_squared
        assert result.numerical.dof == fit.dof

        # Validar serialización de FitResult
        data = result.to_dict()
        assert "r_squared" in data["numerical"]


class TestCustomInterpreterReal:
    """Tests de interpretador personalizado con datos reales"""

    def test_custom_interpreter_implementation(self):
        """Test: Implementar interpretador personalizado funciona"""

        class SimpleEfficiencyInterpreter(SemanticInterpreter):
            def validate_input(self, result):
                return isinstance(result, dict) and "D_efficiency" in result

            def interpret(self, numerical_result, context=None):
                d_eff = numerical_result["D_efficiency"]

                if d_eff >= 80:
                    interp = f"Excelente D-efficiency ({d_eff:.1f}%)"
                    conf = "Alta"
                elif d_eff >= 60:
                    interp = f"Buena D-efficiency ({d_eff:.1f}%)"
                    conf = "Moderada"
                else:
                    interp = f"D-efficiency aceptable ({d_eff:.1f}%)"
                    conf = "Moderada-Baja"

                return SemanticResult(
                    numerical=numerical_result,
                    interpretation=interp,
                    reasoning=f"D-efficiency mide precisión de estimación de parámetros",
                    context=context.get("design_info", "") if context else "",
                    warnings=[],
                    recommendations=[],
                    confidence_level=conf
                )

        # Usar con datos reales de doekit
        design = ed.box_behnken({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)})
        effs = ed.efficiencies(design, model=ed.Model.full_quadratic(["X1", "X2", "X3"]))

        # Interpretar
        interpreter = SimpleEfficiencyInterpreter()
        assert interpreter.validate_input(effs)

        semantic = interpreter.interpret(
            effs,
            context={"design_info": f"Box-Behnken {design.n_runs} runs"}
        )

        # Validar resultado
        assert isinstance(semantic, SemanticResult)
        assert "D-efficiency" in semantic.interpretation
        assert "Box-Behnken" in semantic.context
        assert semantic.confidence_level in ["Alta", "Moderada", "Moderada-Baja"]


class TestSemanticRegistryReal:
    """Tests del registro con datos reales"""

    def test_registry_register_and_retrieve(self):
        """Test: Registro funciona para registrar y recuperar interpretadores"""
        registry = SemanticRegistry()

        # Crear interpretador simple
        class TestInterpreter(SemanticInterpreter):
            def validate_input(self, result):
                return isinstance(result, dict) and "test_key" in result

            def interpret(self, numerical_result, context=None):
                return SemanticResult(
                    numerical=numerical_result,
                    interpretation="Test",
                    reasoning="Test reasoning",
                    context=""
                )

        # Registrar
        registry.register("TestDict", TestInterpreter)

        # Crear objeto del tipo registrado
        class TestDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        test_obj = TestDict({"test_key": "value"})

        # Recuperar interpretador
        interpreter = registry.get_interpreter(test_obj)

        assert interpreter is not None
        assert isinstance(interpreter, TestInterpreter)

    def test_registry_interpret_auto_detection(self):
        """Test: Registro auto-detecta e interpreta"""
        registry = SemanticRegistry()

        # Registrar interpretador para dict
        class DictInterpreter(SemanticInterpreter):
            def validate_input(self, result):
                return isinstance(result, dict)

            def interpret(self, numerical_result, context=None):
                return SemanticResult(
                    numerical=numerical_result,
                    interpretation=f"Dict with {len(numerical_result)} keys",
                    reasoning="Auto-interpreted dict",
                    context=""
                )

        registry.register("dict", DictInterpreter)

        # Interpretar automáticamente
        data = {"key1": 1, "key2": 2}
        result = registry.interpret(data)

        assert isinstance(result, SemanticResult)
        assert "2 keys" in result.interpretation


class TestValidationReal:
    """Tests de validación de calidad con datos reales"""

    def test_validate_good_semantic_result(self):
        """Test: Validación aprueba resultado bien formado"""
        good = SemanticResult(
            numerical={"value": 1},
            interpretation="Concise interpretation",  # < 300 chars
            reasoning="Detailed reasoning that explains the result properly",  # < 1000 chars
            context="Context information",  # < 500 chars
            warnings=["Specific warning with enough detail"],  # > 10 chars
            recommendations=["Actionable recommendation with clear steps"]  # > 15 chars
        )

        assert validate_semantic_result(good) == True

    def test_validate_bad_semantic_result_long_interpretation(self):
        """Test: Validación rechaza interpretación muy larga"""
        bad = SemanticResult(
            numerical={"value": 1},
            interpretation="X" * 301,  # > 300 chars
            reasoning="OK",
            context="OK"
        )

        assert validate_semantic_result(bad) == False

    def test_validate_bad_semantic_result_long_reasoning(self):
        """Test: Validación rechaza razonamiento muy largo"""
        bad = SemanticResult(
            numerical={"value": 1},
            interpretation="OK",
            reasoning="X" * 1001,  # > 1000 chars
            context="OK"
        )

        assert validate_semantic_result(bad) == False

    def test_validate_bad_semantic_result_short_warning(self):
        """Test: Validación rechaza warnings muy cortas"""
        bad = SemanticResult(
            numerical={"value": 1},
            interpretation="OK",
            reasoning="OK reasoning",
            context="OK",
            warnings=["Short"]  # < 10 chars
        )

        assert validate_semantic_result(bad) == False

    def test_validate_bad_semantic_result_short_recommendation(self):
        """Test: Validación rechaza recomendaciones muy cortas"""
        bad = SemanticResult(
            numerical={"value": 1},
            interpretation="OK",
            reasoning="OK reasoning",
            context="OK",
            recommendations=["Do it"]  # < 15 chars
        )

        assert validate_semantic_result(bad) == False


class TestIntegrationWithDoekitWorkflow:
    """Tests de integración con workflow completo de doekit"""

    def test_full_workflow_recommend_to_semantic(self):
        """Test: Workflow completo desde recommend_design hasta semantic"""
        # 1. Recomendar diseño (doekit real)
        rec = ed.recommend_design(
            goal="optimization",
            factors=3,
            budget=20,
            model_order="quadratic"
        )

        # 2. Envolver en semántica
        semantic = SemanticResult(
            numerical=rec,
            interpretation=f"Recomendado: {rec.method} con {rec.design.n_runs} corridas",
            reasoning=rec.rationale[:500],  # Tomar primeros 500 chars
            context=f"Optimización con {rec.scenario['n_factors']} factores",
            warnings=[c[:200] for c in rec.caveats[:3]],
            recommendations=[
                f"Ejecutar diseño {rec.method}",
                "Monitorear R² después de primera wave"
            ]
        )

        # 3. Validar semántica
        assert validate_semantic_result(semantic)

        # 4. Verificar que datos originales están preservados
        assert semantic.numerical == rec
        assert semantic.numerical.design.n_runs == rec.design.n_runs

        # 5. Verificar que prompt es útil
        assert rec.method in semantic.context_addition
        assert str(rec.design.n_runs) in semantic.context_addition

    def test_full_workflow_evaluate_to_semantic(self):
        """Test: Workflow desde evaluate hasta semantic"""
        # 1. Crear y evaluar diseño (doekit real)
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(["X1", "X2"])
        evaluation = ed.evaluate(design, model=model)

        # 2. Envolver en semántica
        semantic = SemanticResult(
            numerical=evaluation,
            interpretation=f"D-eff: {evaluation.d_efficiency:.1f}%, G-eff: {evaluation.g_efficiency:.1f}%",
            reasoning=f"Diseño de {design.n_runs} corridas para modelo cuadrático",
            context=f"Central Composite Design, DOF={evaluation.dof}",
            warnings=[] if evaluation.d_efficiency > 70 else ["D-efficiency bajo - considerar más corridas"],
            recommendations=["Proceder si presupuesto limitado"] if evaluation.d_efficiency > 60 else ["Aumentar corridas"]
        )

        # 3. Validar
        assert validate_semantic_result(semantic)
        assert evaluation.d_efficiency in [semantic.numerical.d_efficiency]

    def test_full_workflow_fit_analyze_to_semantic(self):
        """Test: Workflow completo de fit y análisis"""
        # 1. Diseño y datos (doekit real)
        design = ed.plackett_burman(6)
        np.random.seed(42)
        y = (
            10
            + 5 * design.matrix.iloc[:, 0]
            + 3 * design.matrix.iloc[:, 2]
            + np.random.randn(design.n_runs) * 2
        )

        # 2. Ajustar modelo
        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        # 3. Envolver en semántica
        semantic = SemanticResult(
            numerical=fit,
            interpretation=f"Modelo ajustado: R²={fit.r_squared:.3f}",
            reasoning=f"Ajuste con {fit.dof} grados de libertad, sigma²={fit.sigma2:.2f}",
            context=f"Plackett-Burman {design.n_runs} corridas",
            warnings=[] if fit.r_squared > 0.7 else ["R² bajo - revisar modelo"],
            recommendations=["Modelo aceptable"] if fit.r_squared > 0.8 else ["Considerar términos adicionales"]
        )

        # 4. Validar
        assert validate_semantic_result(semantic)
        assert abs(semantic.numerical.r_squared - fit.r_squared) < 0.001


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v", "--tb=short"])
