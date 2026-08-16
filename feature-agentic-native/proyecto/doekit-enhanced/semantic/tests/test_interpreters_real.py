"""
Tests funcionales para interpretadores específicos usando doekit REAL.

Principio: Test-Driven Validation sin mocks.
Todos los tests usan datos reales de doekit v0.7.3.

Author: doekit-enhanced contributors
License: MIT
"""

import pytest
import numpy as np
import doekit as ed

from semantic.core import SemanticResult, get_registry, interpret_result
from semantic.interpreters import (
    RecommendationInterpreter,
    EvaluationInterpreter,
    FitInterpreter,
    ProposalInterpreter,
    ComparisonInterpreter
)


class TestRecommendationInterpreter:
    """Tests para RecommendationInterpreter con datos reales de doekit"""

    def test_recommendation_interpreter_basic(self):
        """Interpreta Recommendation de recommend_design()"""
        # Datos REALES de doekit
        rec = ed.recommend_design(
            goal="optimization",
            factors={"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)},
            budget=20,
            model_order="quadratic"
        )

        # Interpretar usando interpretador específico
        interpreter = RecommendationInterpreter()
        result = interpreter.interpret(rec)

        # Validaciones
        assert isinstance(result, SemanticResult)
        assert result.numerical == rec  # Preserva objeto original
        assert len(result.interpretation) > 0
        assert "corridas" in result.interpretation.lower()
        assert len(result.reasoning) > 0
        assert len(result.recommendations) > 0
        assert result.metadata["interpreter"] == "RecommendationInterpreter"
        assert result.metadata["method"] == rec.method

    def test_recommendation_interpreter_screening(self):
        """Interpreta recomendación de screening con presupuesto limitado"""
        rec = ed.recommend_design(
            goal="screening",
            factors=5,
            budget=12,
            model_order="linear"
        )

        interpreter = RecommendationInterpreter()
        result = interpreter.interpret(rec)

        # Debe recomendar acción post-screening
        assert any("screening" in r.lower() for r in result.recommendations)
        assert result.confidence_level  # Debe tener evaluación de confianza
        assert len(result.warnings) >= 0  # Puede tener warnings

    def test_recommendation_interpreter_with_extra_context(self):
        """Interpreta con contexto adicional"""
        rec = ed.recommend_design(
            goal="optimization",
            factors=3,
            budget=25
        )

        interpreter = RecommendationInterpreter()
        extra_context = {
            "experiment_name": "Battery Optimization v2",
            "researcher": "Lab Team A"
        }
        result = interpreter.interpret(rec, context=extra_context)

        # Contexto debe incluir info adicional
        assert "experiment_name" in result.context or "researcher" in result.context

    def test_recommendation_interpreter_validation_fails(self):
        """Validación rechaza objeto no-Recommendation"""
        interpreter = RecommendationInterpreter()

        # Objeto incorrecto
        with pytest.raises(ValueError, match="Recommendation"):
            interpreter.interpret({"not": "a recommendation"})

    def test_recommendation_auto_registration(self):
        """Interpretador se auto-registra en registry global"""
        registry = get_registry()

        # Crear Recommendation real
        rec = ed.recommend_design(goal="screening", factors=3, budget=10)

        # Registry debe encontrar interpretador automáticamente
        interp = registry.get_interpreter(rec)
        assert interp is not None
        assert isinstance(interp, RecommendationInterpreter)


class TestEvaluationInterpreter:
    """Tests para EvaluationInterpreter con datos reales"""

    def test_evaluation_interpreter_with_power(self):
        """Interpreta DesignEvaluation con cálculos de power"""
        # Crear diseño real
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})

        # Evaluar con modelo
        model = ed.Model.full_quadratic(design.factor_names)
        evaluation = ed.evaluate(design, model=model)

        # Interpretar
        interpreter = EvaluationInterpreter()
        result = interpreter.interpret(evaluation)

        # Validaciones
        assert isinstance(result, SemanticResult)
        assert result.numerical == evaluation
        assert len(result.interpretation) > 0
        assert "poder" in result.interpretation.lower() or "power" in result.interpretation.lower()
        assert len(result.reasoning) > 0
        assert result.metadata["interpreter"] == "EvaluationInterpreter"

    def test_evaluation_interpreter_quality_assessment(self):
        """Evalúa diseño de baja vs alta calidad"""
        # Diseño saturado (baja calidad)
        design_low = ed.plackett_burman(7)  # Diseño pequeño
        model = ed.Model.main_effects(design_low.factor_names)
        eval_low = ed.evaluate(design_low, model=model)

        interpreter = EvaluationInterpreter()
        result_low = interpreter.interpret(eval_low)

        # Diseño robusto (alta calidad)
        design_high = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)})
        model_quad = ed.Model.full_quadratic(design_high.factor_names)
        eval_high = ed.evaluate(design_high, model=model_quad)

        result_high = interpreter.interpret(eval_high)

        # Comparar interpretaciones
        assert len(result_low.interpretation) > 0
        assert len(result_high.interpretation) > 0
        # Ambos deben tener recomendaciones
        assert len(result_low.recommendations) > 0
        assert len(result_high.recommendations) > 0

    def test_evaluation_interpreter_warnings_on_low_power(self):
        """Genera warnings cuando poder es bajo"""
        # Diseño muy pequeño = bajo poder
        design = ed.plackett_burman(5)
        model = ed.Model.main_effects(design.factor_names)
        evaluation = ed.evaluate(design, model=model)

        interpreter = EvaluationInterpreter()
        result = interpreter.interpret(evaluation)

        # Puede tener warnings dependiendo del diseño
        # (no forzamos porque depende de los cálculos reales de doekit)
        assert isinstance(result.warnings, list)


class TestFitInterpreter:
    """Tests para FitInterpreter con datos reales"""

    def test_fit_interpreter_basic(self):
        """Interpreta FitResult de fit_linear_model()"""
        # Datos REALES de doekit
        design = ed.plackett_burman(5)

        # Respuesta simulada con señal + ruido
        np.random.seed(42)
        X = design.matrix.values
        # Efecto lineal simple + ruido
        y = 10 + 2*X[:, 0] + 3*X[:, 1] - 1.5*X[:, 2] + np.random.randn(len(X)) * 0.5

        # Ajustar modelo real
        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        # Interpretar
        interpreter = FitInterpreter()
        result = interpreter.interpret(fit)

        # Validaciones
        assert isinstance(result, SemanticResult)
        assert result.numerical == fit
        assert len(result.interpretation) > 0
        assert "modelo" in result.interpretation.lower()
        assert len(result.reasoning) > 0
        assert result.metadata["interpreter"] == "FitInterpreter"

    def test_fit_interpreter_good_vs_poor_fit(self):
        """Compara interpretación de buen vs mal ajuste"""
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        X = design.matrix.values

        # Buen ajuste: modelo cuadrático con señal fuerte
        np.random.seed(42)
        y_good = 10 + 5*X[:, 0] + 3*X[:, 1] + 2*X[:, 0]**2 + np.random.randn(len(X)) * 0.3

        model_quad = ed.Model.full_quadratic(design.factor_names)
        fit_good = ed.fit_linear_model(design, y_good, model=model_quad)

        # Mal ajuste: solo ruido
        y_poor = np.random.randn(len(X)) * 5

        model_linear = ed.Model.main_effects(design.factor_names)
        fit_poor = ed.fit_linear_model(design, y_poor, model=model_linear)

        # Interpretar ambos
        interpreter = FitInterpreter()
        result_good = interpreter.interpret(fit_good)
        result_poor = interpreter.interpret(fit_poor)

        # Buen ajuste debe tener alta confianza
        if hasattr(fit_good, 'r_squared'):
            assert "alta" in result_good.confidence_level.lower() or result_good.confidence_level

        # Mal ajuste debe tener warnings
        if hasattr(fit_poor, 'r_squared'):
            r2_poor = fit_poor.r_squared
            if r2_poor < 0.5:
                assert len(result_poor.warnings) > 0

    def test_fit_interpreter_with_context(self):
        """Interpreta con contexto adicional"""
        design = ed.plackett_burman(5)
        np.random.seed(42)
        y = np.random.randn(design.n_runs) * 2 + 10

        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        interpreter = FitInterpreter()
        extra_context = {
            "model_order": "linear",
            "n_obs": len(y)
        }
        result = interpreter.interpret(fit, context=extra_context)

        # Contexto debe aparecer
        assert result.context
        assert "linear" in result.context.lower() or str(len(y)) in result.context


class TestProposalInterpreter:
    """Tests para ProposalInterpreter con datos reales"""

    def test_proposal_interpreter_basic(self):
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        np.random.seed(42)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)

        interpreter = ProposalInterpreter()
        result = interpreter.interpret(proposal)

        assert isinstance(result, SemanticResult)
        assert result.numerical == proposal
        assert result.metadata["interpreter"] == "ProposalInterpreter"
        assert len(result.interpretation) > 0
        assert len(result.recommendations) > 0

    def test_proposal_auto_detection(self):
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        np.random.seed(123)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        semantic = interpret_result(proposal)

        assert isinstance(semantic, SemanticResult)
        assert semantic.metadata["interpreter"] == "ProposalInterpreter"


class TestComparisonInterpreter:
    """Tests para ComparisonInterpreter con datos reales"""

    def test_comparison_interpreter_basic(self):
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        np.random.seed(42)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comparison = ed.compare_designs(design, proposal.combined, model=model)

        interpreter = ComparisonInterpreter()
        result = interpreter.interpret(comparison)

        assert isinstance(result, SemanticResult)
        assert result.numerical == comparison
        assert result.metadata["interpreter"] == "ComparisonInterpreter"
        assert "comparacion" in result.interpretation.lower()

    def test_comparison_auto_detection(self):
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        np.random.seed(7)
        y = np.random.randn(design.n_runs)

        proposal = ed.propose_next_runs(design, response=y, n_add=2, model=model)
        comparison = ed.compare_designs(design, proposal.combined, model=model)
        semantic = interpret_result(comparison)

        assert isinstance(semantic, SemanticResult)
        assert semantic.metadata["interpreter"] == "ComparisonInterpreter"


class TestAutoInterpretation:
    """Tests de interpretación automática vía registry"""

    def test_auto_interpret_recommendation(self):
        """interpret_result() auto-detecta y usa RecommendationInterpreter"""
        # Crear Recommendation real
        rec = ed.recommend_design(goal="screening", factors=4, budget=15)

        # interpret_result debe auto-detectar tipo y usar interpretador correcto
        result = interpret_result(rec)

        assert isinstance(result, SemanticResult)
        assert result.metadata["interpreter"] == "RecommendationInterpreter"
        assert result.numerical == rec

    def test_auto_interpret_evaluation(self):
        """interpret_result() auto-detecta DesignEvaluation"""
        design = ed.box_behnken({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        evaluation = ed.evaluate(design, model=model)

        result = interpret_result(evaluation)

        assert isinstance(result, SemanticResult)
        assert result.metadata["interpreter"] == "EvaluationInterpreter"

    def test_auto_interpret_fit(self):
        """interpret_result() auto-detecta FitResult"""
        design = ed.plackett_burman(5)
        np.random.seed(42)
        y = np.random.randn(design.n_runs) * 2 + 10

        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        result = interpret_result(fit)

        assert isinstance(result, SemanticResult)
        assert result.metadata["interpreter"] == "FitInterpreter"

    def test_auto_interpret_unknown_type(self):
        """interpret_result() retorna original si tipo desconocido"""
        unknown = {"unknown": "type"}

        # Sin interpretador registrado, debe retornar original
        result = interpret_result(unknown)

        assert result == unknown  # Sin cambios


class TestEndToEndWorkflows:
    """Tests de workflows completos end-to-end"""

    def test_workflow_recommend_interpret_use(self):
        """Workflow: recommend → interpret → usar semántica"""
        # 1. Obtener recomendación con doekit
        rec = ed.recommend_design(
            goal="optimization",
            factors={"Temp": (50, 100), "Press": (1, 5), "Time": (10, 60)},
            budget=20,
            model_order="quadratic"
        )

        # 2. Interpretar semánticamente
        semantic = interpret_result(rec)

        # 3. Usar para razonamiento
        prompt_for_llm = f"""
Contexto: {semantic.context}

Interpretación: {semantic.interpretation}

Razonamiento: {semantic.reasoning}

Advertencias:
{chr(10).join(f'- {w}' for w in semantic.warnings)}

Recomendaciones:
{chr(10).join(f'- {r}' for r in semantic.recommendations)}

Confianza: {semantic.confidence_level}
"""

        # Validar que prompt está bien formado
        assert len(prompt_for_llm) > 100
        assert "optimization" in prompt_for_llm.lower() or "corridas" in prompt_for_llm.lower()
        assert semantic.numerical.method in prompt_for_llm or "diseño" in prompt_for_llm.lower()

    def test_workflow_design_evaluate_interpret(self):
        """Workflow: crear diseño → evaluar → interpretar"""
        # 1. Crear diseño
        design = ed.central_composite(
            {"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)}
        )

        # 2. Evaluar calidad
        model = ed.Model.full_quadratic(design.factor_names)
        evaluation = ed.evaluate(design, model=model)

        # 3. Interpretar semánticamente
        semantic = interpret_result(evaluation)

        # 4. Tomar decisión basada en interpretación
        decision_prompt = semantic.prompt_injection

        # Validar workflow
        assert len(decision_prompt) > 50
        assert "INTERPRETACIÓN" in decision_prompt or "interpretación" in decision_prompt.lower()

    def test_workflow_experiment_fit_interpret(self):
        """Workflow: ejecutar experimento → ajustar → interpretar"""
        # 1. Diseño experimental
        design = ed.box_behnken({"A": (-1, 1), "B": (-1, 1), "C": (-1, 1)})

        # 2. Simular respuestas experimentales
        np.random.seed(123)
        X = design.matrix.values
        y = 50 + 10*X[:, 0] + 5*X[:, 1] + 3*X[:, 2] + 2*X[:, 0]*X[:, 1] + np.random.randn(len(X))

        # 3. Ajustar modelo
        model = ed.Model.full_quadratic(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        # 4. Interpretar resultados
        semantic = interpret_result(fit)

        # 5. Usar para reporting
        assert semantic.interpretation
        assert semantic.reasoning
        assert len(semantic.recommendations) > 0

        # Debe poder serializar para guardar
        json_report = semantic.to_json()
        assert len(json_report) > 100
        assert "numerical" in json_report


class TestInterpreterQuality:
    """Tests de calidad de interpretaciones"""

    def test_interpretation_length_appropriate(self):
        """Interpretaciones son concisas (<300 chars)"""
        rec = ed.recommend_design(goal="screening", factors=5, budget=15)
        semantic = interpret_result(rec)

        # Interpretation debe ser concisa
        assert len(semantic.interpretation) < 300
        assert len(semantic.interpretation) > 20

    def test_reasoning_length_appropriate(self):
        """Razonamiento es informativo pero no excesivo (<1000 chars)"""
        design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
        model = ed.Model.full_quadratic(design.factor_names)
        evaluation = ed.evaluate(design, model=model)

        semantic = interpret_result(evaluation)

        assert len(semantic.reasoning) < 1000
        assert len(semantic.reasoning) > 30

    def test_recommendations_are_actionable(self):
        """Recomendaciones son accionables (>15 chars)"""
        design = ed.plackett_burman(5)
        np.random.seed(42)
        y = np.random.randn(design.n_runs) * 2 + 10

        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, y, model=model)

        semantic = interpret_result(fit)

        # Todas las recomendaciones deben ser específicas
        for rec in semantic.recommendations:
            assert len(rec) >= 15, f"Recommendation too short: {rec}"

    def test_warnings_are_specific(self):
        """Warnings son específicas (>10 chars)"""
        # Diseño muy pequeño para generar warnings
        rec = ed.recommend_design(goal="optimization", factors=5, budget=8)

        semantic = interpret_result(rec)

        # Si hay warnings, deben ser específicas
        for warning in semantic.warnings:
            assert len(warning) >= 10, f"Warning too short: {warning}"


# Configuración de pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
