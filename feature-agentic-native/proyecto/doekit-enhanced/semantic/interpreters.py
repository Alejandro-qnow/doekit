"""
semantic.interpreters - Interpretadores especializados para resultados de doekit

Provee interpretadores concretos para cada tipo de resultado que doekit retorna,
generando semántica contextualmente apropiada.

Theory:
    Cada interpretador aplica heurísticas específicas del dominio DoE para
    convertir datos numéricos en interpretaciones accionables.

Author: doekit-enhanced contributors
License: MIT
"""

from typing import Any, Optional, Dict, List
from semantic.core import SemanticInterpreter, SemanticResult


class RecommendationInterpreter(SemanticInterpreter):
    """
    Interpreta objetos Recommendation de doekit.recommend_design().

    Contexto:
        Recommendation contiene método sugerido, diseño concreto, escenario,
        rationale, y caveats. Interpreta el trade-off entre eficiencia y presupuesto.

    Theory:
        - Contrastive explanation: Por qué este diseño vs alternativas
        - Selective: Enfoca en factores críticos de decisión
        - Social: Usa lenguaje apropiado para contexto experimental
    """

    def validate_input(self, result: Any) -> bool:
        """Valida que sea un objeto Recommendation de doekit"""
        return (
            hasattr(result, 'method') and
            hasattr(result, 'design') and
            hasattr(result, 'rationale')
        )

    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        """
        Interpreta Recommendation en lenguaje accionable.

        Args:
            numerical_result: Objeto Recommendation de doekit
            context: Contexto adicional (opcional)

        Returns:
            SemanticResult con interpretación completa
        """
        if not self.validate_input(numerical_result):
            raise ValueError(
                "Input debe ser Recommendation de doekit.recommend_design()"
            )

        rec = numerical_result

        # Construir componentes semánticos
        interpretation = self._build_interpretation(rec)
        reasoning = self._build_reasoning(rec)
        ctx = self._build_context(rec, context)
        warnings = self._extract_warnings(rec)
        recommendations = self._extract_recommendations(rec)
        confidence = self._assess_confidence(rec)

        return SemanticResult(
            numerical=rec,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recommendations,
            confidence_level=confidence,
            metadata={
                "interpreter": "RecommendationInterpreter",
                "method": rec.method,
                "n_runs": rec.design.n_runs,
                "goal": rec.scenario.get('goal', 'unknown')
            }
        )

    def _build_interpretation(self, rec) -> str:
        """Resumen de qué se recomienda"""
        method = rec.method.replace('_', ' ').title()
        n_runs = rec.design.n_runs

        return (
            f"Se recomienda diseño {method} con {n_runs} corridas experimentales"
        )

    def _build_reasoning(self, rec) -> str:
        """Por qué esta recomendación"""
        # Usar rationale de doekit como base
        base = rec.rationale

        # Agregar contexto cuantitativo si está disponible
        extras = []
        if hasattr(rec.design, 'n_factors'):
            extras.append(
                f"para {rec.design.n_factors} factores experimentales"
            )

        if extras:
            return f"{base}. Este diseño es apropiado {' '.join(extras)}."
        return base

    def _build_context(self, rec, extra_context: Optional[Dict] = None) -> str:
        """Contexto del problema"""
        parts = []

        # Escenario
        scenario = rec.scenario
        if 'goal' in scenario:
            parts.append(f"Objetivo: {scenario['goal']}")
        if 'budget' in scenario:
            parts.append(f"Presupuesto: {scenario['budget']} corridas")
        if 'model_order' in scenario:
            parts.append(f"Modelo: {scenario['model_order']}")

        # Contexto adicional
        if extra_context:
            for key, value in extra_context.items():
                parts.append(f"{key}: {value}")

        return ". ".join(parts) if parts else ""

    def _extract_warnings(self, rec) -> List[str]:
        """Advertencias importantes"""
        warnings = []

        # Usar caveats de doekit
        if hasattr(rec, 'caveats') and rec.caveats:
            # Tomar primeros 3 caveats más importantes
            for caveat in rec.caveats[:3]:
                if len(caveat) >= 10:  # Mínimo de calidad
                    warnings.append(caveat)

        # Agregar advertencia si diseño es muy pequeño
        if rec.design.n_runs < 8:
            warnings.append(
                f"Diseño pequeño ({rec.design.n_runs} corridas) - "
                "considerar réplicas si variabilidad es alta"
            )

        return warnings

    def _extract_recommendations(self, rec) -> List[str]:
        """Acciones recomendadas"""
        recs = []

        # Acción primaria
        recs.append(
            f"Ejecutar {rec.design.n_runs} experimentos según diseño {rec.method}"
        )

        # Acciones secundarias según contexto
        if rec.scenario.get('goal') == 'screening':
            recs.append(
                "Después de screening, usar factores significativos "
                "en diseño de optimización"
            )
        elif rec.scenario.get('goal') == 'optimization':
            recs.append(
                "Analizar superficie de respuesta para identificar óptimo"
            )

        # Si hay presupuesto restante
        budget = rec.scenario.get('budget', 0)
        used = rec.design.n_runs
        if budget > used and budget - used >= 3:
            remaining = budget - used
            recs.append(
                f"Considerar usar {remaining} corridas restantes "
                "para réplicas o puntos de validación"
            )

        return recs

    def _assess_confidence(self, rec) -> str:
        """Evalúa confianza en recomendación"""
        # Heurística: más corridas relativo a complejidad = más confianza
        n_runs = rec.design.n_runs

        if hasattr(rec.design, 'n_factors'):
            n_factors = rec.design.n_factors
            ratio = n_runs / n_factors

            if ratio >= 5:
                return (
                    "Alta - diseño robusto con suficientes grados de libertad "
                    "para estimación precisa"
                )
            elif ratio >= 3:
                return (
                    "Moderada - diseño balanceado entre eficiencia y presupuesto"
                )
            else:
                return (
                    "Limitada - diseño económico pero con menor precisión. "
                    "Considerar aumentar corridas si es crítico"
                )

        # Si no tenemos n_factors, confianza moderada por default
        return "Moderada - basada en recomendación de doekit"


class EvaluationInterpreter(SemanticInterpreter):
    """
    Interpreta objetos DesignEvaluation de doekit.evaluate().

    Contexto:
        DesignEvaluation contiene métricas de calidad: power, efficiencies,
        correlation, etc. Interpreta si diseño es adecuado para objetivo.

    Theory:
        - Quantitative thresholds: Umbrales establecidos en literatura DoE
        - Trade-off analysis: Balance entre métricas competidoras
    """

    def validate_input(self, result: Any) -> bool:
        """Valida que sea DesignEvaluation"""
        return (
            hasattr(result, 'power') or
            hasattr(result, 'correlation') or
            hasattr(result, 'summary')
        )

    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        """Interpreta DesignEvaluation"""
        if not self.validate_input(numerical_result):
            raise ValueError("Input debe ser DesignEvaluation de doekit.evaluate()")

        eval_result = numerical_result

        interpretation = self._build_interpretation(eval_result)
        reasoning = self._build_reasoning(eval_result)
        ctx = self._build_context(eval_result, context)
        warnings = self._extract_warnings(eval_result)
        recommendations = self._extract_recommendations(eval_result)
        confidence = self._assess_confidence(eval_result)

        return SemanticResult(
            numerical=eval_result,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recommendations,
            confidence_level=confidence,
            metadata={
                "interpreter": "EvaluationInterpreter",
                "has_power": hasattr(eval_result, 'power'),
                "has_correlation": hasattr(eval_result, 'correlation')
            }
        )

    def _build_interpretation(self, eval_result) -> str:
        """Resumen de calidad del diseño"""
        # Analizar power si está disponible
        if hasattr(eval_result, 'power') and eval_result.power is not None:
            # power es una pandas Series
            avg_power = eval_result.power.mean()
            if avg_power >= 0.8:
                quality = "excelente"
            elif avg_power >= 0.6:
                quality = "buena"
            else:
                quality = "limitada"

            return f"Diseño tiene poder estadístico {quality} (promedio: {avg_power:.1%})"

        # Si no hay power, evaluar por d_efficiency
        if hasattr(eval_result, 'd_efficiency') and eval_result.d_efficiency is not None:
            d_eff = eval_result.d_efficiency
            if d_eff >= 0.8:
                return f"Diseño eficiente con D-efficiency de {d_eff:.1%}"
            elif d_eff >= 0.5:
                return f"Diseño aceptable con D-efficiency de {d_eff:.1%}"
            else:
                return f"Diseño limitado con D-efficiency de {d_eff:.1%}"

        return "Evaluación de diseño completada"

    def _build_reasoning(self, eval_result) -> str:
        """Justificación técnica"""
        parts = []

        if hasattr(eval_result, 'power') and eval_result.power is not None:
            # power es pandas Series
            powers = eval_result.power
            high_power = powers[powers >= 0.8].index.tolist()
            low_power = powers[powers < 0.6].index.tolist()

            if high_power:
                parts.append(
                    f"Factores {', '.join(high_power[:3])} tienen poder estadístico alto (>80%) "
                    "para detectar efectos significativos"
                )

            if low_power:
                parts.append(
                    f"Factores {', '.join(low_power[:3])} tienen poder limitado (<60%) - "
                    "pueden requerir más réplicas"
                )

        if hasattr(eval_result, 'vif') and eval_result.vif is not None:
            # vif es pandas Series de Variance Inflation Factors
            max_vif = eval_result.vif.max()
            if max_vif > 5:
                parts.append(
                    f"VIF máximo de {max_vif:.1f} indica multicolinealidad potencial"
                )

        if not parts:
            parts.append("Diseño evaluado contra criterios estándar de DoE")

        return ". ".join(parts)

    def _build_context(self, eval_result, extra_context: Optional[Dict] = None) -> str:
        """Contexto de evaluación"""
        parts = []

        if hasattr(eval_result, 'n_runs'):
            parts.append(f"Diseño con {eval_result.n_runs} corridas")

        if extra_context:
            if 'model' in extra_context:
                parts.append(f"Modelo: {extra_context['model']}")
            if 'alpha' in extra_context:
                parts.append(f"Nivel de significancia: {extra_context['alpha']}")

        return ". ".join(parts) if parts else ""

    def _extract_warnings(self, eval_result) -> List[str]:
        """Advertencias sobre el diseño"""
        warnings = []

        # Advertencias sobre poder estadístico
        if hasattr(eval_result, 'power') and eval_result.power is not None:
            avg_power = eval_result.power.mean()
            if avg_power < 0.6:
                warnings.append(
                    f"Poder estadístico promedio bajo ({avg_power:.1%}) - "
                    "alta probabilidad de no detectar efectos reales (error tipo II)"
                )

        # Advertencias sobre VIF (multicolinealidad)
        if hasattr(eval_result, 'vif') and eval_result.vif is not None:
            max_vif = eval_result.vif.max()
            if max_vif > 10:
                warnings.append(
                    f"VIF máximo muy alto ({max_vif:.1f}) - "
                    "multicolinealidad severa puede afectar estimación de efectos"
                )

        # Advertencias sobre D-efficiency
        if hasattr(eval_result, 'd_efficiency') and eval_result.d_efficiency is not None:
            if eval_result.d_efficiency < 0.5:
                warnings.append(
                    f"D-efficiency baja ({eval_result.d_efficiency:.1%}) - "
                    "considerar diseño más eficiente si presupuesto lo permite"
                )

        return warnings

    def _extract_recommendations(self, eval_result) -> List[str]:
        """Recomendaciones para mejorar diseño"""
        recs = []

        if hasattr(eval_result, 'power') and eval_result.power is not None:
            avg_power = eval_result.power.mean()

            if avg_power >= 0.8:
                recs.append("Diseño es adecuado - proceder con experimentos")
            elif avg_power >= 0.6:
                recs.append(
                    "Diseño aceptable - considerar réplicas adicionales "
                    "para factores críticos"
                )
            else:
                recs.append(
                    "Aumentar número de corridas o réplicas para mejorar poder estadístico"
                )
        elif hasattr(eval_result, 'd_efficiency') and eval_result.d_efficiency is not None:
            if eval_result.d_efficiency >= 0.7:
                recs.append("Diseño eficiente - proceder con experimentos")
            else:
                recs.append("Considerar diseño D-optimal para mayor eficiencia")

        if hasattr(eval_result, 'vif') and eval_result.vif is not None:
            max_vif = eval_result.vif.max()
            if max_vif > 5:
                recs.append(
                    "Considerar diseño ortogonal para reducir multicolinealidad"
                )

        if not recs:
            recs.append("Revisar trade-offs entre costo y precisión antes de ejecutar")

        return recs

    def _assess_confidence(self, eval_result) -> str:
        """Confianza en evaluación"""
        if hasattr(eval_result, 'power') and eval_result.power is not None:
            avg_power = eval_result.power.mean()
            return (
                f"Alta - basada en cálculos de poder estadístico "
                f"(promedio {avg_power:.1%})"
            )

        return "Moderada - basada en métricas estructurales del diseño"


class FitInterpreter(SemanticInterpreter):
    """
    Interpreta objetos FitResult de doekit.fit_linear_model().

    Contexto:
        FitResult contiene coeficientes, R², residuos, efectos significativos.
        Interpreta calidad de ajuste y significancia estadística.

    Theory:
        - Statistical significance: p-valores y intervalos de confianza
        - Effect size: Magnitud práctica vs significancia estadística
    """

    def validate_input(self, result: Any) -> bool:
        """Valida que sea FitResult"""
        return (
            hasattr(result, 'coefficients') or
            hasattr(result, 'r_squared') or
            hasattr(result, 'summary')
        )

    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        """Interpreta FitResult"""
        if not self.validate_input(numerical_result):
            raise ValueError("Input debe ser FitResult de doekit.fit_linear_model()")

        fit = numerical_result

        interpretation = self._build_interpretation(fit)
        reasoning = self._build_reasoning(fit)
        ctx = self._build_context(fit, context)
        warnings = self._extract_warnings(fit)
        recommendations = self._extract_recommendations(fit)
        confidence = self._assess_confidence(fit)

        return SemanticResult(
            numerical=fit,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recommendations,
            confidence_level=confidence,
            metadata={
                "interpreter": "FitInterpreter",
                "has_r_squared": hasattr(fit, 'r_squared'),
                "has_coefficients": hasattr(fit, 'coefficients')
            }
        )

    def _build_interpretation(self, fit) -> str:
        """Resumen de calidad de ajuste"""
        if hasattr(fit, 'r_squared'):
            r2 = fit.r_squared
            if r2 >= 0.9:
                quality = "excelente"
            elif r2 >= 0.7:
                quality = "bueno"
            elif r2 >= 0.5:
                quality = "moderado"
            else:
                quality = "pobre"

            return f"Modelo explica {r2:.1%} de variabilidad (ajuste {quality})"

        if hasattr(fit, 'coefficients'):
            n_coefs = len(fit.coefficients)
            return f"Modelo ajustado con {n_coefs} coeficientes estimados"

        return "Modelo lineal ajustado a datos experimentales"

    def _build_reasoning(self, fit) -> str:
        """Justificación del ajuste"""
        parts = []

        if hasattr(fit, 'r_squared'):
            r2 = fit.r_squared
            if r2 >= 0.7:
                parts.append(
                    f"R² de {r2:.1%} indica que modelo captura "
                    "la mayoría de variación sistemática en respuesta"
                )
            else:
                parts.append(
                    f"R² de {r2:.1%} sugiere que hay variabilidad no capturada - "
                    "considerar términos adicionales o efectos no lineales"
                )

        if hasattr(fit, 'coefficients'):
            # Identificar coeficientes grandes (si son numéricos)
            try:
                coef_values = [abs(v) for v in fit.coefficients.values() if isinstance(v, (int, float))]
                if coef_values:
                    max_coef = max(coef_values)
                    parts.append(
                        f"Coeficiente máximo: {max_coef:.3f} indica "
                        "magnitud de efectos principales"
                    )
            except:
                pass

        if not parts:
            parts.append("Modelo ajustado usando mínimos cuadrados ordinarios")

        return ". ".join(parts)

    def _build_context(self, fit, extra_context: Optional[Dict] = None) -> str:
        """Contexto del ajuste"""
        parts = []

        if extra_context:
            if 'model_order' in extra_context:
                parts.append(f"Modelo {extra_context['model_order']}")
            if 'n_obs' in extra_context:
                parts.append(f"{extra_context['n_obs']} observaciones")

        return ". ".join(parts) if parts else ""

    def _extract_warnings(self, fit) -> List[str]:
        """Advertencias sobre el ajuste"""
        warnings = []

        if hasattr(fit, 'r_squared'):
            r2 = fit.r_squared
            if r2 < 0.5:
                warnings.append(
                    f"R² bajo ({r2:.1%}) - modelo explica menos de 50% de variación. "
                    "Revisar especificación del modelo o calidad de datos"
                )
            elif r2 > 0.98 and hasattr(fit, 'coefficients'):
                # R² muy alto puede indicar overfitting
                warnings.append(
                    f"R² muy alto ({r2:.1%}) - validar que no hay overfitting "
                    "usando validación cruzada o conjunto de test"
                )

        return warnings

    def _extract_recommendations(self, fit) -> List[str]:
        """Recomendaciones para usar modelo"""
        recs = []

        if hasattr(fit, 'r_squared'):
            r2 = fit.r_squared
            if r2 >= 0.7:
                recs.append(
                    "Modelo es adecuado para predicción y optimización de respuesta"
                )
                recs.append(
                    "Validar supuestos de residuos (normalidad, homocedasticidad) "
                    "antes de inferencia"
                )
            else:
                recs.append(
                    "Explorar términos de interacción o no lineales para mejorar ajuste"
                )
                recs.append(
                    "Considerar transformaciones de respuesta (log, sqrt) si hay heterocedasticidad"
                )

        if not recs:
            recs.append("Usar modelo para exploración inicial de superficie de respuesta")

        return recs

    def _assess_confidence(self, fit) -> str:
        """Confianza en modelo ajustado"""
        if hasattr(fit, 'r_squared'):
            r2 = fit.r_squared
            if r2 >= 0.9:
                return f"Alta - R² de {r2:.1%} indica ajuste muy bueno"
            elif r2 >= 0.7:
                return f"Moderada-Alta - R² de {r2:.1%} es aceptable para DoE"
            elif r2 >= 0.5:
                return f"Moderada - R² de {r2:.1%} sugiere mejoras posibles"
            else:
                return f"Baja - R² de {r2:.1%} indica modelo inadecuado"

        return "Moderada - basada en estimación por mínimos cuadrados"


class ProposalInterpreter(SemanticInterpreter):
    """
    Interpreta objetos NextRunsProposal de doekit.propose_next_runs().

    Enfatiza la decision de expandir diseño y el valor de la propuesta.
    """

    def validate_input(self, result: Any) -> bool:
        return (
            hasattr(result, 'comparison') and
            hasattr(result, 'added') and
            hasattr(result, 'combined') and
            hasattr(result, 'rationale')
        )

    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        if not self.validate_input(numerical_result):
            raise ValueError("Input debe ser NextRunsProposal de doekit.propose_next_runs()")

        proposal = numerical_result
        comparison = getattr(proposal, 'comparison', None)

        interpretation = self._build_interpretation(proposal, comparison)
        reasoning = self._build_reasoning(proposal, comparison)
        ctx = self._build_context(proposal, context)
        warnings = self._extract_warnings(proposal, comparison)
        recommendations = self._extract_recommendations(proposal, comparison)
        confidence = self._assess_confidence(proposal, comparison)

        return SemanticResult(
            numerical=proposal,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recommendations,
            confidence_level=confidence,
            metadata={
                "interpreter": "ProposalInterpreter",
                "n_added": getattr(getattr(proposal, 'added', None), 'n_runs', None),
                "criterion": getattr(proposal, 'criterion', None),
                "worth_it": getattr(comparison, 'worth_it', None)
            }
        )

    def _build_interpretation(self, proposal: Any, comparison: Any) -> str:
        n_added = getattr(getattr(proposal, 'added', None), 'n_runs', None)
        if n_added is None:
            n_added = "N"

        if comparison is not None and getattr(comparison, 'worth_it', False):
            return f"La propuesta de {n_added} corridas adicionales es favorable"

        return f"La propuesta de {n_added} corridas adicionales requiere cautela"

    def _build_reasoning(self, proposal: Any, comparison: Any) -> str:
        base = getattr(proposal, 'rationale', 'Se generaron puntos adicionales para mejorar el diseño')
        parts = [base]

        if comparison is not None and hasattr(comparison, 'delta'):
            delta = comparison.delta or {}
            d_eff = delta.get('D_efficiency')
            mean_power = delta.get('mean_power')

            delta_bits = []
            if isinstance(d_eff, (int, float)):
                delta_bits.append(f"Delta D-efficiency: {d_eff:+.2f}")
            if isinstance(mean_power, (int, float)):
                delta_bits.append(f"Delta mean power: {mean_power:+.3f}")

            if delta_bits:
                parts.append(". ".join(delta_bits))

        return ". ".join(parts)

    def _build_context(self, proposal: Any, extra_context: Optional[Dict[str, Any]] = None) -> str:
        parts = []
        criterion = getattr(proposal, 'criterion', None)
        sigma_hat = getattr(proposal, 'sigma_hat', None)

        if criterion:
            parts.append(f"Criterio de propuesta: {criterion}")
        if sigma_hat is not None:
            parts.append(f"Sigma estimada: {sigma_hat:.4f}" if isinstance(sigma_hat, (int, float)) else f"Sigma estimada: {sigma_hat}")

        if extra_context:
            for key, value in extra_context.items():
                parts.append(f"{key}: {value}")

        return ". ".join(parts)

    def _extract_warnings(self, proposal: Any, comparison: Any) -> List[str]:
        warnings: List[str] = []

        caveats = getattr(proposal, 'caveats', []) or []
        for caveat in caveats[:3]:
            if len(caveat) >= 10:
                warnings.append(caveat)

        if comparison is not None and hasattr(comparison, 'delta'):
            delta = comparison.delta or {}
            if delta.get('G_efficiency', 0) < 0:
                warnings.append(
                    "La G-efficiency disminuye con la propuesta; revisar trade-off entre estimacion y prediccion"
                )

        return warnings

    def _extract_recommendations(self, proposal: Any, comparison: Any) -> List[str]:
        n_added = getattr(getattr(proposal, 'added', None), 'n_runs', None)
        if n_added is None:
            n_added = 0

        recs = [
            f"Ejecutar las {n_added} corridas sugeridas y re-evaluar eficiencias en la siguiente wave"
        ]

        if comparison is not None and not getattr(comparison, 'worth_it', True):
            recs.append(
                "Si el costo es alto, considerar reducir n_add o cambiar criterio de propuesta"
            )
        else:
            recs.append(
                "Actualizar modelo con nuevos datos y verificar estabilidad de terminos activos"
            )

        return recs

    def _assess_confidence(self, proposal: Any, comparison: Any) -> str:
        if comparison is None or not hasattr(comparison, 'delta'):
            return "Moderada - basada en rationale y estructura de la propuesta"

        delta = comparison.delta or {}
        d_eff = delta.get('D_efficiency', 0)
        mean_power = delta.get('mean_power', 0)

        signal = 0
        if isinstance(d_eff, (int, float)) and d_eff > 0:
            signal += 1
        if isinstance(mean_power, (int, float)) and mean_power > 0:
            signal += 1

        if signal == 2:
            return "Alta - mejoras consistentes en eficiencia y poder"
        if signal == 1:
            return "Moderada - mejora parcial, con trade-offs"
        return "Baja - beneficio limitado o incierto"


class ComparisonInterpreter(SemanticInterpreter):
    """
    Interpreta objetos DesignComparison de doekit.compare_designs().
    """

    def validate_input(self, result: Any) -> bool:
        return (
            hasattr(result, 'delta') and
            hasattr(result, 'worth_it') and
            hasattr(result, 'a_label') and
            hasattr(result, 'b_label')
        )

    def interpret(
        self,
        numerical_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticResult:
        if not self.validate_input(numerical_result):
            raise ValueError("Input debe ser DesignComparison de doekit.compare_designs()")

        comp = numerical_result
        interpretation = self._build_interpretation(comp)
        reasoning = self._build_reasoning(comp)
        ctx = self._build_context(comp, context)
        warnings = self._extract_warnings(comp)
        recommendations = self._extract_recommendations(comp)
        confidence = self._assess_confidence(comp)

        return SemanticResult(
            numerical=comp,
            interpretation=interpretation,
            reasoning=reasoning,
            context=ctx,
            warnings=warnings,
            recommendations=recommendations,
            confidence_level=confidence,
            metadata={
                "interpreter": "ComparisonInterpreter",
                "a_label": comp.a_label,
                "b_label": comp.b_label,
                "worth_it": comp.worth_it
            }
        )

    def _build_interpretation(self, comp: Any) -> str:
        action = "conviene" if comp.worth_it else "no conviene claramente"
        return f"La comparacion entre {comp.a_label} y {comp.b_label} indica que {action} migrar al diseno B"

    def _build_reasoning(self, comp: Any) -> str:
        delta = comp.delta or {}
        d_eff = delta.get('D_efficiency')
        a_eff = delta.get('A_efficiency')
        g_eff = delta.get('G_efficiency')

        parts = [
            f"Se evaluaron diferencias netas de eficiencia entre {comp.a_label} y {comp.b_label}"
        ]

        metrics = []
        if isinstance(d_eff, (int, float)):
            metrics.append(f"D-eff {d_eff:+.2f}")
        if isinstance(a_eff, (int, float)):
            metrics.append(f"A-eff {a_eff:+.2f}")
        if isinstance(g_eff, (int, float)):
            metrics.append(f"G-eff {g_eff:+.2f}")

        if metrics:
            parts.append("Cambios: " + ", ".join(metrics))

        return ". ".join(parts)

    def _build_context(self, comp: Any, extra_context: Optional[Dict[str, Any]] = None) -> str:
        parts = []
        delta = comp.delta or {}
        n_runs = delta.get('n_runs')
        if isinstance(n_runs, (int, float)):
            parts.append(f"Cambio en corridas: {n_runs:+.0f}")

        if extra_context:
            for key, value in extra_context.items():
                parts.append(f"{key}: {value}")

        return ". ".join(parts)

    def _extract_warnings(self, comp: Any) -> List[str]:
        warnings = []
        delta = comp.delta or {}

        g_eff = delta.get('G_efficiency')
        if isinstance(g_eff, (int, float)) and g_eff < 0:
            warnings.append(
                "La eficiencia de prediccion global (G-efficiency) empeora; validar impacto operativo"
            )

        n_runs = delta.get('n_runs')
        if isinstance(n_runs, (int, float)) and n_runs > 5:
            warnings.append(
                "El incremento de corridas es considerable; revisar restriccion de presupuesto y calendario"
            )

        return warnings

    def _extract_recommendations(self, comp: Any) -> List[str]:
        recs = []

        if comp.worth_it:
            recs.append(
                "Adoptar el diseno B y ejecutar una validacion posterior para confirmar las mejoras"
            )
        else:
            recs.append(
                "Mantener el diseno actual y explorar alternativas con mejor relacion beneficio-costo"
            )

        recs.append(
            "Documentar los deltas de eficiencia y su impacto en precision antes de decidir la siguiente wave"
        )

        return recs

    def _assess_confidence(self, comp: Any) -> str:
        delta = comp.delta or {}
        improvements = 0

        for key in ('D_efficiency', 'A_efficiency', 'mean_power'):
            val = delta.get(key)
            if isinstance(val, (int, float)) and val > 0:
                improvements += 1

        if improvements >= 2:
            return "Alta - multiples metricas mejoran en la misma direccion"
        if improvements == 1:
            return "Moderada - mejora parcial con posibles compensaciones"
        return "Baja - senal de mejora debil o contradictoria"


# Registro automático de interpretadores
from semantic.core import register_interpreter

# Registrar interpretadores en el registry global
register_interpreter("Recommendation", RecommendationInterpreter)
register_interpreter("DesignEvaluation", EvaluationInterpreter)
register_interpreter("FitResult", FitInterpreter)
register_interpreter("NextRunsProposal", ProposalInterpreter)
register_interpreter("DesignComparison", ComparisonInterpreter)


__all__ = [
    "RecommendationInterpreter",
    "EvaluationInterpreter",
    "FitInterpreter",
    "ProposalInterpreter",
    "ComparisonInterpreter"
]
