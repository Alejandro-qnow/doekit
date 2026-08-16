"""
Ejemplos Prácticos de doekit para Agentes de Investigación
===========================================================

Ejemplos ejecutables que demuestran casos de uso clave.
Ejecutar en el ambiente virtual creado.
"""

import doekit as ed
import numpy as np
import pandas as pd

print("=" * 70)
print("EJEMPLOS PRÁCTICOS DE DOEKIT PARA AGENTES DE INVESTIGACIÓN")
print("=" * 70)

# ============================================================================
# EJEMPLO 1: Recomendación Automática de Diseño
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 1: Recomendación Automática de Diseño")
print("=" * 70)

# Escenario: Optimizar un proceso con 3 factores, presupuesto de 20 corridas
rec = ed.recommend_design(
    goal="optimization",
    factors={
        "Temperature": (50, 100),
        "Pressure": (1, 5),
        "pH": (3, 9)
    },
    budget=20,
    model_order="quadratic",
    priorities={"runs": 1.0, "precision": 2.0, "prediction": 1.5}
)

print("\n" + rec.summary())

# Acceder al diseño ganador
design = rec.design
print(f"\nDiseño seleccionado: {rec.method}")
print(f"Número de corridas: {design.n_runs}")
print(f"\nPrimeras 5 corridas:")
print(design.matrix.head())

# ============================================================================
# EJEMPLO 2: Evaluación de Calidad del Diseño
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 2: Evaluación de Calidad del Diseño")
print("=" * 70)

model = ed.Model.full_quadratic(["Temperature", "Pressure", "pH"])
eval_result = ed.evaluate(design, model=model)

print("\n" + eval_result.summary())

# Métricas individuales
effs = ed.efficiencies(design, model=model)
print(f"\nMétricas de Eficiencia:")
print(f"  D-efficiency: {effs['D_efficiency']:.1f}%")
print(f"  A-efficiency: {effs['A_efficiency']:.1f}%")
print(f"  G-efficiency: {effs['G_efficiency']:.1f}%")
print(f"  SPV mean: {effs['spv_mean']:.4f}")
print(f"  Rank deficient: {effs['rank_deficient']}")

# VIF (Variance Inflation Factor)
vif_values = ed.vif(design, model=model)
print(f"\nVariance Inflation Factors (VIF < 10 es bueno):")
print(vif_values)

# ============================================================================
# EJEMPLO 3: Análisis de Resultados Experimentales
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 3: Análisis de Resultados Experimentales")
print("=" * 70)

# Simular respuestas experimentales
np.random.seed(42)
# Modelo verdadero: y = 50 + 2*Temp + 3*Press - 1.5*pH + noise
true_response = (
    50
    + 2.0 * design.matrix["Temperature"]
    + 3.0 * design.matrix["Pressure"]
    - 1.5 * design.matrix["pH"]
    + np.random.randn(design.n_runs) * 2  # Ruido
)

# Ajustar modelo
fit = ed.fit_linear_model(design, true_response, model=model)

print(f"\nCalidad del ajuste:")
print(f"  R² = {fit.r_squared:.4f}")
print(f"  Adjusted R² = {fit.r_squared_adj:.4f}")
print(f"  Residual σ² = {fit.sigma2:.4f}")
print(f"  DOF = {fit.dof}")

print(f"\nCoeficientes estimados (top 10 por significancia):")
results_df = fit.summary_frame().sort_values("p_value").head(10)
print(results_df[["term", "estimate", "std_error", "t_value", "p_value"]])

# Main effects
effects = ed.main_effects(design, true_response)
print(f"\nEfectos Principales:")
print(effects.sort_values(ascending=False))

# ANOVA table
anova = ed.anova_table(fit)
print(f"\nTabla ANOVA:")
print(anova)

# ============================================================================
# EJEMPLO 4: Experimentación Secuencial/Adaptativa
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 4: Experimentación Secuencial/Adaptativa")
print("=" * 70)

# Diseño inicial más pequeño
initial_design = ed.recommend_design(
    goal="screening",
    factors=4,
    budget=12,
    model_order="linear"
).design

print(f"Diseño inicial: {initial_design.n_runs} corridas")
print(initial_design.matrix)

# Simular datos iniciales
np.random.seed(123)
y_initial = np.random.randn(initial_design.n_runs) * 5 + 50

# Proponer siguiente lote
proposal = ed.propose_next_runs(
    initial_design,
    response=y_initial,
    n_add=4,
    criterion="D",
    active_p=0.05
)

print("\n" + proposal.summary())

print(f"\nTerminos activos detectados: {proposal.active_terms}")
print(f"Sigma estimado: {proposal.sigma_hat:.4f}")

# Comparación
comp = proposal.comparison
print(f"\n¿Vale la pena agregar corridas? {comp.worth_it}")
print(f"Resumen: {comp.summary}")

print(f"\nNuevas corridas propuestas:")
print(proposal.added.matrix)

# ============================================================================
# EJEMPLO 5: Workflow Completo con Clase Experiment
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 5: Workflow Completo con Clase Experiment")
print("=" * 70)

# Crear experimento
exp = ed.experiment(
    goal="optimization",
    factors={
        "X1": (-1, 1),
        "X2": (-1, 1),
        "X3": (-1, 1)
    },
    budget=15,
    responses=["yield", "purity", "cost"]
)

print(f"Experimento creado:")
print(f"  Diseño: {exp.design.metadata.get('kind', 'Custom')}")
print(f"  Corridas: {exp.design.n_runs}")

# Evaluar calidad
eval_exp = exp.evaluate()
print(f"\nCalidad del diseño:")
print(f"  D-eff: {eval_exp.metrics['D_efficiency']:.1f}%")
print(f"  G-eff: {eval_exp.metrics['G_efficiency']:.1f}%")

# Plantilla para colección
plan = exp.plan
print(f"\nPlantilla de colección (primeras 3 filas):")
print(plan.head(3))

# Simular datos multi-respuesta
np.random.seed(456)
results_df = pd.DataFrame({
    "yield": 70 + np.random.randn(exp.design.n_runs) * 5,
    "purity": 85 + np.random.randn(exp.design.n_runs) * 3,
    "cost": 100 + np.random.randn(exp.design.n_runs) * 10
})

# Ingerir datos
exp.ingest(results_df, fit=True)

# Resumen multi-respuesta
summary = exp.multi_response_summary(
    goals={"yield": "max", "purity": "max", "cost": "min"}
)

print(f"\nResumen multi-respuesta:")
print(f"  {summary['note']}")
print(f"\nCalidad de ajuste por respuesta:")
for resp, metrics in summary["per_response"].items():
    print(f"  {resp}: R²={metrics['r_squared']:.3f}, σ={metrics['sigma']:.2f}")

# Desirabilidad
desirability = exp.desirability(
    goals={"yield": "max", "purity": "max", "cost": "min"}
)
best_idx = desirability.idxmax()
best_run = exp.design.matrix.iloc[best_idx]

print(f"\nMejor configuración (por desirabilidad):")
print(f"  Índice: {best_idx}")
print(f"  Desirabilidad: {desirability.iloc[best_idx]:.3f}")
print(f"  Configuración:")
for factor, value in best_run.items():
    print(f"    {factor}: {value:.3f}")
print(f"  Respuestas:")
print(f"    yield: {results_df.iloc[best_idx]['yield']:.2f}")
print(f"    purity: {results_df.iloc[best_idx]['purity']:.2f}")
print(f"    cost: {results_df.iloc[best_idx]['cost']:.2f}")

# Proponer siguiente lote
next_prop = exp.next(n_add=5, criterion="I")  # I-optimal para predicción
print(f"\nPropuesta de siguientes corridas:")
print(f"  Nuevas corridas: {next_prop.added.n_runs}")
print(f"  Vale la pena: {next_prop.comparison.worth_it}")

# ============================================================================
# EJEMPLO 6: Diseño de Screening con Identificación de Factores Activos
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 6: Screening y Identificación de Factores Activos")
print("=" * 70)

# Screening de 8 factores
screening_design = ed.plackett_burman(8)
print(f"Diseño Plackett-Burman para 8 factores: {screening_design.n_runs} corridas")

# Simular respuesta donde solo X1, X3, X5 son activos
np.random.seed(789)
X = screening_design.matrix
y_screen = (
    10
    + 5.0 * X["X1"]      # Activo
    + 0.5 * X["X2"]      # Ruido
    + 3.0 * X["X3"]      # Activo
    + 0.3 * X["X4"]      # Ruido
    + 4.0 * X["X5"]      # Activo
    + 0.2 * X["X6"]      # Ruido
    + 0.4 * X["X7"]      # Ruido
    + 0.1 * X["X8"]      # Ruido
    + np.random.randn(screening_design.n_runs) * 1.5
)

# Analizar
model_screen = ed.Model.main_effects(screening_design.factor_names)
fit_screen = ed.fit_linear_model(screening_design, y_screen, model=model_screen)

# Identificar activos
print(f"\nIdentificación de factores activos (p < 0.05):")
actives = []
for name, pval, coef in zip(fit_screen.names, fit_screen.pvalues, fit_screen.coefficients):
    if name != "(Intercept)" and pval < 0.05:
        effect = coef * 2  # Efecto total (de -1 a +1)
        actives.append((name, effect, pval))
        print(f"  {name}: efecto={effect:+.3f}, p={pval:.4f}")

print(f"\nFactores activos detectados: {len(actives)}/8")
print(f"R² del modelo: {fit_screen.r_squared:.3f}")

# ============================================================================
# EJEMPLO 7: Diseño de Mezclas (Mixture Design)
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 7: Diseño de Mezclas (Mixture Design)")
print("=" * 70)

# Mezcla de 3 componentes que deben sumar 100%
mix_factors = [
    ed.MixtureFactor("Component_A"),
    ed.MixtureFactor("Component_B"),
    ed.MixtureFactor("Component_C")
]

# Diseño simplex lattice grado 2
mix_design = ed.simplex_lattice(mix_factors, degree=2)
print(f"Diseño Simplex Lattice (grado 2): {mix_design.n_runs} corridas")
print(f"\nPuntos del diseño (suman 1.0):")
print(mix_design.matrix)
print(f"\nVerificación de suma:")
print(mix_design.matrix.sum(axis=1))

# Simular respuesta de mezcla
np.random.seed(321)
M = mix_design.matrix
y_mix = (
    50 * M["Component_A"]
    + 70 * M["Component_B"]
    + 40 * M["Component_C"]
    + 30 * M["Component_A"] * M["Component_B"]  # Interacción
    + np.random.randn(mix_design.n_runs) * 2
)

# Modelo de Scheffé para mezclas
model_mix = ed.Model.scheffe_quadratic(["Component_A", "Component_B", "Component_C"])
fit_mix = ed.fit_linear_model(mix_design, y_mix, model=model_mix)

print(f"\nModelo de Scheffé ajustado:")
print(f"  R² = {fit_mix.r_squared:.3f}")
print(f"\nCoeficientes:")
print(fit_mix.summary_frame()[["term", "estimate", "std_error", "p_value"]])

# ============================================================================
# EJEMPLO 8: Integración con Bayesian Optimization
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 8: Integración con Bayesian Optimization")
print("=" * 70)

from doekit.adapters import bo

# Definir espacio de búsqueda tipo BO
bounds = [
    ("learning_rate", 0.001, 0.1),
    ("batch_size", [16, 32, 64, 128]),
    ("dropout", 0.1, 0.5)
]

# Generar candidatos para augment_design
candidates = bo.candidates_from_bounds(bounds, n=300, seed=999)
print(f"Candidatos generados: {candidates.n_runs}")
print(f"\nPrimeros 5 candidatos:")
print(candidates.matrix.head())

# Usar con diseño secuencial
# (En un caso real, tendrías un diseño inicial y datos)
# initial_bo_design = ...
# y_bo = ...
# proposal_bo = ed.propose_next_runs(
#     initial_bo_design,
#     response=y_bo,
#     n_add=10,
#     candidates=candidates  # Usa estos candidatos
# )

# ============================================================================
# EJEMPLO 9: Comparación de Diseños
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 9: Comparación de Diseños")
print("=" * 70)

# Crear dos diseños para comparar
design_a = ed.plackett_burman(4)
design_b = ed.box_behnken({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1), "X4": (-1, 1)})

print(f"Diseño A (Plackett-Burman): {design_a.n_runs} corridas")
print(f"Diseño B (Box-Behnken): {design_b.n_runs} corridas")

# Usar modelo lineal para comparación justa
model_comp = ed.Model.main_effects(["X1", "X2", "X3", "X4"])

comparison = ed.compare_designs(
    a=design_a,
    b=design_b,
    model=model_comp,
    a_label="PB-12",
    b_label="BB-25"
)

print(f"\nComparación:")
print(f"  {comparison.summary}")
print(f"  ¿Vale la pena el diseño B? {comparison.worth_it}")

print(f"\nTabla de comparación:")
print(comparison.table)

# ============================================================================
# EJEMPLO 10: Serialización y Persistencia
# ============================================================================
print("\n" + "=" * 70)
print("EJEMPLO 10: Serialización y Persistencia")
print("=" * 70)

# Crear experimento simple
exp_to_save = ed.experiment(
    goal="screening",
    factors=5,
    budget=12,
    responses=["response1"]
)

# Simular datos
np.random.seed(111)
y_save = np.random.randn(exp_to_save.design.n_runs)
exp_to_save.ingest(y_save, fit=True)

# Serializar a dict
exp_dict = exp_to_save.to_dict()
print(f"Experimento serializado a dict con schema: {exp_dict['schema']}")
print(f"Claves: {list(exp_dict.keys())}")

# Guardar a JSON
import json
with open("experiment_saved.json", "w") as f:
    json.dump(exp_dict, f, indent=2)
print(f"\nExperimento guardado en 'experiment_saved.json'")

# Cargar de JSON
with open("experiment_saved.json", "r") as f:
    exp_dict_loaded = json.load(f)

exp_loaded = ed.Experiment.from_dict(exp_dict_loaded)
print(f"Experimento cargado exitosamente")
print(f"  R² del modelo: {exp_loaded.fit.r_squared:.3f}")
print(f"  Número de corridas: {exp_loaded.design.n_runs}")

# Exportar a Excel
exp_to_save.export_excel("experiment_template.xlsx")
print(f"\nPlantilla exportada a 'experiment_template.xlsx'")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RESUMEN DE EJEMPLOS")
print("=" * 70)
print("""
Hemos demostrado:

1. ✅ Recomendación automática de diseño experimental
2. ✅ Evaluación de calidad (D/A/G-efficiency, VIF)
3. ✅ Análisis de resultados (ANOVA, efectos, p-values)
4. ✅ Experimentación secuencial/adaptativa
5. ✅ Workflow completo con clase Experiment
6. ✅ Screening y detección de factores activos
7. ✅ Diseños de mezclas (mixture)
8. ✅ Integración con Bayesian Optimization
9. ✅ Comparación de diseños alternativos
10. ✅ Serialización y persistencia

Todos estos ejemplos son ejecutables en el ambiente virtual creado.
Los agentes de investigación pueden usar estos patrones para automatizar
la experimentación científica de forma rigurosa y eficiente.
""")

print("\n" + "=" * 70)
print("FIN DE LOS EJEMPLOS")
print("=" * 70)
