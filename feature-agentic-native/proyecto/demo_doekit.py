"""
Demo Ejecutable de doekit - Casos de Uso Clave para Agentes
===========================================================
"""

import doekit as ed
import numpy as np
import pandas as pd

np.random.seed(42)

print("=" * 80)
print("DEMO: doekit para Agentes de Investigación")
print("=" * 80)

# =============================================================================
# DEMO 1: Recomendación y Evaluación de Diseño
# =============================================================================
print("\n[DEMO 1] Recomendación Automática de Diseño Experimental")
print("-" * 80)

# Recomendar diseño para optimización
rec = ed.recommend_design(
    goal="optimization",
    factors={"Temp": (50, 100), "Press": (1, 5), "pH": (3, 9)},
    budget=20,
    model_order="quadratic"
)

print(f"\n✓ Diseño recomendado: {rec.method}")
print(f"✓ Corridas: {rec.design.n_runs}")
print(f"\nDiseño experimental:")
print(rec.design.matrix)

# Evaluar calidad
eval_result = ed.evaluate(rec.design)
print(f"\nCalidad del diseño:")
print(f"  D-efficiency: {eval_result.d_efficiency:.1f}%")
print(f"  G-efficiency: {eval_result.g_efficiency:.1f}%")
print(f"  DOF: {eval_result.dof}")

# =============================================================================
# DEMO 2: Experimentación Secuencial/Adaptativa
# =============================================================================
print("\n\n[DEMO 2] Experimentación Secuencial/Adaptativa")
print("-" * 80)

# Diseño inicial de screening
screen_design = ed.plackett_burman(5)
print(f"\n✓ Diseño inicial (Plackett-Burman): {screen_design.n_runs} corridas")

# Obtener nombres de factores
factor_names = list(screen_design.matrix.columns)
print(f"  Factores: {', '.join(factor_names)}")

# Simular resultados experimentales
X_screen = screen_design.matrix
y_screen = (
    50
    + 8 * X_screen[factor_names[0]]   # Efecto grande
    + 2 * X_screen[factor_names[1]]   # Efecto pequeño
    - 6 * X_screen[factor_names[3]]   # Efecto grande negativo
    + np.random.randn(screen_design.n_runs) * 3
)

# Analizar para identificar factores activos
model_screen = ed.Model.main_effects(screen_design.factor_names)
fit_screen = ed.fit_linear_model(screen_design, y_screen, model=model_screen)

print(f"\nAnálisis del screening:")
print(f"  R² = {fit_screen.r_squared:.3f}")

# Mostrar factores significativos
summary = fit_screen.summary_frame()
significant = summary[summary['p_value'] < 0.05]
print(f"\n✓ Factores significativos (p < 0.05):")
for _, row in significant.iterrows():
    if row['term'] != '(Intercept)':
        print(f"    {row['term']}: estimado={row['estimate']:+.2f}, p={row['p_value']:.4f}")

# Proponer siguiente lote de experimentos
proposal = ed.propose_next_runs(
    screen_design,
    response=y_screen,
    n_add=6,
    criterion="D"
)

print(f"\n✓ Propuesta de siguientes corridas:")
print(f"  Corridas a agregar: {proposal.added.n_runs}")
print(f"  ¿Vale la pena? {proposal.comparison.worth_it}")
print(f"  Razón: {proposal.comparison.summary}")

# =============================================================================
# DEMO 3: Workflow Completo con Clase Experiment
# =============================================================================
print("\n\n[DEMO 3] Workflow Completo con Clase Experiment")
print("-" * 80)

# Crear experimento multi-respuesta
exp = ed.experiment(
    goal="optimization",
    factors={"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)},
    budget=15,
    responses=["yield", "quality", "cost"]
)

print(f"\n✓ Experimento creado:")
print(f"  Método: Box-Behnken o similar")
print(f"  Corridas: {exp.design.n_runs}")

# Simular datos experimentales multi-respuesta
np.random.seed(123)
X = exp.design.matrix
results = pd.DataFrame({
    "yield": 70 + 10*X["X1"] + 5*X["X2"] + np.random.randn(exp.design.n_runs)*2,
    "quality": 85 + 3*X["X1"] - 4*X["X3"] + np.random.randn(exp.design.n_runs)*1.5,
    "cost": 100 - 8*X["X2"] + 5*X["X3"] + np.random.randn(exp.design.n_runs)*3
})

# Ingestar datos
exp.ingest(results, fit=True)

print(f"\n✓ Datos ingeridos y modelos ajustados:")
for response_name in ["yield", "quality", "cost"]:
    fit = exp.fits[response_name]
    print(f"  {response_name}: R²={fit.r_squared:.3f}, σ²={fit.sigma2:.2f}")

# Calcular desirabilidad (optimización multi-objetivo)
desirability = exp.desirability(
    goals={"yield": "max", "quality": "max", "cost": "min"}
)

best_idx = desirability.idxmax()
print(f"\n✓ Mejor configuración por desirabilidad:")
print(f"  Índice: {best_idx}")
print(f"  Desirabilidad: {desirability.iloc[best_idx]:.3f}")
print(f"  Configuración:")
for col in exp.design.matrix.columns:
    print(f"    {col}: {exp.design.matrix.iloc[best_idx][col]:.3f}")
print(f"  Respuestas esperadas:")
print(f"    yield: {results.iloc[best_idx]['yield']:.1f}")
print(f"    quality: {results.iloc[best_idx]['quality']:.1f}")
print(f"    cost: {results.iloc[best_idx]['cost']:.1f}")

# =============================================================================
# DEMO 4: Serialización y Persistencia
# =============================================================================
print("\n\n[DEMO 4] Serialización y Persistencia")
print("-" * 80)

# Serializar experimento a JSON
exp_dict = exp.to_dict()
print(f"\n✓ Experimento serializado")
print(f"  Schema: {exp_dict['schema']}")
print(f"  Claves: {', '.join(exp_dict.keys())}")

# Guardar a archivo
import json
with open("experiment.json", "w") as f:
    json.dump(exp_dict, f, indent=2)
print(f"  Guardado en: experiment.json")

# Cargar de archivo
exp_loaded = ed.Experiment.from_dict(exp_dict)
print(f"\n✓ Experimento recargado exitosamente")
print(f"  R² (primera respuesta): {exp_loaded.fit.r_squared:.3f}")

# Exportar a Excel (requiere openpyxl)
try:
    exp.export_excel("experiment_plan.xlsx")
    print(f"  Template exportado a: experiment_plan.xlsx")
except ImportError:
    print(f"  (Excel export requiere: pip install 'doekit[export]')")

# =============================================================================
# DEMO 5: Integración con Bayesian Optimization
# =============================================================================
print("\n\n[DEMO 5] Integración con Bayesian Optimization")
print("-" * 80)

from doekit.adapters import bo

# Definir espacio de búsqueda tipo BO
bounds = [
    ("lr", 0.001, 0.1),
    ("depth", [2, 3, 4, 5, 6]),
    ("dropout", 0.1, 0.5)
]

# Generar candidatos usando doekit
candidates = bo.candidates_from_bounds(bounds, n=200)
print(f"\n✓ Candidatos BO generados: {candidates.n_runs} puntos")
print(f"\nPrimeros 5 candidatos:")
print(candidates.matrix.head())

print(f"\n✓ Estos candidatos pueden usarse con propose_next_runs() para")
print(f"  combinar Bayesian Optimization con métricas de DoE clásico")

# =============================================================================
# RESUMEN
# =============================================================================
print("\n\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)
print("""
✓ Recomendación automática de diseños experimentales
✓ Evaluación rigurosa de calidad (D/A/G-efficiency, VIF)
✓ Experimentación secuencial/adaptativa con propose_next_runs()
✓ Workflow completo con clase Experiment
✓ Optimización multi-objetivo con desirabilidad
✓ Serialización JSON y exportación Excel
✓ Integración con Bayesian Optimization

doekit permite a agentes de investigación:
• Diseñar experimentos óptimos automáticamente
• Cerrar el bucle experimental de forma autónoma
• Minimizar número de experimentos (eficiencia)
• Tomar decisiones basadas en métricas estadísticas rigurosas
• Manejar múltiples respuestas y restricciones complejas

Archivos generados:
- experiment.json (estado serializado)
- experiment_plan.xlsx (plantilla de colección)
""")

print("\n" + "=" * 80)
print("FIN DE LA DEMO")
print("=" * 80)
