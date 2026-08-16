# Análisis de doekit: Design of Experiments para Investigadores con Agentes

## Instalación Completada
✅ **doekit v0.7.3** instalado exitosamente en ambiente virtual Python 3.13.7

### Dependencias principales:
- numpy >= 1.23
- pandas >= 1.5
- scipy >= 1.9
- statsmodels >= 0.14
- asttokens >= 3.0.2

---

## 1. Visión General de doekit

**doekit** es una librería completa de **Design of Experiments (DoE)** en Python que proporciona herramientas para:

- 🔬 **Diseño experimental**: Creación de diseños óptimos para experimentos científicos
- 📊 **Análisis estadístico**: Evaluación rigurosa de resultados experimentales
- 🔄 **Experimentación secuencial/adaptativa**: Iteración inteligente basada en resultados previos
- 🎯 **Optimización**: Búsqueda de condiciones óptimas usando criterios de optimalidad
- 📈 **Visualización y reportes**: Generación automática de reportes HTML

---

## 2. Arquitectura del Código

### 2.1 Estructura Modular

```
doekit/
├── domain/              # Lógica de dominio central
│   ├── factors.py       # Factores: Continuo, Discreto, Categórico, Mixture
│   ├── model.py         # Modelos: Lineales, Interacciones, Cuadráticos
│   ├── design.py        # Clase Design (contenedor principal)
│   ├── constraints.py   # Restricciones experimentales
│   ├── criteria/        # Criterios de optimalidad (D, A, I, G, E, T)
│   └── region.py        # Regiones factoriales (Hipercubo, Simplex)
│
├── generation/          # Generación de diseños
│   ├── catalog/         # Diseños clásicos pre-definidos
│   │   ├── factorial.py        # Full factorial, fraccional
│   │   ├── screening.py        # Plackett-Burman
│   │   ├── response_surface.py # Box-Behnken, Central Composite
│   │   ├── definitive.py       # Definitive Screening Design
│   │   ├── mixture.py          # Simplex lattice, centroid
│   │   ├── split_plot.py       # Split-plot designs
│   │   └── random_design.py    # Latin Hypercube, Random
│   └── search/          # Búsqueda de diseños óptimos
│       └── optimal.py   # D-optimal, I-optimal (Fedorov, KL-exchange)
│
├── assessment/          # Evaluación y análisis
│   ├── evaluation/      # Métricas de calidad
│   │   └── metrics.py   # D/A/G-efficiency, VIF, alias matrix, FDS
│   └── analysis/        # Análisis estadístico
│       ├── ols.py       # Regresión OLS
│       ├── mixed.py     # Modelos mixtos (split-plot)
│       ├── anova.py     # Tablas ANOVA
│       ├── effects.py   # Cálculo de efectos principales
│       └── lof.py       # Lack of fit tests
│
├── orchestration/       # Orquestación de experimentos ⭐
│   ├── advise/          # Sistema de recomendación
│   │   ├── recommend.py # recommend_design() - advisor automático
│   │   ├── rules.py     # Reglas de selección de diseño
│   │   └── rank.py      # Ranking de alternativas
│   ├── sequential/      # DoE adaptativo ⭐⭐⭐
│   │   └── propose.py   # propose_next_runs(), augment_design()
│   └── experiment/      # Agregado de alto nivel
│       └── aggregate.py # Clase Experiment - workflow completo
│
├── adapters/            # Integraciones externas
│   └── bo.py            # Puente a Bayesian Optimization
│
├── presentation/        # Capa de presentación
│   ├── report.py        # Generación de reportes HTML
│   ├── export.py        # Exportación CSV/Excel
│   ├── workspace.py     # Proyectos multi-wave
│   └── render/          # Visualizaciones matplotlib
│
└── shared/              # Utilidades compartidas
    ├── serialize.py     # Serialización JSON
    └── errors.py        # Excepciones personalizadas
```

---

## 3. Módulos Clave para Investigadores con Agentes

### 3.1 🤖 **orchestration/advise** - Sistema de Recomendación Inteligente

**Archivo**: `orchestration/advise/recommend.py`

#### Funcionalidad Principal: `recommend_design()`

Este es un **asesor transparente** (no una "caja negra AutoML") que recomienda el mejor método de diseño experimental para un caso dado.

**Lógica de funcionamiento**:
1. **Reglas** → Genera una lista corta de diseños plausibles según:
   - Objetivo (screening vs optimization)
   - Tipos de factores (continuo, discreto, categórico, mixture)
   - Factores difíciles de cambiar (hard-to-change)
   - Orden del modelo (lineal, interacciones, cuadrático)

2. **Evaluación** → Rankea la lista corta usando métricas doekit:
   - D-efficiency (precisión de estimación)
   - A-efficiency (varianza promedio)
   - G-efficiency (predicción en el peor caso)
   - SPV mean (varianza de predicción escalada)
   - Número de corridas (presupuesto)

**Ejemplo de uso**:
```python
import doekit as ed

# Recomendar diseño para screening de 6 factores con presupuesto de 12 corridas
rec = ed.recommend_design(
    goal="screening",           # "screening" o "optimization"
    factors=6,                   # Número de factores o lista detallada
    budget=12,                   # Máximo de corridas experimentales
    model_order="linear",        # "linear", "interactions", "quadratic"
    priorities={"runs": 1.0, "precision": 1.0, "prediction": 1.0}
)

print(rec.summary())            # Rationale + tabla de alternativas
design = rec.design             # Diseño ganador
```

**Salida típica**:
```
Recommendation: Plackett-Burman
----------------------------------------------
To identify influential factors with 6 factors with a budget of 12 runs 
and a 'linear' model, the best compromise under your priorities is 
Plackett-Burman (12 runs, D-efficiency 100.0%, G-efficiency 100.0%).

Evaluated alternatives:
            method  runs  D_eff  G_eff  SPV_mean  in_budget  supports_model
  Plackett-Burman    12  100.0  100.0      0.58       True            True
Definitive Screen    13   95.3   88.2      0.62      False            True
   Full factorial    64  100.0  100.0      0.11      False            True

Caveats:
  - Recommendation is conditional on model_order='linear'...
  - "Best" is a multi-objective trade-off (runs vs precision vs prediction)...
  - After the first wave, use propose_next_runs / augment_design...
```

**✨ Utilidad para Agentes de Investigación**:
- **Automatización de decisiones**: Un agente puede usar esto para seleccionar automáticamente el mejor diseño sin conocimiento experto en DoE
- **Explicabilidad**: Proporciona rationale y tabla de alternativas → decisiones transparentes
- **Flexibilidad**: Admite restricciones complejas (mixture, split-plot, irregular)

---

### 3.2 🔄 **orchestration/sequential** - Experimentación Adaptativa (★★★ MUY IMPORTANTE)

**Archivo**: `orchestration/sequential/propose.py`

Este módulo implementa **DoE secuencial/adaptativo**, fundamental para agentes que necesitan iterar experimentos.

#### 3.2.1 `propose_next_runs()` - Proponer siguientes corridas

**Caso de uso**: Tienes un diseño inicial, ya recolectaste datos, y necesitas decidir qué experimentos hacer a continuación.

**Flujo**:
1. Diseño inicial ejecutado → datos de respuesta (`y`)
2. Ajusta modelo lineal → identifica términos activos (p < 0.05)
3. Estima σ residual empírico
4. Augmenta diseño usando criterio D/I-optimal
5. Compara diseño actual vs augmentado (eficiencias, power)
6. Retorna propuesta + comparación

**Código**:
```python
import doekit as ed
import numpy as np

# Diseño inicial
design = ed.plackett_burman(6)

# Simular respuestas experimentales
y = np.random.randn(design.n_runs)

# Proponer siguiente lote de 4 corridas
proposal = ed.propose_next_runs(
    design, 
    response=y,           # Datos experimentales
    n_add=4,              # Número de corridas a añadir
    criterion="D",        # "D", "I", "A", "G", "E", "T"
    active_p=0.05         # Umbral p-value para términos activos
)

print(proposal.summary())
# Muestra:
# - Rationale (por qué estos runs)
# - Comparación de eficiencias (current vs augmented)
# - Términos activos detectados
# - Sigma estimado
# - Matriz de nuevas corridas

# Acceder a componentes
new_runs = proposal.added.matrix        # Solo las nuevas corridas
full_design = proposal.combined.matrix  # Diseño completo (viejo + nuevo)
comparison = proposal.comparison        # DesignComparison object
```

**Salida típica**:
```
Next runs proposal (4 new, criterion=D)
------------------------------------------------
Propose 4 new run(s) by D-optimal augmentation of the current 12-run design 
(sigma_hat=0.9234 from residual df=5). Yes: 4 extra run(s) 
(ΔD=+12.3 pts, ΔG=+8.7 pts, ΔSPV_mean=-0.082, Δpower=+0.15).

Proposed runs:
   X1   X2   X3   X4   X5   X6
0  +1   -1   +1   -1   +1   -1
1  -1   +1   -1   +1   -1   +1
2   0    0   +1   +1   -1   -1
3  +1   +1   -1   -1   +1   +1

Caveats:
  - Active terms at p<0.05: X1, X3, X1:X3. Consider focusing the next wave...
```

#### 3.2.2 `augment_design()` - Augmentar diseño sin respuesta

Si NO tienes datos de respuesta aún, pero quieres extender un diseño usando criterios de información.

```python
# Augmentar basándose solo en geometría/información
augmented = ed.augment_design(
    design,
    n_add=4,
    criterion="D",         # Maximizar determinante de (X'X)
    n_starts=5             # Número de inicializaciones greedy
)
```

#### 3.2.3 `compare_designs()` - Comparar dos diseños

```python
comparison = ed.compare_designs(
    a=design_current,
    b=design_augmented,
    model=model,
    effect_size=1.0,
    sigma=1.0,
    a_label="current",
    b_label="proposed"
)

print(comparison.summary)
# "Yes: 4 extra run(s) (ΔD=+12.3 pts, ΔG=+8.7 pts, ...)"

print(comparison.worth_it)  # True/False/None
print(comparison.table)      # DataFrame con métricas
```

**✨ Utilidad para Agentes de Investigación**:
- **Cierre del bucle experimental**: `evaluate → run → analyze → propose_next` completamente automatizable
- **Aprendizaje activo**: El agente puede decidir dónde experimentar a continuación basándose en información ganada
- **Detección automática de patrones**: Identifica términos activos sin intervención humana
- **Estimación empírica**: Usa σ real de los datos, no supuestos previos
- **Criterio de decisión**: `worth_it` heuristic indica si vale la pena gastar recursos en más corridas

---

### 3.3 🎯 **orchestration/experiment** - Workflow End-to-End

**Archivo**: `orchestration/experiment/aggregate.py`

#### Clase `Experiment` - Agregado Stateful

Encapsula el **ciclo completo** de experimentación en un objeto stateful.

**Flujo típico**:
```python
import doekit as ed

# 1. Crear experimento desde recomendación
exp = ed.experiment(
    goal="optimization",
    factors={"X1": (0, 100), "X2": (50, 150), "X3": (10, 30)},
    budget=20,
    responses=["yield", "purity"]  # Multi-respuesta
)

# 2. Evaluar calidad del diseño
eval_result = exp.evaluate()
print(eval_result.summary())  # D-eff, G-eff, VIF, alias, power

# 3. Exportar plantilla para colección de datos
exp.export_excel("lab_runs.xlsx")  # Plantilla con columnas de factores + respuestas

# 4. Ingerir datos experimentales
import pandas as pd
data = pd.read_excel("lab_results.xlsx")
exp.ingest(data, fit=True)  # Ajusta modelos automáticamente

# 5. Ver ajuste multi-respuesta
summary = exp.multi_response_summary(goals={"yield": "max", "purity": "max"})
print(summary["note"])
# "Stronger fit on 'yield' (R²=0.923); weaker on 'purity' (R²=0.681)."

# 6. Calcular desirabilidad (multi-objetivo)
desirability = exp.desirability(goals={"yield": "max", "purity": "max"})
best_run = desirability.idxmax()  # Índice de la mejor corrida

# 7. Proponer siguiente lote
next_proposal = exp.next(n_add=5, criterion="I")  # I-optimal para predicción

# 8. Comparar si vale la pena
comp = exp.compare(n_add=5)
if comp.worth_it:
    print(f"Sí, vale la pena: {comp.summary}")
    new_plan = next_proposal.added.matrix
    # Ejecutar nuevos experimentos...

# 9. Generar reporte HTML
exp.report(output_dir="reports/", write_html=True)

# 10. Guardar estado completo
exp.save("project/wave1/")
```

**Métodos clave**:
- `evaluate()` → Calcula eficiencias, VIF, alias, power
- `ingest(response, fit=True)` → Carga datos y ajusta modelos
- `multi_response_summary()` → Analiza múltiples respuestas simultáneamente
- `desirability(goals)` → Función de desirabilidad multi-objetivo (Derringer-Suich)
- `next(n_add)` → Propone siguiente lote usando `propose_next_runs()`
- `compare(n_add)` → Evalúa si vale la pena agregar corridas
- `report()` → Genera reporte HTML completo
- `save()` / `load()` → Persistencia de estado

**Serialización**:
```python
# Serializar a JSON
exp_dict = exp.to_dict()  # schema: doekit.Experiment/1
import json
with open("exp.json", "w") as f:
    json.dump(exp_dict, f, indent=2)

# Reconstruir
exp_loaded = ed.Experiment.from_dict(exp_dict)
```

**✨ Utilidad para Agentes de Investigación**:
- **Estado persistente**: El agente puede guardar/cargar experimentos entre sesiones
- **Workflow automatizable**: Toda la lógica de experimentación en un objeto
- **Multi-respuesta nativo**: Optimización multi-objetivo sin código extra
- **Trazabilidad**: Metadata completo de recomendaciones, evaluaciones, análisis
- **Interoperabilidad**: Exporta/importa JSON, Excel, CSV

---

### 3.4 🔌 **adapters/bo** - Puente a Bayesian Optimization

**Archivo**: `adapters/bo.py`

Convierte espacios de búsqueda de BO (bounds, skopt spaces) en conjuntos de candidatos de doekit.

**¿Por qué importa?**

Combina:
- **BO** (black-box sequential optimization) 
- **DoE clásico** (métricas de calidad transparentes: D-eff, G-eff, power)

```python
from doekit.adapters import bo

# Definir bounds como BO
bounds = [
    ("learning_rate", 0.001, 0.1),   # Continuo
    ("n_layers", [2, 3, 4, 5]),       # Discreto
    ("activation", ["relu", "tanh", "sigmoid"])  # Categórico
]

# Generar candidatos para augment_design
candidates = bo.candidates_from_bounds(
    bounds, 
    n=500,           # Número de candidatos
    model=my_model,  # Opcional
    seed=42
)

# Usar con propose_next_runs
proposal = ed.propose_next_runs(
    current_design,
    response=y,
    n_add=4,
    candidates=candidates  # Usa estos candidatos en lugar de grid/random
)
```

**Integración con scikit-optimize**:
```python
from skopt.space import Real, Integer, Categorical
from doekit.adapters import bo

space = [
    Real(0.001, 0.1, name="lr"),
    Integer(2, 10, name="depth"),
    Categorical(["adam", "sgd"], name="optimizer")
]

candidates = bo.candidates_from_skopt_space(space, n=500)
```

**✨ Utilidad para Agentes de Investigación**:
- **Lenguaje común**: BO y DoE comparten métricas (efficiencies, FDS, power)
- **Hibridación**: Combinar búsqueda BO con evaluación de calidad estadística
- **Flexibilidad**: El agente puede usar BO cuando sea apropiado, DoE cuando no

---

### 3.5 📊 **assessment/evaluation** - Evaluación de Calidad

**Métricas clave para evaluar diseños**:

```python
import doekit as ed

design = ed.box_behnken({"X1": (-1, 1), "X2": (-1, 1), "X3": (-1, 1)})
model = ed.Model.full_quadratic(["X1", "X2", "X3"])

eval_result = ed.evaluate(design, model=model)
print(eval_result.summary())
```

**Métricas disponibles**:
- **D-efficiency**: Precisión de estimación de parámetros (0-100%)
- **A-efficiency**: Varianza promedio de coeficientes
- **G-efficiency**: Peor caso de predicción en la región
- **I-efficiency**: Varianza integrada de predicción
- **E-efficiency**: Mínimo eigenvalue de (X'X)
- **VIF** (Variance Inflation Factor): Multicolinealidad
- **Alias Matrix**: Confusión de efectos
- **FDS** (Fraction of Design Space): Gráfico de varianza de predicción
- **Power Analysis**: Probabilidad de detectar efectos

```python
# Eficiencias individuales
effs = ed.efficiencies(design, model=model)
print(f"D-eff: {effs['D_efficiency']:.1f}%")
print(f"G-eff: {effs['G_efficiency']:.1f}%")

# VIF (> 10 indica multicolinealidad problemática)
vif_values = ed.vif(design, model=model)

# Power analysis
power = ed.power_analysis(design, model=model, effect_size=1.0, sigma=1.0)
print(power)  # DataFrame con power para cada término

# FDS data (para graficar)
fds = ed.fds_data(design, model=model)
# fds["relative_spv"] = varianza de predicción escalada por región
```

**✨ Utilidad para Agentes de Investigación**:
- **Decisiones cuantitativas**: El agente puede comparar diseños objetivamente
- **Diagnóstico automático**: Detectar problemas (aliasing, multicolinealidad)
- **Optimización guiada**: Seleccionar criterio apropiado (D para estimar, I para predecir)

---

### 3.6 📈 **assessment/analysis** - Análisis Estadístico

**Análisis de resultados experimentales**:

```python
import doekit as ed
import numpy as np

design = ed.central_composite({"X1": (-1, 1), "X2": (-1, 1)})
y = np.random.randn(design.n_runs)  # Respuestas simuladas
model = ed.Model.full_quadratic(["X1", "X2"])

# Ajustar modelo lineal
fit = ed.fit_linear_model(design, y, model=model)

print(f"R² = {fit.r_squared:.3f}")
print(f"Adjusted R² = {fit.r_squared_adj:.3f}")
print(f"Residual σ² = {fit.sigma2:.4f}")
print(f"DOF = {fit.dof}")

# Coeficientes y significancia
results_df = fit.results_frame
print(results_df[["coef", "std_err", "t_value", "p_value"]])

# ANOVA table
anova = ed.anova_table(fit)
print(anova)

# Main effects
effects = ed.main_effects(design, y)
print(effects)  # DataFrame con efectos por factor

# Half-normal plot data
hn_data = ed.half_normal_data(design, y)
# Para identificar efectos significativos gráficamente

# Lack of fit test (si hay réplicas)
lof = ed.lack_of_fit(design, y, model=model)
print(f"LOF p-value: {lof['p_value']:.4f}")
```

**Modelos mixtos (split-plot)**:
```python
# Para diseños split-plot con whole-plot factors
spd = ed.split_plot_design(
    whole_plot={"Temperature": [100, 150]},
    subplot={"Pressure": [1, 2, 3]}
)

fit_mixed = ed.fit_mixed_model(
    spd, 
    y, 
    groups="whole_plot_id"  # Columna de agrupación
)
# Modela correctamente la estructura jerárquica
```

**✨ Utilidad para Agentes de Investigación**:
- **Interpretación automática**: El agente puede extraer insights de resultados
- **Detección de efectos**: Identificar qué factores son importantes
- **Validación de modelos**: Lack of fit, R², residual plots
- **Soporte para diseños complejos**: Split-plot, bloques, réplicas

---

## 4. Casos de Uso para Agentes de Investigación

### 4.1 🤖 Agente de Optimización Experimental Autónomo

**Escenario**: Optimizar un proceso químico con 5 factores desconocidos.

**Workflow con doekit**:

```python
import doekit as ed
import numpy as np

class ExperimentalOptimizationAgent:
    def __init__(self, factors, budget_per_wave=10, max_waves=5):
        self.factors = factors
        self.budget_per_wave = budget_per_wave
        self.max_waves = max_waves
        self.wave = 0
        self.experiment = None
        self.history = []
        
    def initialize(self):
        """Wave 0: Screening inicial"""
        self.experiment = ed.experiment(
            goal="screening",
            factors=self.factors,
            budget=self.budget_per_wave,
            model_order="linear"
        )
        self.wave = 0
        
    def execute_wave(self, run_experiment_func):
        """Ejecutar una wave de experimentos"""
        if self.wave == 0:
            # Primera wave: usar diseño recomendado
            design_matrix = self.experiment.design.matrix
        else:
            # Waves siguientes: usar propuesta adaptativa
            proposal = self.experiment.next(
                n_add=self.budget_per_wave,
                criterion="I"  # I-optimal para predicción
            )
            if not proposal.comparison.worth_it:
                print("No vale la pena continuar. Convergido.")
                return False
            design_matrix = proposal.added.matrix
        
        # Ejecutar experimentos (función externa)
        results = run_experiment_func(design_matrix)
        
        # Ingerir resultados
        self.experiment.ingest(results, fit=True)
        
        # Guardar historial
        self.history.append({
            "wave": self.wave,
            "n_runs": len(design_matrix),
            "fit_quality": self.experiment.fit.r_squared if self.experiment.fit else None,
            "active_terms": self._get_active_terms()
        })
        
        self.wave += 1
        return True
        
    def _get_active_terms(self):
        """Extraer términos significativos"""
        if not self.experiment.fit:
            return []
        fit = self.experiment.fit
        active = []
        for name, pval in zip(fit.names, fit.pvalues):
            if name != "(Intercept)" and pval < 0.05:
                active.append(name)
        return active
    
    def optimize(self, run_experiment_func):
        """Bucle principal de optimización"""
        self.initialize()
        
        for wave in range(self.max_waves):
            print(f"\n=== Wave {wave} ===")
            continue_flag = self.execute_wave(run_experiment_func)
            
            if not continue_flag:
                break
                
            # Diagnóstico
            print(f"Active terms: {self.history[-1]['active_terms']}")
            print(f"R²: {self.history[-1]['fit_quality']:.3f}")
            
        # Encontrar mejor condición
        best_idx = np.argmax(self.experiment.response)
        best_conditions = self.experiment.design.matrix.iloc[best_idx]
        
        return {
            "best_conditions": best_conditions.to_dict(),
            "best_response": float(self.experiment.response[best_idx]),
            "total_runs": sum(h["n_runs"] for h in self.history),
            "history": self.history
        }

# Uso
agent = ExperimentalOptimizationAgent(
    factors={
        "Temperature": (50, 100),
        "Pressure": (1, 5),
        "pH": (3, 9),
        "Catalyst": (0.1, 1.0),
        "Time": (10, 60)
    },
    budget_per_wave=12
)

def my_experiment_runner(design_matrix):
    """Simula ejecución de experimentos"""
    # En la práctica, esto llamaría a un equipo de laboratorio, simulación, etc.
    return np.random.randn(len(design_matrix))

result = agent.optimize(my_experiment_runner)
print(result["best_conditions"])
```

---

### 4.2 🧪 Agente de Diseño Multi-Objetivo

**Escenario**: Optimizar formulación farmacéutica con múltiples respuestas (eficacia, estabilidad, costo).

```python
import doekit as ed
import pandas as pd

class MultiObjectiveFormulationAgent:
    def __init__(self, formulation_space, responses):
        self.space = formulation_space
        self.responses = responses
        
    def design(self, budget=20):
        """Diseñar experimento óptimo"""
        rec = ed.recommend_design(
            goal="optimization",
            factors=self.space,
            budget=budget,
            model_order="quadratic",
            priorities={"prediction": 2.0, "precision": 1.0}  # Enfatizar predicción
        )
        return rec.design
    
    def analyze(self, design, results_df):
        """Analizar resultados multi-respuesta"""
        exp = ed.Experiment.from_design(design, responses=self.responses)
        exp.ingest(results_df, fit=True)
        
        # Resumen multi-respuesta
        summary = exp.multi_response_summary(
            goals={"efficacy": "max", "stability": "max", "cost": "min"}
        )
        
        # Desirabilidad
        desirability = exp.desirability(
            goals={"efficacy": "max", "stability": "max", "cost": "min"}
        )
        
        # Mejor formulación
        best_idx = desirability.idxmax()
        best_formulation = design.matrix.iloc[best_idx].to_dict()
        
        return {
            "best_formulation": best_formulation,
            "desirability": float(desirability.iloc[best_idx]),
            "summary": summary,
            "all_results": results_df.assign(desirability=desirability)
        }

# Uso
agent = MultiObjectiveFormulationAgent(
    formulation_space={
        "API_content": (5, 20),      # % Active ingredient
        "Excipient_A": (10, 40),
        "Excipient_B": (5, 15),
        "pH": (5.5, 7.5)
    },
    responses=["efficacy", "stability", "cost"]
)

design = agent.design(budget=30)
# ... ejecutar experimentos ...
results = pd.DataFrame({
    "efficacy": [...],
    "stability": [...],
    "cost": [...]
})
analysis = agent.analyze(design, results)
```

---

### 4.3 🔬 Agente de Screening Rápido

**Escenario**: Identificar factores importantes entre muchos candidatos (high-throughput screening).

```python
import doekit as ed

class RapidScreeningAgent:
    def screen(self, n_factors, budget=None):
        """Screening eficiente de muchos factores"""
        # Auto-selección de presupuesto si no se especifica
        if budget is None:
            budget = min(n_factors + 4, n_factors * 2)
        
        # Recomendar diseño de screening
        rec = ed.recommend_design(
            goal="screening",
            factors=n_factors,
            budget=budget,
            model_order="linear"
        )
        
        print(f"Recommended: {rec.method}")
        print(f"Runs: {rec.design.n_runs}")
        
        return rec.design
    
    def identify_actives(self, design, response, alpha=0.05):
        """Identificar factores activos"""
        model = ed.Model.main_effects(design.factor_names)
        fit = ed.fit_linear_model(design, response, model=model)
        
        # Términos significativos
        actives = []
        for name, pval in zip(fit.names, fit.pvalues):
            if name != "(Intercept)" and pval < alpha:
                actives.append({
                    "factor": name,
                    "p_value": pval,
                    "coefficient": fit.coefficients[fit.names.index(name)],
                    "effect": fit.coefficients[fit.names.index(name)] * 2  # Efecto total
                })
        
        # Ordenar por magnitud de efecto
        actives.sort(key=lambda x: abs(x["effect"]), reverse=True)
        
        return {
            "active_factors": actives,
            "n_active": len(actives),
            "r_squared": fit.r_squared,
            "total_factors": len(design.factor_names)
        }

# Uso
agent = RapidScreeningAgent()

# Screening de 15 factores
design = agent.screen(n_factors=15, budget=20)

# ... recolectar datos ...
import numpy as np
y = np.random.randn(design.n_runs)

# Identificar factores importantes
results = agent.identify_actives(design, y)
print(f"Active factors: {results['n_active']}/{results['total_factors']}")
for factor in results["active_factors"]:
    print(f"  {factor['factor']}: effect={factor['effect']:.3f}, p={factor['p_value']:.4f}")
```

---

### 4.4 🎯 Agente con Restricciones Complejas

**Escenario**: Diseño de mezclas (mixture design) para formulaciones donde componentes suman 100%.

```python
import doekit as ed

class MixtureOptimizationAgent:
    def design_mixture(self, components, degree=2):
        """Diseño de mezcla"""
        # Componentes deben ser MixtureFactor
        factors = [ed.MixtureFactor(name) for name in components]
        
        rec = ed.recommend_design(
            goal="optimization",
            factors=factors,
            mixture=True,  # Activa lógica de mixture
            model_order="quadratic" if degree == 2 else "linear"
        )
        
        return rec.design
    
    def analyze_mixture(self, design, response):
        """Analizar con modelo de Scheffé"""
        # Modelo de Scheffé (apropiado para mezclas)
        model = ed.Model.scheffe_quadratic(design.factor_names)
        
        fit = ed.fit_linear_model(design, response, model=model)
        
        # ANOVA
        anova = ed.anova_table(fit)
        
        return {
            "model": model,
            "fit": fit,
            "anova": anova,
            "r_squared": fit.r_squared
        }

# Uso
agent = MixtureOptimizationAgent()

# Diseño para mezcla de 3 componentes
design = agent.design_mixture(["Component_A", "Component_B", "Component_C"])

# Los puntos suman 1.0 (100%)
print(design.matrix.sum(axis=1))  # Todos ~1.0

# Analizar
import numpy as np
y = np.random.randn(design.n_runs)
results = agent.analyze_mixture(design, y)
```

---

## 5. Integración con Ecosistema ML/AI

### 5.1 Integración con Optuna (Bayesian Optimization)

```python
import optuna
import doekit as ed

def doekit_suggest_trials(study, n_trials=5):
    """Generar trials usando doekit en lugar de sampler de Optuna"""
    # Obtener bounds de las variables
    trial_params = study.best_trials[0].params if study.best_trials else {}
    
    # Convertir a formato doekit
    from doekit.adapters import bo
    bounds = [
        ("param1", 0.001, 0.1),
        ("param2", 1, 100),
        # ...
    ]
    
    # Generar candidatos óptimos
    candidates = bo.candidates_from_bounds(bounds, n=200)
    
    # Seleccionar n_trials usando D-optimal
    design = ed.optimal_design(candidates, n_runs=n_trials, criterion="D")
    
    return design.matrix

# Integrar con Optuna
def objective(trial):
    # Obtener parámetros desde doekit design
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return x**2 + y**2

study = optuna.create_study()
# ... usar doekit_suggest_trials para inicialización ...
```

---

### 5.2 Integración con scikit-learn (Hyperparameter Tuning)

```python
import doekit as ed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def optimize_hyperparameters_doe(X_train, y_train):
    """Optimizar hiperparámetros con DoE en lugar de grid/random search"""
    
    # Definir espacio de búsqueda
    factors = {
        "n_estimators": (10, 200),
        "max_depth": (3, 20),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 10)
    }
    
    # Diseño inicial (screening)
    rec = ed.recommend_design(
        goal="screening",
        factors=factors,
        budget=20,
        model_order="interactions"
    )
    
    design = rec.design
    
    # Evaluar cada configuración
    scores = []
    for idx, row in design.matrix.iterrows():
        clf = RandomForestClassifier(
            n_estimators=int(row["n_estimators"]),
            max_depth=int(row["max_depth"]),
            min_samples_split=int(row["min_samples_split"]),
            min_samples_leaf=int(row["min_samples_leaf"]),
            random_state=42
        )
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        scores.append(score)
    
    # Analizar
    model = ed.Model.main_effects(list(factors.keys())) + \
            ed.Model.from_terms([ed.Interaction(("n_estimators", "max_depth"))])
    
    fit = ed.fit_linear_model(design, np.array(scores), model=model)
    
    # Identificar configuración óptima
    best_idx = np.argmax(scores)
    best_config = design.matrix.iloc[best_idx].to_dict()
    
    # Proponer siguiente lote (adaptativo)
    proposal = ed.propose_next_runs(design, response=np.array(scores), n_add=10)
    
    return {
        "best_config": best_config,
        "best_score": scores[best_idx],
        "fit": fit,
        "next_trials": proposal.added.matrix
    }
```

---

### 5.3 Integración con Ray Tune

```python
import doekit as ed
from ray import tune

def doekit_search_algorithm(config_space, n_initial=20, n_per_wave=5):
    """Custom search algorithm usando doekit sequential DoE"""
    
    class DoekitSearch(tune.search.SearchAlgorithm):
        def __init__(self, config_space):
            self.config_space = config_space
            self.design = None
            self.results = []
            self.wave = 0
            
        def suggest(self, trial_id):
            if self.design is None:
                # Primera wave: diseño óptimo
                rec = ed.recommend_design(
                    goal="optimization",
                    factors=self.config_space,
                    budget=n_initial
                )
                self.design = rec.design
                self.current_idx = 0
            
            if self.current_idx >= len(self.design.matrix):
                # Generar siguiente wave
                if len(self.results) > 0:
                    proposal = ed.propose_next_runs(
                        self.design,
                        response=np.array([r["score"] for r in self.results]),
                        n_add=n_per_wave
                    )
                    self.design = proposal.combined
                    self.current_idx = len(proposal.combined.matrix) - n_per_wave
                    self.wave += 1
            
            config = self.design.matrix.iloc[self.current_idx].to_dict()
            self.current_idx += 1
            return config
        
        def on_trial_complete(self, trial_id, result):
            self.results.append(result)
    
    return DoekitSearch(config_space)
```

---

## 6. Ventajas de doekit para Agentes de Investigación

### ✅ Automatización Total
- **recommend_design()**: Selección automática de método experimental
- **propose_next_runs()**: Cierre del bucle adaptativo sin intervención
- **Experiment class**: Workflow completo encapsulado

### ✅ Transparencia y Explicabilidad
- Rationales textuales de por qué se recomienda cada diseño
- Tablas de comparación de alternativas
- Métricas cuantitativas (D-eff, G-eff, power) en lugar de "cajas negras"

### ✅ Eficiencia de Datos
- Diseños óptimos minimizan número de experimentos
- Sequential DoE evita desperdiciar recursos
- Active learning: aprende dónde experimentar

### ✅ Robustez Estadística
- Análisis estadístico riguroso (ANOVA, effects, LOF)
- Detección automática de términos significativos
- Estimación empírica de varianza

### ✅ Flexibilidad
- Múltiples tipos de factores (continuo, discreto, categórico, mixture)
- Restricciones complejas (split-plot, irregular)
- Multi-respuesta y multi-objetivo (desirabilidad)

### ✅ Persistencia y Reproducibilidad
- Serialización JSON completa
- Exportación a Excel/CSV
- Reportes HTML auto-generados

### ✅ Interoperabilidad
- Puente a BO (scikit-optimize, Optuna)
- Compatible con pandas/numpy
- Integrable con Ray Tune, MLflow, etc.

---

## 7. Patrones de Uso Recomendados para Agentes

### 7.1 Patrón: Bucle Adaptativo Simple

```python
design = ed.recommend_design(goal="screening", factors=k).design
for wave in range(max_waves):
    y = execute_experiments(design)
    proposal = ed.propose_next_runs(design, response=y, n_add=n_per_wave)
    if not proposal.comparison.worth_it:
        break
    design = proposal.combined
```

### 7.2 Patrón: Multi-Objetivo con Desirabilidad

```python
exp = ed.experiment(goal="optimization", factors=space, responses=["y1", "y2", "y3"])
exp.ingest(results_df, fit=True)
desirability = exp.desirability(goals={"y1": "max", "y2": "min", "y3": "max"})
best_idx = desirability.idxmax()
```

### 7.3 Patrón: Screening → Optimization

```python
# Phase 1: Screening
screen_design = ed.recommend_design(goal="screening", factors=10, budget=16).design
y_screen = run_experiments(screen_design)
actives = identify_active_factors(screen_design, y_screen)  # p < 0.05

# Phase 2: Optimization de factores activos
opt_design = ed.recommend_design(
    goal="optimization",
    factors={f: bounds[f] for f in actives},
    model_order="quadratic"
).design
y_opt = run_experiments(opt_design)
best_conditions = find_optimum(opt_design, y_opt)
```

### 7.4 Patrón: Sequential con Budget Constraint

```python
total_budget = 50
spent = 0
design = initial_design
while spent < total_budget:
    remaining = total_budget - spent
    n_add = min(remaining, 10)
    proposal = ed.propose_next_runs(design, response=y, n_add=n_add, budget=total_budget)
    # ...
    spent += n_add
```

---

## 8. Comparación con Alternativas

| Característica | doekit | scikit-optimize | Optuna | pyDOE2 |
|---|---|---|---|---|
| **DoE Clásico** | ✅ Completo | ❌ No | ❌ No | ✅ Básico |
| **Sequential/Adaptativo** | ✅ Nativo | ✅ BO | ✅ BO | ❌ No |
| **Transparencia** | ✅ Metrics + rationale | ⚠️ BO (black-box) | ⚠️ BO | ✅ Sí |
| **Multi-objetivo** | ✅ Desirabilidad | ⚠️ Limited | ✅ Sí | ❌ No |
| **Análisis estadístico** | ✅ ANOVA, effects, LOF | ❌ No | ❌ No | ❌ No |
| **Reportes** | ✅ HTML automático | ❌ No | ⚠️ Dashboard | ❌ No |
| **Persistencia** | ✅ JSON + workspace | ⚠️ Pickle | ✅ Database | ❌ No |
| **Integración BO** | ✅ Adapter | ✅ Nativo | ✅ Nativo | ❌ No |

---

## 9. Conclusiones para Investigadores con Agentes

### ¿Cuándo usar doekit?

#### ✅ Ideal para:
1. **Optimización experimental sistemática**: Cuando quieres diseños eficientes y estadísticamente rigurosos
2. **Presupuesto limitado**: Maximizar información por experimento
3. **Transparencia requerida**: Necesitas explicar por qué se eligió X diseño
4. **Multi-etapa**: Screening → Optimization → Refinement
5. **Multi-respuesta**: Optimizar varios objetivos simultáneamente
6. **Restricciones complejas**: Mezclas, split-plot, factores difíciles de cambiar

#### ⚠️ Considera alternativas si:
1. **Black-box puro**: Si la función objetivo es completamente opaca y suave → BO puro puede ser mejor
2. **Muy alta dimensionalidad**: > 20 factores → BO o métodos evolutivos
3. **Datos baratos**: Si puedes evaluar millones de puntos → grid search está bien

### Valor único para agentes

doekit permite a un agente de investigación:
- **Tomar decisiones experimentales automáticamente** sin conocimiento experto
- **Cerrar el bucle** de experimentación de forma autónoma
- **Justificar** decisiones con métricas cuantitativas
- **Adaptarse** a resultados experimentales (sequential DoE)
- **Optimizar múltiples objetivos** con desirabilidad
- **Manejar restricciones complejas** (mezclas, factores difíciles)

---

## 10. Recursos y Siguiente Pasos

### Documentación
- Buscar documentación oficial: https://doekit.readthedocs.io/
- Ejemplos en notebooks: Buscar en GitHub repositorio oficial

### Exploración Práctica
```python
# Tutorial rápido
import doekit as ed

# 1. Recomendar diseño
rec = ed.recommend_design(goal="optimization", factors=3, budget=15)
print(rec.summary())

# 2. Evaluar calidad
eval_result = ed.evaluate(rec.design)
print(eval_result.summary())

# 3. Exportar
ed.export_excel(rec.design, "experiment.xlsx", response_names=["yield"])

# 4. Simular datos y analizar
import numpy as np
y = np.random.randn(rec.design.n_runs)
fit = ed.fit_linear_model(rec.design, y)
print(fit.results_frame)

# 5. Proponer siguiente lote
proposal = ed.propose_next_runs(rec.design, response=y, n_add=5)
print(proposal.summary())
```

### Experimentos de Aprendizaje

1. **Screening simple**: 6 factores, Plackett-Burman
2. **Optimization**: Box-Behnken para 3 factores
3. **Sequential**: Iniciar con screening → refinar con optimization
4. **Multi-respuesta**: Desirabilidad con 3 respuestas
5. **Mixture**: Diseño simplex lattice para formulación
6. **Split-plot**: Factores difíciles de cambiar

---

## Resumen Ejecutivo

**doekit** es una librería de Design of Experiments (DoE) completa y production-ready que permite a agentes de investigación:

1. **Diseñar experimentos óptimos** automáticamente con `recommend_design()`
2. **Cerrar el bucle experimental** con `propose_next_runs()` (sequential DoE)
3. **Analizar resultados** con ANOVA, effects, power analysis
4. **Optimizar multi-objetivo** con funciones de desirabilidad
5. **Integrarse con BO** (Optuna, scikit-optimize) vía adaptadores
6. **Persistir estado** completo en JSON/Excel
7. **Generar reportes** HTML automáticamente

**Diferenciador clave**: Combina la rigurosidad estadística del DoE clásico con la adaptabilidad del aprendizaje activo, proporcionando transparencia y explicabilidad que los métodos de BO puros no ofrecen.

**Ideal para agentes que necesitan**:
- Minimizar número de experimentos (datos caros)
- Justificar decisiones experimentales
- Manejar múltiples respuestas y restricciones complejas
- Cerrar el bucle experimental de forma autónoma

---

**Instalación**: `pip install doekit` ✅  
**Versión analizada**: 0.7.3  
**Licencia**: Revisar en repositorio oficial  
**Mantenimiento**: Activo (última versión 2024)
