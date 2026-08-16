# Capa agéntica (interpret · decide · monitor · memory)

doekit está hecho para un **público híbrido** — personas y agentes LLM — sobre un
mismo motor. La capa agéntica (**doekit 0.9**) es lo que permite a un agente leer
resultados y decidir *cuándo actuar y cuándo no confiar en sí mismo*, bajo una
única regla:

> **Los hechos vienen de doekit; el juicio, del usuario.** Cada señal de abajo se
> *compone* de los propios resultados de doekit (`to_dict()` / `interpret`); nada se
> inventa.

La capa es la columna semántica del flujo de razonamiento *brief → recommend →
evaluate → \[lab: ingest\] → analyze → **interpret** → **decide** → next*.

## Interpretar — la capa semántica

`interpret(result)` le da a cualquier resultado — Recommendation, DesignEvaluation,
FitResult, NextRunsProposal, DesignComparison — una **lectura uniforme** como
`Interpretation` (`doekit.Interpretation/1`):

```python
view = ed.interpret(result)
view.for_llm()     # un bloque de contexto: titular + razonamiento + avisos + siguientes pasos
view.to_dict()     # los mismos hechos, estructurados
```

Compone `rationale` / `caveats` / `summary` del objeto subyacente — la **misma
verdad en dos superficies**: prosa/`summary()` para una persona, `for_llm()` para
un agente. Nunca fabrica una eficiencia, un p-valor ni un ranking.

## Decidir — el motor de decisión

`decide_next_action(context)` (y el agregado `Experiment.decide_next`) mapea señales
a una acción: **stop · augment · refine · redesign** (`doekit.Decision/1`).

**Los hard gates ganan primero**, antes de cualquier score:

- diseño rank-deficient → **redesign**;
- presupuesto agotado, o `check_convergence` dice parar → **stop**.

Si no, decide un **score compuesto** transparente, con un desglose beneficio / costo
/ riesgo / incertidumbre que puedes leer en `score.to_dict()`:

$$
\text{composite} = w_b\,\text{benefit} - w_c\,\text{cost} - w_r\,\text{risk}
- w_u\,\text{uncertainty},
$$

con pesos por defecto $w_b{=}1.0,\ w_c{=}0.7,\ w_r{=}0.8,\ w_u{=}0.6$ y
$\text{cost} = \min\!\big(2,\ \text{extra\_runs}/\text{budget\_remaining}\big)$. El
scoring es **consciente de la intención**:

- **learn:** $\text{benefit} = 0.6\,\mathrm{norm}(d\_gain, 20) + 0.4\,\mathrm{norm}(p\_gain, 0.2)$,
  $\ \text{risk} = \max(0,\ -g\_delta/10)$ — ganancias de precisión y potencia,
  penalizadas por una caída de varianza de predicción (G).
- **optimize:** $\text{benefit} = 1$ si $\text{predicted\_improvement} > 0$ si no $0$,
  $\ \text{risk} = \text{uncertainty}$ — y **nunca se penaliza** la caída de
  D-eficiencia que el loop de surrogate puede causar.

Las políticas mueven los umbrales sin tocar el score: `ThresholdPolicy` (por
defecto), `RiskAdaptivePolicy` (más conservadora cuando el riesgo es alto),
`BudgetAwarePolicy`. El `gate_board` dentro de `AutomaticConclusions` **delega** en
este mismo motor — una sola lógica de decisión, no dos.

## Monitorizar — convergencia y diagnósticos por paso

Dos primitivas alimentan el motor y exponen la salud por oleada:

- **`check_convergence(history, metric_key=...)`** (`doekit.ConvergenceResult/1`):
  declara convergencia cuando los últimos cambios marginales se mantienen dentro de
  la tolerancia (`best_so_far` para optimize, una métrica delta para learn). Su
  `should_stop` es un hard gate.
- **`diagnose_step(metrics, ...)`** (`doekit.DiagnosticsReport/1`): señala ganancia
  de potencia pobre, degradación de predicción (G), desbordamiento de presupuesto
  (un bloqueador), alta incertidumbre y convergencia. `Experiment.decide_next` **lo
  corre automáticamente** y adjunta el reporte en `decision.metadata["diagnostics"]`.

```python
decision = exp.decide_next(intent="optimize", history=best_so_far_por_gen)
decision.action, decision.confidence
decision.metadata["diagnostics"]   # avisos de potencia / G-eff / presupuesto / incertidumbre
```

## Memoria — meta-aprendizaje entre estudios

`ExperimentHistory` (su almacén es el workspace trazable proyecto→waves) deja que
oleadas pasadas informen a las nuevas: `learn_priors` estima priors de efectos y
`historical_recommendation` transfiere señales a un brief nuevo.

```python
hist = ed.ExperimentHistory.from_project(proj)
priors = ed.learn_priors(hist)
rec = ed.historical_recommendation(goal=..., factors=..., history=hist)
```

## Servida por MCP

Todo el loop se expone a agentes por el [servidor MCP](../agents/mcp.md):
`propose_and_decide` devuelve `interpret` + `decide` + las señales de monitoring
(`diagnostics`, y `convergence` cuando se pasa una `history`) — la capa agéntica, de
extremo a extremo.
