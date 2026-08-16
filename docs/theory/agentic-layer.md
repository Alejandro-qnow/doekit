# Agentic layer (interpret · decide · monitor · memory)

doekit is built for a **hybrid audience** — people and LLM agents — on one engine.
The agentic layer (**doekit 0.9**) is what lets an agent read results and decide
*when to act and when not to trust itself*, under one rule:

> **Facts from doekit, judgment from the user.** Every signal below is *composed*
> from doekit's own results (`to_dict()` / `interpret`); nothing is invented.

The layer is the semantic backbone of the reasoning flow *brief → recommend →
evaluate → \[lab: ingest\] → analyze → **interpret** → **decide** → next*.

## Interpret — the semantic layer

`interpret(result)` gives any result — Recommendation, DesignEvaluation, FitResult,
NextRunsProposal, DesignComparison — a **uniform reading** as an `Interpretation`
(`doekit.Interpretation/1`):

```python
view = ed.interpret(result)
view.for_llm()     # a context block: headline + reasoning + warnings + next steps
view.to_dict()     # the same facts, structured
```

It composes `rationale` / `caveats` / `summary` from the underlying object — the
**same truth in two surfaces**: prose/`summary()` for a person, `for_llm()` for an
agent. It never fabricates an efficiency, a p-value, or a ranking.

## Decide — the decision engine

`decide_next_action(context)` (and the aggregate `Experiment.decide_next`) maps
signals to an action: **stop · augment · refine · redesign** (`doekit.Decision/1`).

**Hard gates win first**, before any scoring:

- design rank-deficient → **redesign**;
- budget exhausted, or `check_convergence` says stop → **stop**.

Otherwise a transparent **composite score** decides, with a benefit / cost / risk /
uncertainty breakdown you can read in `score.to_dict()`:

$$
\text{composite} = w_b\,\text{benefit} - w_c\,\text{cost} - w_r\,\text{risk}
- w_u\,\text{uncertainty},
$$

with default weights $w_b{=}1.0,\ w_c{=}0.7,\ w_r{=}0.8,\ w_u{=}0.6$ and
$\text{cost} = \min\!\big(2,\ \text{extra\_runs}/\text{budget\_remaining}\big)$. The
scoring is **intent-aware**:

- **learn:** $\text{benefit} = 0.6\,\mathrm{norm}(d\_gain, 20) + 0.4\,\mathrm{norm}(p\_gain, 0.2)$,
  $\ \text{risk} = \max(0,\ -g\_delta/10)$ — precision and power gains, penalized by
  a prediction-variance (G) drop.
- **optimize:** $\text{benefit} = 1$ if $\text{predicted\_improvement} > 0$ else $0$,
  $\ \text{risk} = \text{uncertainty}$ — and it is **never penalized** for the
  D-efficiency drop the surrogate loop can cause.

Policies shift the thresholds without touching the score: `ThresholdPolicy`
(default), `RiskAdaptivePolicy` (more conservative when risk is high),
`BudgetAwarePolicy`. The `gate_board` inside `AutomaticConclusions` **delegates** to
this same engine — one decision logic, not two.

## Monitor — convergence and step diagnostics

Two primitives feed the engine and expose per-wave health:

- **`check_convergence(history, metric_key=...)`** (`doekit.ConvergenceResult/1`):
  declares convergence when the last few marginal changes stay within tolerance
  (`best_so_far` for optimize, a delta metric for learn). Its `should_stop` is a
  hard gate.
- **`diagnose_step(metrics, ...)`** (`doekit.DiagnosticsReport/1`): flags thin power
  gain, prediction (G) degradation, budget overflow (a blocker), high uncertainty,
  and convergence. `Experiment.decide_next` **runs it automatically** and attaches
  the report to `decision.metadata["diagnostics"]`.

```python
decision = exp.decide_next(intent="optimize", history=best_so_far_per_gen)
decision.action, decision.confidence
decision.metadata["diagnostics"]   # power / G-eff / budget / uncertainty warnings
```

## Memory — meta-learning across studies

`ExperimentHistory` (its store is the traceable project→waves workspace) lets past
waves inform new ones: `learn_priors` estimates effect priors, and
`historical_recommendation` transfers signals into a fresh brief.

```python
hist = ed.ExperimentHistory.from_project(proj)
priors = ed.learn_priors(hist)
rec = ed.historical_recommendation(goal=..., factors=..., history=hist)
```

## Served over MCP

The whole loop is exposed to agents by the [MCP server](../agents/mcp.md):
`propose_and_decide` returns `interpret` + `decide` + the monitoring signals
(`diagnostics`, and `convergence` when a `history` is passed) — the agentic layer,
end to end.
