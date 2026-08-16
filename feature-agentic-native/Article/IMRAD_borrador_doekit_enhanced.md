# IMRaD Draft v0: doekit-enhanced as an Agent-Oriented Experimental Decision System

## Title
From Numeric DoE Outputs to Agent-Operational Decisions: A Methodological and Empirical Assessment of doekit-enhanced

## Abstract
This paper presents doekit-enhanced, an extension layer over doekit that reframes Design of Experiments (DoE) for AI agents. The central hypothesis is that numeric-only outputs are necessary but insufficient for autonomous research workflows, where agents must justify decisions, quantify uncertainty, and maintain auditable traces across iterative waves. We propose a semantic-first architecture with five components: semantic interpretation, policy-based decision engine, uncertainty handling, convergence monitoring, and memory-driven transfer. Empirically, we evaluate the approach under controlled With/Without conditions using real datasets and paired task execution. In a pilot (20 runs), Agent_With improves ImpactScore by +3.02% (84.58 vs 82.10, p=0.031, d=0.83) but increases total execution time (+429%). In the meta-design extension (126 runs), the global ImpactScore gain remains (+3.01%, d=0.86), with consistent gains in mean_power (+10.00%), predicted_gain (+23.67%), and uncertainty_index (-20.18%), while d_efficiency drops (-10.93%). Results suggest a quality-latency trade-off: doekit-enhanced improves decision robustness and traceability but requires latency optimization and task-specific design calibration. We discuss implications for adoption in research and industrial settings.

Keywords: design of experiments, autonomous agents, sequential experimentation, decision support, uncertainty, reproducibility

## 1. Introduction
### 1.1 Problem statement
Classical DoE libraries are built for human experts who can interpret numeric diagnostics and translate them into action. Agent-based workflows require an additional layer: machine-usable reasoning that is explicit, auditable, and stable across repeated decisions.

The baseline problem is practical: outputs such as D-efficiency, power, or worth-it flags do not directly encode operational rationale for autonomous execution. Without explicit rationale, agents may behave inconsistently, fail to communicate risk, or overfit local heuristics.

### 1.2 Research gap
Current DoE tooling largely optimizes design quality metrics but under-specifies three capabilities required by autonomous systems:
1. semantic interpretation of quantitative evidence;
2. decision policies with explicit trade-offs;
3. experiment-memory and convergence mechanisms for multi-wave operation.

### 1.3 Contributions
This work contributes:
1. A methodological framing of DoE for autonomous agents (semantic-first + policy-governed decisions).
2. An extensible architecture (doekit-enhanced) that remains additive to doekit.
3. A real-data empirical protocol comparing Agent_With vs Agent_Without.
4. A critical reading of trade-offs: quality gains under measurable latency costs.

### 1.4 Research questions
RQ1: Does enabling doekit-enhanced improve global decision quality (ImpactScore) versus a control workflow?

RQ2: Are quality gains stable across tasks and difficulty strata?

RQ3: What operational trade-offs emerge between decision quality, uncertainty reduction, and execution latency?

## 2. Methods
### 2.1 System design philosophy ("spirit" of the library)
The project follows five principles:
1. Additive compatibility: enhancements do not break baseline doekit usage.
2. Semantic-first results: every critical numeric output can be lifted to an interpretable decision artifact.
3. Policy explicitness: stop/continue/refine decisions are policy-bound, not hidden heuristics.
4. Auditable operation: each run yields machine-checkable evidence.
5. Modular autonomy: semantic, decision, monitoring, memory, and integrations can be used independently or as pipeline.

### 2.2 Architectural decomposition
The enhanced stack is organized in modules:
1. semantic: interpreters, builders, templates for numeric-to-semantic conversion.
2. decision: policy and scoring layer for action selection.
3. monitoring: convergence and diagnostics.
4. memory: experiment store and transfer of priors.
5. integrations: pragmatic external optimization adapters.

This decomposition preserves separation of concerns while enabling end-to-end autonomous loops.

### 2.3 Experimental design
#### 2.3.1 Conditions
Two conditions are compared:
1. Agent_With: access to doekit-enabled methods and enhanced decision support.
2. Agent_Without: no DoEkit access; baseline statistical/manual workflow.

#### 2.3.2 Tasks and data
Three cyclic tasks are used:
1. Task-01: initial optimization design.
2. Task-02: statistical modeling and diagnostics.
3. Task-03: sequential iteration and next-wave decision.

Approved real datasets are assigned by task (california_housing, diabetes, wine), with strict real-data compliance checks.

#### 2.3.3 Validity controls
The protocol enforces:
1. paired execution (same task and seed per pair);
2. fixed budget constraints and model-order settings;
3. sandbox isolation by condition;
4. required run artifacts: recommendation.json, metrics.json, trace.log, evidence.json, and executable code artifact.

#### 2.3.4 Metrics
The main endpoint is ImpactScore:
ImpactScore = 0.35*TechnicalQuality + 0.30*Efficiency + 0.20*Risk + 0.15*UX

Secondary outcomes include total_time_sec, d_efficiency, mean_power, predicted_gain, uncertainty_index, and risk counters.

### 2.4 Two-phase empirical strategy
Phase 1 (pilot): 20 runs (10 per condition), balanced pairing.

Phase 2 (meta-design V2): 126 valid runs, balanced by condition, task, and difficulty strata. The design rationale used a D-optimal basis for planning and crossed operational factors (prompt strictness, context visibility, timeout budget) with condition.

### 2.5 Statistical reading strategy
Primary interpretation combines:
1. effect direction and magnitude (delta, percent change),
2. inferential signal when available (Mann-Whitney p-values),
3. effect size (Cohen's d),
4. consistency across tasks and difficulty strata.

## 3. Results
### 3.1 Pilot (20 runs)
Global endpoint:
1. ImpactScore: 84.58 (With) vs 82.10 (Without), delta +2.48 (+3.02%), p=0.031, d=0.83.
2. Total time: 0.1438 vs 0.0272, delta +429.37%.

Key technical metrics:
1. mean_power: +10.00% (favorable for With).
2. predicted_gain: +23.85% (favorable).
3. uncertainty_index: -20.12% (lower is better; favorable).
4. d_efficiency: -10.70% (unfavorable for With).

Task-level ImpactScore deltas are positive in all three tasks (~+2.0 to +2.7 points).

### 3.2 Meta-design extension (126 runs)
Balance check:
1. 63 runs per condition.
2. 42 runs per task.
3. 42 runs per difficulty stratum (high, medium, low).

Global endpoint:
1. ImpactScore: 84.62 (With) vs 82.15 (Without), delta +2.47 (+3.01%), p<0.001, d=0.86.
2. Total time: 0.1767 vs 0.0313, delta +465.13%.

Technical metrics:
1. mean_power: +10.00%.
2. predicted_gain: +23.67%.
3. uncertainty_index: -20.18%.
4. d_efficiency: -10.93%.

Pattern stability:
1. ImpactScore gain remains positive across all tasks.
2. Gains persist in all difficulty strata.
3. Risk counters remain near floor (limited discrimination in this dataset).

## 4. Discussion
### 4.1 Main interpretation
Results support a robust quality-latency trade-off:
1. Decision quality and confidence-related proxies improve with doekit-enhanced.
2. Execution latency increases substantially in current implementation.
3. d_efficiency underperforms despite global quality gains, indicating mismatch between some design proxies and end-to-end decision utility.

### 4.2 Methodological implications
For agentic research systems, the objective should not be single-metric optimization. A policy-driven multi-objective framework is required to balance:
1. quality of inference,
2. uncertainty reduction,
3. computational/operational cost,
4. reproducible evidence quality.

### 4.3 Practical adoption criteria
A realistic deployment criterion should include:
1. positive and stable primary endpoint effect;
2. non-degrading risk behavior;
3. acceptable latency under the target operating context.

This allows selective adoption: high-stakes decision contexts may tolerate extra latency; real-time contexts may require aggressive optimization first.

### 4.4 Threats to validity
1. Pilot sample size is limited.
2. Some risk metrics saturate at zero, limiting sensitivity.
3. time metrics reflect the current pipeline implementation and may mix computation and orchestration overhead.
4. Additional mixed-effects modeling is still needed for full hierarchical inference.

### 4.5 Next methodological step
The next iteration should estimate:
ImpactScore ~ Condition + Difficulty + Condition:Difficulty + (1|Task) + (1|Seed) + (1|Source)

This model would separate fixed treatment effects from cross-task and source-level variability.

## 5. Conclusion
doekit-enhanced demonstrates a consistent increase in composite decision quality under controlled real-data conditions, with simultaneous uncertainty reduction and better gain-oriented behavior. The cost is higher latency and unresolved d_efficiency degradation in selected settings. Therefore, the evidence supports conditional adoption with explicit optimization priorities: preserve semantic and decision gains while reducing operational overhead and calibrating design strategy by task.

## 6. Reproducibility Appendix (Draft)
Minimum reproducibility package per run:
1. recommendation.json
2. metrics.json
3. trace.log with REAL_DATA_CONFIRMED=true
4. evidence.json with approved source mapping
5. executable artifact (.py/.ipynb/.md with runnable code)

Suggested release bundle for paper supplement:
1. protocol documents (pilot + meta-design V2)
2. experiment configuration file
3. consolidated metrics tables
4. report generation scripts
5. versioned commit hash and environment lockfile
