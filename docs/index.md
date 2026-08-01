# doekit

**Design of Experiments (DoE) in Python**: screening, factorial designs, response
surface and optimal design (D/A/I) — plus a **design-evaluation layer** that most
Python DoE libraries lack.

Depends on `numpy`, `pandas`, `scipy` and `statsmodels`. `matplotlib` is optional
(plots and HTML reports).

```bash
pip install doekit            # core
pip install "doekit[plot]"    # with plots (matplotlib)
pip install "doekit[report]"  # with HTML reports
```

## The DoE funnel

Classic DoE is a funnel: **screening** (which factors matter?) →
**response surface** (where is the optimum?), with **optimal design** as an
alternative route when the standard templates do not fit.

```python
import doekit as ed

# 1) Screening — Plackett-Burman for 6 factors in 8 runs
pb = ed.plackett_burman(6)

# 2) Response surface — Box-Behnken in natural units
bb = ed.box_behnken({"temp": (20, 80), "ph": (3, 9), "conc": (0.1, 0.5)})

# 3) Optimal design — D-optimal subset from a candidate set
cand = ed.random_design([ed.ContinuousFactor("x1", -1, 1),
                         ed.ContinuousFactor("x2", -1, 1)], n=200, seed=0)
cand.model = ed.Model.parse("0 ~ x1 + x2 + x1:x2")
opt = ed.optimal_design(cand, n_runs=12, criterion="D", n_starts=5, seed=1)
```

## What sets doekit apart: evaluate the design, not just build it

Most Python DoE libraries **only generate** designs. `doekit` also gives each one a
reproducible **quality report card** answering *"how far is my design from the
theoretical optimum?"* — the half of the work that commercial tools (JMP,
Design-Expert) do:

```python
report = ed.evaluate(bb, effect_size=1.0, sigma=1.0)
print(report.summary())
#   D-efficiency, A-efficiency, G-efficiency, SPV distribution (FDS),
#   power per term, VIF, alias structure ...

ed.report(bb, response=y)   # -> a report/ folder: index.html + images/ + data/
```

## Where to go next

- **Theory** — one page per methodology, each with *motivation → theory (math) →
  a doekit example*. Start with [Factors & coding](theory/factors-and-coding.md).
- **API reference** — auto-generated from the source docstrings.
- **Guide** — the [notebooks](guide/notebooks.md) walk through real, domain-specific
  cases (chemistry, ML, quantum ML) with the *build → evaluate → benchmark* pattern.
