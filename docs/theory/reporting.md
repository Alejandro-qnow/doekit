# Reporting

## Motivation

The report is the artifact of the evaluation layer: the thing a scientist shares with
colleagues or management. It closes the loop **build → evaluate → communicate**, and
— later — is what the MCP server returns as a resource.

## What the report contains

A rule-based (deterministic, no LLM) HTML report with:

1. **Executive summary** — quality verdict, significant factors, top recommendation.
2. **Methodology** — design kind and *why*, factors with ranges, the model.
3. **Design matrix** — collapsible.
4. **Design quality** — D/A/G-efficiency cards with a semaphore, FDS plot, power, VIF,
   alias heatmap, each with a plain-language gloss.
5. **Results** (with a response) — fitted model, coefficients with significance,
   $R^2$, half-normal plot.
6. **Anomalous values** — outliers (studentized residual $|r|>3$), high leverage
   ($h_{ii} > 2p/N$), influential (Cook's $D > 1$).
7. **Conclusions and recommendations** — rule-based narrative.

## Output modes

- **folder** (default): a `report/` folder with `index.html`, `report.css`,
  `images/*.png` and `data/*.csv` (design matrix, coefficients, efficiencies, power,
  VIF, FDS, anomalies) — portable assets a researcher can reuse.
- **self-contained**: a single `.html` with inlined CSS and base64 plots — emailable.

The report is **bilingual** (`lang="en"` / `"es"`).

## In doekit

```python
import doekit as ed

ed.report(bb, response=y)                        # -> report/ folder (default)
ed.report(bb, response=y, self_contained=True)   # -> single .html
ed.report(bb, response=y, lang="es")             # Spanish

# or as an argument of the experiment functions:
ed.fit_linear_model(bb, y, report="report/")     # path in fit.report_path
ed.evaluate(bb, report=True)                     # path in ev.report_path
```

Requires the `[report]` extra (matplotlib).

## See also

- Theory: [Evaluation metrics](evaluation-metrics.md)
- API: [`report`, `report_summary`](../api/analysis-report.md)
