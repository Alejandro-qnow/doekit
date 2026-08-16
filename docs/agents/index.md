# Agents

doekit is **hybrid**: the same engine serves people and **LLM agents**. Agents
reach it two ways, sharing one contract — *facts from doekit, judgment from the
agent*:

- **Skill** — the portable **experiment-designer** skill teaches the DoE loop
  (brief → recommend → evaluate → lab → ingest → analyze → **interpret** → decide →
  next) without inventing metrics. Below.
- **[MCP server](mcp.md)** — the same loop as callable tools (recommend / evaluate
  / propose_and_decide) over the Model Context Protocol.

## Skill package (copy these two files)

| File | Role |
|------|------|
| [SKILL.md](SKILL.md) | Workflow, gates, reply template. Keep filename `SKILL.md`. |
| [reference.md](reference.md) | Self-contained API cheat sheet |

Do **not** require this index page inside the skills folder — only `SKILL.md` +
`reference.md`. The package is self-contained.

**Contract:** agent owns process context and decisions; doekit owns rankings,
efficiencies, fits, and reports. Always call the library and read `to_dict()` /
summaries.

## Install

### Cursor

```text
SKILL.md + reference.md  →  .cursor/skills/doekit-experiment-designer/
```

- Project: `.cursor/skills/doekit-experiment-designer/`
- Personal: `~/.cursor/skills/doekit-experiment-designer/`
- Never use `~/.cursor/skills-cursor/` (reserved).

Trigger with experiment / DoE / doekit questions, or `@doekit-experiment-designer`.

### Claude

```text
SKILL.md + reference.md  →  .claude/skills/doekit-experiment-designer/
```

### VS Code / Copilot

Point custom instructions at `docs/agents/SKILL.md` and `docs/agents/reference.md`,
or `@`-mention them in chat.

## Session hygiene

- No secrets in factor names, metadata, or reports.
- Prefer `ed.experiment(...)` / `Experiment.to_dict()` for handoff; export run sheets with `exp.export_csv`.
- For multi-session research, persist with `ed.project(name)` → `wave` →
  `automatic-conclusions/conclusions.json` (read gates; do not invent metrics).
- Treat lab responses in HTML reports as sensitive when required.

## Traceable workspace

```text
experiments/experiment_project_<slug>/
  PROJECT.json
  waves/wave_001/
    doe-configuration/   # INPUT: experiment.json, design.json, thresholds.json
    data/                # run_sheet.csv (+ responses.csv after lab)
    results/             # evaluation.json, fit.json, next_runs.json
    reports/             # optional HTML
    automatic-conclusions/  # conclusions.json + .md (LLM/agent/human)
    metadata/            # provenance + checksums
    assets/              # researcher auxiliaries
```

- **Wave** = one DoE cycle (not a lab row). Lab row ids stay in `run_sheet.csv` as `run_id`.
- Agents should load `conclusions.json` (`gate_board`, `rules`, `facts`) and only paraphrase those strings.
