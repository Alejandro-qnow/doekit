# Agent instructions (doekit)

For Design of Experiments workflows with this library, follow the portable skill:

- **Skill:** [`docs/agents/SKILL.md`](docs/agents/SKILL.md)
- **API cheat sheet:** [`docs/agents/reference.md`](docs/agents/reference.md)
- **Install / Cursor / Claude / VS Code:** [`docs/agents/index.md`](docs/agents/index.md)

**Rule of thumb:** elicit the experimental brief; call doekit
(`recommend_design`, `evaluate`, `Experiment`, `propose_next_runs`, `report`);
interpret JSON and caveats. Do not invent efficiencies or rankings. Do not
treat doekit as a drop-in replacement for Optuna, MLflow, or CVXPY.
