# Agentes

Skill portable de **diseñador de experimentos doekit**: enseña a un agente el
bucle DoE (brief → recommend → evaluate → lab → ingest → analyze → next) sin
inventar métricas.

## Paquete de skill (copia estos dos archivos)

| Archivo | Rol |
|---------|-----|
| [SKILL.md](SKILL.md) | Flujo, gates, plantilla de respuesta. Conserva el nombre `SKILL.md`. |
| [reference.md](reference.md) | Cheat sheet de API autocontenido |

No hace falta este index dentro de la carpeta de skills — solo `SKILL.md` +
`reference.md`. El paquete es autocontenido.

**Contrato:** el agente posee contexto y decisiones; doekit calcula rankings,
eficiencias, ajustes y reportes. Siempre llamar a la librería y leer `to_dict()` /
resúmenes.

## Instalación

### Cursor

```text
SKILL.md + reference.md  →  .cursor/skills/doekit-experiment-designer/
```

- Proyecto: `.cursor/skills/doekit-experiment-designer/`
- Personal: `~/.cursor/skills/doekit-experiment-designer/`
- Nunca `~/.cursor/skills-cursor/` (reservado).

Activa con preguntas de experimento / DoE / doekit, o `@doekit-experiment-designer`.

### Claude

```text
SKILL.md + reference.md  →  .claude/skills/doekit-experiment-designer/
```

### VS Code / Copilot

Apunta instrucciones custom a `docs/agents/SKILL.md` y `docs/agents/reference.md`,
o menciónalos con `@` en el chat.

## Higiene en sesión

- Sin secretos en factores, metadata o reportes.
- Preferir `ed.experiment(...)` / `Experiment.to_dict()` para handoff; exportar plantilla con `exp.export_csv`.
- Tratar respuestas de lab en HTML como sensibles si aplica.
