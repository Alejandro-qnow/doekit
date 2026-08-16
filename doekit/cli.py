"""Minimal CLI for doekit design, evaluation, and project workflows.

Subcommands: ``recommend``, ``evaluate``, ``experiment``, and ``project``
(``init``, ``sync``, ``conclude``). Invoke via ``python -m doekit.cli`` or the
``doekit`` console script when installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_factors(spec: str):
    """Parse ``n`` or ``name:low:high,name2:low:high``."""
    spec = spec.strip()
    if spec.isdigit():
        return int(spec)
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise SystemExit(
                f"bad factor spec {part!r}; use name:low:high or an integer"
            )
        name, lo, hi = bits[0], float(bits[1]), float(bits[2])
        out[name] = (lo, hi)
    return out


def cmd_recommend(args):
    """Run ``doekit recommend``: suggest a design method and optional CSV export.

    Parameters
    ----------
    args : argparse.Namespace
        ``--factors``, ``--goal``, ``--budget``, ``--model-order``, ``--mixture``,
        ``--hard-to-change``, ``--irregular``, ``--seed``, ``--export``, ``--json``.
    """
    import doekit as ed
    factors = _parse_factors(args.factors)
    kwargs = {}
    if args.mixture:
        kwargs["mixture"] = True
    if args.hard_to_change:
        kwargs["hard_to_change"] = args.hard_to_change.split(",")
    if args.irregular:
        kwargs["constraints"] = ed.Constraints(irregular=True)
    rec = ed.recommend_design(
        args.goal, factors, budget=args.budget, model_order=args.model_order,
        seed=args.seed, **kwargs,
    )
    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(rec.summary())
        if args.export:
            path = ed.experiment(design=rec.design).export_csv(args.export)
            print(f"Wrote run sheet: {path}")


def cmd_evaluate(args):
    """Run ``doekit evaluate``: score a Design JSON file.

    Parameters
    ----------
    args : argparse.Namespace
        Positional ``design_json`` path; ``--n-region``, ``--seed``, ``--json``.
    """
    import doekit as ed
    from doekit.domain.design import Design
    data = json.loads(Path(args.design_json).read_text(encoding="utf-8"))
    design = Design.from_dict(data)
    ev = ed.evaluate(design, n_region=args.n_region, seed=args.seed)
    if args.json:
        print(json.dumps(ev.to_dict(), indent=2))
    else:
        print(ev.summary())


def cmd_experiment(args):
    """Run ``doekit experiment``: build and evaluate an Experiment from flags.

    Parameters
    ----------
    args : argparse.Namespace
        ``--factors``, ``--goal``, ``--budget``, ``--responses``, ``--export``,
        ``--n-region``, ``--seed``, ``--json``.
    """
    import doekit as ed
    factors = _parse_factors(args.factors)
    responses = args.responses.split(",") if args.responses else ["y"]
    exp = ed.experiment(
        goal=args.goal, factors=factors, budget=args.budget,
        responses=responses, seed=args.seed,
    )
    exp.evaluate(n_region=args.n_region, seed=args.seed)
    if args.export:
        path = exp.export_csv(args.export)
        print(f"Wrote run sheet: {path}")
    if args.json:
        print(json.dumps(exp.to_dict(), indent=2))
    else:
        print(f"Experiment: {exp.design.metadata.get('kind')} "
              f"({exp.design.n_runs} runs)")
        if exp.recommendation:
            print(exp.recommendation.rationale)
        if exp.evaluation:
            print(exp.evaluation.summary())
        print("\nPlan (head):")
        print(exp.plan.head().to_string(index=False))


def _load_experiment_for_project(args):
    import doekit as ed
    if args.experiment_json:
        data = json.loads(Path(args.experiment_json).read_text(encoding="utf-8"))
        return ed.Experiment.from_dict(data)
    if not args.factors:
        raise SystemExit("provide --experiment-json or --factors")
    factors = _parse_factors(args.factors)
    responses = args.responses.split(",") if args.responses else ["y"]
    exp = ed.experiment(
        goal=args.goal, factors=factors, budget=args.budget,
        responses=responses, seed=args.seed,
    )
    exp.evaluate(n_region=args.n_region, seed=args.seed)
    return exp


def cmd_project_init(args):
    """Run ``doekit project init``: create ``experiment_project_<slug>/``.

    Parameters
    ----------
    args : argparse.Namespace
        ``--name``, ``--root``, ``--description``.
    """
    import doekit as ed
    proj = ed.ExperimentProject.create(
        args.name, root=args.root, description=args.description or "",
    )
    print(f"Project ready: {proj.path}")
    return 0


def cmd_project_sync(args):
    """Run ``doekit project sync``: write or update a wave from an Experiment.

    Parameters
    ----------
    args : argparse.Namespace
        ``--path`` (project or wave dir), ``--experiment-json`` or factor flags,
        ``--goal``, ``--budget``, ``--responses``, ``--n-region``, ``--seed``,
        ``--report``.
    """
    import doekit as ed
    from doekit.presentation.workspace import Wave, open_project
    path = Path(args.path)
    exp = _load_experiment_for_project(args)
    if (path / "PROJECT.json").exists():
        proj = open_project(path)
        wave = proj.new_wave(exp, seed=args.seed)
    elif (path / "manifest.json").exists():
        wave = Wave(path)
        wave.sync(exp, seed=args.seed, write_report=args.report)
    else:
        raise SystemExit(f"path is neither a project nor a wave: {path}")
    print(f"Synced wave: {wave.path} (status={wave.manifest.get('status')})")
    return 0


def cmd_project_conclude(args):
    """Run ``doekit project conclude``: write automatic conclusions for a wave.

    Parameters
    ----------
    args : argparse.Namespace
        ``--path`` (wave dir), ``--lang``, ``--report``, ``--json``.
    """
    import doekit as ed
    from doekit.presentation.workspace import Wave
    path = Path(args.path)
    if not (path / "manifest.json").exists():
        raise SystemExit(f"conclude requires a wave path, got: {path}")
    wave = Wave(path)
    exp = wave.load_experiment()
    conclusions = wave.conclude(
        exp, lang=args.lang, write_html=args.report,
    )
    out = path / "automatic-conclusions" / "conclusions.json"
    if args.json:
        print(json.dumps(conclusions, indent=2))
    else:
        process = (conclusions.get("gate_board") or {}).get("process") or {}
        print(f"Wrote {out}")
        print(f"process gate: {process.get('status')}")
        print(f"quality: {(conclusions.get('gate_board') or {}).get('quality')}")
    return 0


def main(argv=None) -> int:
    """Parse CLI arguments and dispatch to the selected subcommand.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success).

    Notes
    -----
    Subcommands: ``recommend``, ``evaluate``, ``experiment``, ``project init``,
    ``project sync``, ``project conclude``. Use ``--help`` on any subcommand
    for flag details.
    """
    p = argparse.ArgumentParser(prog="doekit", description="doekit DoE CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("recommend", help="Recommend a design method")
    pr.add_argument("--factors", required=True,
                    help="Integer N or name:low:high,name2:low:high")
    pr.add_argument("--goal", default="screening",
                    choices=["screening", "optimization"])
    pr.add_argument("--budget", type=int, default=None)
    pr.add_argument("--model-order", default=None,
                    choices=["linear", "interactions", "quadratic"])
    pr.add_argument("--mixture", action="store_true")
    pr.add_argument("--hard-to-change", default=None,
                    help="Comma-separated whole-plot factor names")
    pr.add_argument("--irregular", action="store_true")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--export", default=None, help="CSV run-sheet path")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_recommend)

    pe = sub.add_parser("evaluate", help="Evaluate a Design JSON file")
    pe.add_argument("design_json", help="Path to Design.to_dict() JSON")
    pe.add_argument("--n-region", type=int, default=4000)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=cmd_evaluate)

    px = sub.add_parser("experiment", help="Build Experiment from goal+factors")
    px.add_argument("--factors", required=True)
    px.add_argument("--goal", default="screening",
                    choices=["screening", "optimization"])
    px.add_argument("--budget", type=int, default=None)
    px.add_argument("--responses", default="y",
                    help="Comma-separated response names")
    px.add_argument("--export", default=None, help="CSV run-sheet path")
    px.add_argument("--n-region", type=int, default=4000)
    px.add_argument("--seed", type=int, default=0)
    px.add_argument("--json", action="store_true")
    px.set_defaults(func=cmd_experiment)

    pp = sub.add_parser("project", help="Traceable experiment project workspace")
    psub = pp.add_subparsers(dest="project_cmd", required=True)

    pi = psub.add_parser("init", help="Create experiment_project_<slug>/")
    pi.add_argument("--name", required=True, help="Human project name")
    pi.add_argument("--root", default="experiments",
                    help="Parent directory (default: experiments)")
    pi.add_argument("--description", default="")
    pi.set_defaults(func=cmd_project_init)

    ps = psub.add_parser("sync", help="Write a wave from Experiment JSON or flags")
    ps.add_argument("--path", required=True,
                    help="Project dir or wave dir")
    ps.add_argument("--experiment-json", default=None,
                    help="Path to Experiment.to_dict() JSON")
    ps.add_argument("--factors", default=None,
                    help="Integer N or name:low:high,... (if no JSON)")
    ps.add_argument("--goal", default="screening",
                    choices=["screening", "optimization"])
    ps.add_argument("--budget", type=int, default=None)
    ps.add_argument("--responses", default="y")
    ps.add_argument("--n-region", type=int, default=4000)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--report", action="store_true",
                    help="Also write HTML under wave/reports/")
    ps.set_defaults(func=cmd_project_sync)

    pc = psub.add_parser("conclude", help="Write automatic-conclusions for a wave")
    pc.add_argument("--path", required=True, help="Wave directory")
    pc.add_argument("--lang", default="en", choices=["en", "es"])
    pc.add_argument("--report", action="store_true",
                    help="Also write HTML under wave/reports/")
    pc.add_argument("--json", action="store_true")
    pc.set_defaults(func=cmd_project_conclude)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
