"""Minimal CLI: ``doekit recommend|evaluate|experiment``."""

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


def main(argv=None) -> int:
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

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
