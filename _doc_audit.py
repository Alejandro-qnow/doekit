"""Temporary docstring audit script."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

root = Path(r"d:\projects\doekit\doekit")

NUMPY_SECTIONS = (
    "Parameters",
    "Returns",
    "Raises",
    "Notes",
    "Examples",
    "Formulas",
    "Attributes",
    "See Also",
    "Yields",
    "Warns",
    "References",
)
GOOGLE_MARKERS = re.compile(r"^\s*Args\s*:", re.M)


def is_public(name: str) -> bool:
    return not name.startswith("_")


def classify_doc(doc: str | None):
    if doc is None or not doc.strip():
        return "missing", [], False, 0, ""
    lines = [ln.rstrip() for ln in doc.strip().splitlines()]
    body = "\n".join(lines)
    has_google = bool(GOOGLE_MARKERS.search(body))
    sections = []
    for s in NUMPY_SECTIONS:
        if re.search(rf"(?m)^{s}\s*\n\s*-{{3,}}", body):
            sections.append(s)
    nlines = len([ln for ln in lines if ln.strip()])
    if sections:
        kind = "full_numpy"
    elif nlines <= 2:
        kind = "one_line"
    else:
        kind = "brief"
    first = lines[0][:100] if lines else ""
    return kind, sections, has_google, nlines, first


results = []
for path in sorted(root.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue
    rel = str(path.relative_to(root)).replace("\\", "/")
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception as e:
        results.append({"file": rel, "error": str(e)})
        continue
    mod_doc = ast.get_docstring(tree)
    items = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not is_public(node.name):
                continue
            kind, sections, google, nlines, first = classify_doc(ast.get_docstring(node))
            items.append(
                {
                    "kind": "func",
                    "name": node.name,
                    "style": kind,
                    "sections": sections,
                    "google": google,
                    "nlines": nlines,
                    "first": first,
                }
            )
        elif isinstance(node, ast.ClassDef):
            if not is_public(node.name):
                continue
            kind, sections, google, nlines, first = classify_doc(ast.get_docstring(node))
            items.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "style": kind,
                    "sections": sections,
                    "google": google,
                    "nlines": nlines,
                    "first": first,
                }
            )
            for sub in node.body:
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                mname = sub.name
                if mname.startswith("_") and mname not in ("__init__", "__call__"):
                    continue
                if mname.startswith("__") and mname not in ("__init__", "__call__"):
                    continue
                mkind, msections, mgoogle, mnlines, mfirst = classify_doc(
                    ast.get_docstring(sub)
                )
                items.append(
                    {
                        "kind": "method",
                        "name": f"{node.name}.{mname}",
                        "style": mkind,
                        "sections": msections,
                        "google": mgoogle,
                        "nlines": mnlines,
                        "first": mfirst,
                    }
                )
    results.append(
        {
            "file": rel,
            "module_doc": bool(mod_doc and mod_doc.strip()),
            "module_doc_first": (mod_doc.strip().splitlines()[0][:100] if mod_doc else ""),
            "items": items,
        }
    )

out = Path(r"d:\projects\doekit\_doc_audit.json")
out.write_text(json.dumps(results, indent=2), encoding="utf-8")

# human summary
for entry in results:
    items = entry.get("items") or []
    if not items and not entry.get("module_doc"):
        continue
    counts = {"full_numpy": 0, "brief": 0, "one_line": 0, "missing": 0}
    for it in items:
        counts[it["style"]] = counts.get(it["style"], 0) + 1
    has_f = any("Formulas" in it["sections"] for it in items)
    has_e = any("Examples" in it["sections"] for it in items)
    has_g = any(it["google"] for it in items)
    print(
        f"=== {entry['file']} | n={len(items)} "
        f"full={counts['full_numpy']} brief={counts['brief']} "
        f"one={counts['one_line']} miss={counts['missing']} "
        f"formulas={has_f} examples={has_e} google={has_g} "
        f"moddoc={entry['module_doc']}"
    )
    for it in items:
        sec = ",".join(it["sections"]) if it["sections"] else "-"
        g = " GOOGLE" if it["google"] else ""
        print(
            f"  {it['kind']:6} {it['name']:50} {it['style']:10} [{sec}]{g} | {it['first']!r}"
        )
