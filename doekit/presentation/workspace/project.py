"""ExperimentProject and Wave: traceable on-disk experiment packages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd

from ..._version import __version__
from ...shared.serialize import jsonify
from ..export import export_csv
from ..io import write_text
from .conclusions import (
    DEFAULT_THRESHOLDS,
    build_conclusions,
    conclusions_to_markdown,
)
from .paths import (
    WAVE_STATUSES,
    ensure_wave_layout,
    project_dirname,
    slugify,
    wave_dirname,
)

PathLike = Union[str, Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonify(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Wave:
    """One DoE cycle on disk (design → lab → ingest → analyze → conclude)."""

    def __init__(self, path: PathLike, project: Optional["ExperimentProject"] = None):
        self.path = Path(path).resolve()
        self.project = project
        if not (self.path / "manifest.json").exists():
            raise FileNotFoundError(f"not a wave directory (missing manifest.json): {self.path}")

    @property
    def wave_id(self) -> str:
        return self.path.name

    @property
    def manifest(self) -> dict:
        return _read_json(self.path / "manifest.json")

    def _write_manifest(self, data: dict) -> None:
        _write_json(self.path / "manifest.json", data)

    def _update_manifest(self, **fields) -> dict:
        m = self.manifest
        m.update(fields)
        m["updated_at"] = _utc_now()
        self._write_manifest(m)
        return m

    def _rel(self, *parts: str) -> str:
        return "/".join(parts)

    def _checksums(self, paths: Sequence[Path]) -> dict:
        out = {}
        for p in paths:
            p = Path(p)
            if p.is_file():
                rel = p.relative_to(self.path).as_posix()
                digest = _file_sha256(p)
                if digest:
                    out[rel] = digest
        return out

    def load_experiment(self):
        """Load :class:`~doekit.orchestration.experiment.Experiment` from this wave."""
        from ...orchestration.experiment import Experiment  # noqa: PLC0415

        cfg = self.path / "doe-configuration" / "experiment.json"
        if not cfg.exists():
            raise FileNotFoundError(f"missing experiment snapshot: {cfg}")
        return Experiment.from_dict(_read_json(cfg))

    def ingest_from(self, source: Union[PathLike, pd.DataFrame, Mapping, Any]):
        """Read responses from CSV/DataFrame and sync into the wave + experiment."""
        exp = self.load_experiment()
        if isinstance(source, (str, Path)):
            frame = pd.read_csv(source)
            # drop run_id if present; keep response columns
            cols = [c for c in frame.columns if c != "run_id"]
            # prefer known response names, else non-factor columns
            factor_cols = set(exp.design.matrix.columns)
            resp_cols = [c for c in cols if c not in factor_cols]
            if not resp_cols:
                resp_cols = list(exp.response_names)
            frame = frame[resp_cols]
        else:
            frame = source
        exp.ingest(frame)
        if exp.evaluation is None:
            exp.evaluate()
        self.sync(exp)
        return exp

    def sync(
        self,
        exp,
        *,
        write_report: bool = False,
        comparison: Optional[Any] = None,
        next_runs: Optional[Any] = None,
        thresholds: Optional[Mapping[str, float]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Write experiment state into the wave (INPUT + available OUTPUT)."""
        thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        ensure_wave_layout(self.path)
        cfg = self.path / "doe-configuration"
        data = self.path / "data"
        results = self.path / "results"
        meta = self.path / "metadata"

        snap = exp.to_dict()
        _write_json(cfg / "experiment.json", snap)
        _write_json(cfg / "design.json", exp.design.to_dict())
        if exp.model is not None:
            _write_json(cfg / "model.json", exp.model.to_dict())
        _write_json(cfg / "thresholds.json", thr)

        export_csv(exp.design, data / "run_sheet.csv",
                   response_names=exp.response_names)

        inputs = [
            self._rel("doe-configuration", "experiment.json"),
            self._rel("doe-configuration", "design.json"),
            self._rel("doe-configuration", "thresholds.json"),
            self._rel("data", "run_sheet.csv"),
        ]
        outputs: list[str] = []

        if exp.responses is not None:
            exp.responses.to_csv(data / "responses.csv", index=False)
            outputs.append(self._rel("data", "responses.csv"))

        if exp.evaluation is not None:
            _write_json(results / "evaluation.json", exp.evaluation.to_dict())
            outputs.append(self._rel("results", "evaluation.json"))

        if exp.fit is not None:
            _write_json(results / "fit.json", exp.fit.to_dict())
            outputs.append(self._rel("results", "fit.json"))
        if exp.fits:
            _write_json(
                results / "fits.json",
                {k: v.to_dict() for k, v in exp.fits.items()},
            )
            outputs.append(self._rel("results", "fits.json"))

        cmp_dict = None
        if comparison is not None:
            cmp_dict = comparison.to_dict() if hasattr(comparison, "to_dict") else dict(comparison)
            _write_json(results / "comparison.json", cmp_dict)
            outputs.append(self._rel("results", "comparison.json"))

        if next_runs is not None:
            nxt = next_runs.to_dict() if hasattr(next_runs, "to_dict") else dict(next_runs)
            _write_json(results / "next_runs.json", nxt)
            outputs.append(self._rel("results", "next_runs.json"))
            if cmp_dict is None and hasattr(next_runs, "comparison"):
                cmp_dict = next_runs.comparison.to_dict()
                _write_json(results / "comparison.json", cmp_dict)
                outputs.append(self._rel("results", "comparison.json"))

        if write_report:
            from ..report import report_html  # noqa: PLC0415
            report_html(
                exp.design,
                response=exp.response,
                model=exp.model,
                output_dir=str(self.path / "reports"),
                thresholds=thr,
                open_browser=False,
            )
            outputs.append(self._rel("reports", "index.html"))

        # status machine
        if exp.responses is not None and exp.evaluation is not None:
            status = "analyzed"
        elif exp.evaluation is not None:
            status = "awaiting_response"
        else:
            status = "planned"
        if status not in WAVE_STATUSES:
            raise ValueError(f"invalid status {status!r}")

        prov = {
            "schema": "doekit.WaveProvenance/1",
            "wave_id": self.wave_id,
            "doekit_version": __version__,
            "synced_at": _utc_now(),
            "seed": seed,
            "design_kind": exp.design.metadata.get("kind"),
            "n_runs": int(exp.design.n_runs),
            "response_names": list(exp.response_names),
            "parent_wave": self.manifest.get("parent_wave"),
        }
        if self.project is not None:
            prov["project"] = self.project.name
            prov["project_slug"] = self.project.slug
        _write_json(meta / "provenance.json", prov)

        tracked = [
            cfg / "experiment.json",
            cfg / "design.json",
            cfg / "thresholds.json",
            data / "run_sheet.csv",
            data / "responses.csv",
            results / "evaluation.json",
            results / "fit.json",
            results / "next_runs.json",
            results / "comparison.json",
        ]
        checksums = self._checksums(tracked)
        _write_json(meta / "checksums.json", {
            "schema": "doekit.WaveChecksums/1",
            "files": checksums,
            "updated_at": _utc_now(),
        })

        # preserve concluded status if conclusions already present and still analyzed
        prev = self.manifest.get("status")
        if prev == "concluded" and (self.path / "automatic-conclusions" / "conclusions.json").exists():
            status = "concluded"

        return self._update_manifest(
            status=status,
            inputs=inputs,
            outputs=sorted(set(outputs)),
            thresholds=thr,
        )

    def conclude(
        self,
        exp=None,
        *,
        thresholds: Optional[Mapping[str, float]] = None,
        lang: str = "en",
        write_html: bool = False,
        comparison: Optional[Any] = None,
    ) -> dict:
        """Generate ``automatic-conclusions/`` from thresholds + computed facts."""
        if exp is None:
            exp = self.load_experiment()
        thr = {**DEFAULT_THRESHOLDS, **(thresholds or self.manifest.get("thresholds") or {})}

        if exp.evaluation is None:
            exp.evaluate()

        cmp_dict = None
        if comparison is not None:
            cmp_dict = comparison.to_dict() if hasattr(comparison, "to_dict") else dict(comparison)
        else:
            cmp_path = self.path / "results" / "comparison.json"
            if cmp_path.exists():
                cmp_dict = _read_json(cmp_path)

        # Ensure disk reflects latest experiment before concluding
        self.sync(exp, write_report=write_html, comparison=comparison, thresholds=thr)

        project_name = ""
        if self.project is not None:
            project_name = self.project.name
        elif (self.path.parent.parent / "PROJECT.json").exists():
            project_name = _read_json(self.path.parent.parent / "PROJECT.json").get("name", "")

        conclusions = build_conclusions(
            exp.design,
            response=exp.response,
            model=exp.model,
            evaluation=exp.evaluation,
            fit=exp.fit,
            thresholds=thr,
            comparison=cmp_dict,
            lang=lang,
            wave_id=self.wave_id,
            project_name=project_name,
        )
        out_dir = self.path / "automatic-conclusions"
        _write_json(out_dir / "conclusions.json", conclusions)
        write_text(out_dir / "conclusions.md", conclusions_to_markdown(conclusions))

        m = self.manifest
        outputs = list(m.get("outputs") or [])
        for rel in (
            self._rel("automatic-conclusions", "conclusions.json"),
            self._rel("automatic-conclusions", "conclusions.md"),
        ):
            if rel not in outputs:
                outputs.append(rel)
        self._update_manifest(status="concluded", outputs=outputs, thresholds=thr)
        return conclusions


class ExperimentProject:
    """On-disk research project containing sequential DoE waves."""

    def __init__(self, path: PathLike):
        self.path = Path(path).resolve()
        meta = self.path / "PROJECT.json"
        if not meta.exists():
            raise FileNotFoundError(f"not an experiment project (missing PROJECT.json): {self.path}")
        self._meta = _read_json(meta)

    @property
    def name(self) -> str:
        return self._meta["name"]

    @property
    def slug(self) -> str:
        return self._meta["slug"]

    @classmethod
    def create(
        cls,
        name: str,
        root: PathLike = "experiments",
        *,
        description: str = "",
    ) -> "ExperimentProject":
        """Create ``<root>/experiment_project_<slug>/`` with PROJECT.json."""
        root = Path(root)
        dirname = project_dirname(name)
        path = root / dirname
        if path.exists() and (path / "PROJECT.json").exists():
            return cls(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "waves").mkdir(exist_ok=True)
        slug = slugify(name)
        meta = {
            "schema": "doekit.ExperimentProject/1",
            "name": name,
            "slug": slug,
            "dirname": dirname,
            "created_at": _utc_now(),
            "doekit_version": __version__,
            "description": description,
        }
        _write_json(path / "PROJECT.json", meta)
        readme = (
            f"# {name}\n\n"
            f"doekit experiment project (`{slug}`).\n\n"
            f"Waves live under `waves/wave_NNN/`.\n"
            f"Lab row ids remain the `run_id` column inside each wave's "
            f"`data/run_sheet.csv`.\n"
        )
        write_text(path / "README.md", readme)
        return cls(path)

    @classmethod
    def open(cls, path: PathLike) -> "ExperimentProject":
        """Open an existing project directory."""
        return cls(path)

    def _next_wave_index(self) -> int:
        waves_dir = self.path / "waves"
        if not waves_dir.exists():
            return 1
        idxs = []
        for p in waves_dir.iterdir():
            if p.is_dir() and p.name.startswith("wave_"):
                try:
                    idxs.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    continue
        return (max(idxs) + 1) if idxs else 1

    def waves(self) -> list[Wave]:
        """List waves sorted by id."""
        waves_dir = self.path / "waves"
        if not waves_dir.exists():
            return []
        paths = sorted(
            p for p in waves_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )
        return [Wave(p, project=self) for p in paths]

    def latest_wave(self) -> Optional[Wave]:
        ws = self.waves()
        return ws[-1] if ws else None

    def get_wave(self, wave_id: str) -> Wave:
        path = self.path / "waves" / wave_id
        return Wave(path, project=self)

    def new_wave(
        self,
        exp,
        *,
        parent_wave: Optional[str] = None,
        thresholds: Optional[Mapping[str, float]] = None,
        seed: Optional[int] = None,
    ) -> Wave:
        """Create the next wave and write INPUT artifacts from ``exp``."""
        thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        idx = self._next_wave_index()
        wave_id = wave_dirname(idx)
        wave_path = self.path / "waves" / wave_id
        ensure_wave_layout(wave_path)

        if parent_wave is None:
            latest = self.latest_wave()
            if latest is not None:
                parent_wave = latest.wave_id

        manifest = {
            "schema": "doekit.WaveManifest/1",
            "wave_id": wave_id,
            "status": "planned",
            "parent_wave": parent_wave,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "inputs": [],
            "outputs": [],
            "thresholds": thr,
            "doekit_version": __version__,
        }
        _write_json(wave_path / "manifest.json", manifest)
        wave = Wave(wave_path, project=self)
        wave.sync(exp, thresholds=thr, seed=seed)
        self._refresh_readme()
        return wave

    def _refresh_readme(self) -> None:
        lines = [
            f"# {self.name}",
            "",
            f"doekit experiment project (`{self.slug}`).",
            "",
            "## Waves",
            "",
        ]
        for w in self.waves():
            m = w.manifest
            lines.append(f"- `{w.wave_id}` — status `{m.get('status')}`")
        lines.append("")
        write_text(self.path / "README.md", "\n".join(lines))


def open_project(path: PathLike) -> ExperimentProject:
    """Open an existing experiment project."""
    return ExperimentProject.open(path)


def project(name: str, root: PathLike = "experiments", **kwargs) -> ExperimentProject:
    """Create or open ``experiments/experiment_project_<slug>/``."""
    root = Path(root)
    path = root / project_dirname(name)
    if (path / "PROJECT.json").exists():
        return ExperimentProject.open(path)
    return ExperimentProject.create(name, root=root, **kwargs)
