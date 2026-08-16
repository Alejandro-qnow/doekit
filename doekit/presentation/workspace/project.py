"""Traceable on-disk experiment projects and waves.

An :class:`ExperimentProject` groups sequential DoE cycles under
``experiments/experiment_project_<slug>/``. Each :class:`Wave` is one cycle
(design → lab → ingest → analyze → conclude) with a fixed directory layout,
manifest, and checksums for reproducibility.
"""

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
    """One DoE cycle persisted on disk.

    A wave directory contains ``manifest.json``, ``doe-configuration/``,
    ``data/``, ``results/``, and optional ``automatic-conclusions/``. Status
    progresses through ``planned`` → ``awaiting_response`` → ``analyzed`` →
    ``concluded``.

    Attributes
    ----------
    path : Path
        Absolute path to the wave root (e.g. ``waves/wave_001/``).
    project : ExperimentProject or None
        Parent project, when opened via :meth:`ExperimentProject.get_wave`.
    """

    def __init__(self, path: PathLike, project: Optional["ExperimentProject"] = None):
        """Open an existing wave directory.

        Parameters
        ----------
        path : str or Path
            Wave root containing ``manifest.json``.
        project : ExperimentProject, optional
            Parent project for provenance metadata.

        Raises
        ------
        FileNotFoundError
            When ``manifest.json`` is missing under ``path``.
        """
        self.path = Path(path).resolve()
        self.project = project
        if not (self.path / "manifest.json").exists():
            raise FileNotFoundError(f"not a wave directory (missing manifest.json): {self.path}")

    @property
    def wave_id(self) -> str:
        """Wave folder name (e.g. ``wave_001``)."""
        return self.path.name

    @property
    def manifest(self) -> dict:
        """Parsed ``manifest.json`` for this wave."""
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
        """Load the experiment snapshot from ``doe-configuration/experiment.json``.

        Returns
        -------
        Experiment
            Reconstructed from the on-disk JSON snapshot.

        Raises
        ------
        FileNotFoundError
            When ``doe-configuration/experiment.json`` is absent.
        """
        from ...orchestration.experiment import Experiment  # noqa: PLC0415

        cfg = self.path / "doe-configuration" / "experiment.json"
        if not cfg.exists():
            raise FileNotFoundError(f"missing experiment snapshot: {cfg}")
        return Experiment.from_dict(_read_json(cfg))

    def ingest_from(self, source: Union[PathLike, pd.DataFrame, Mapping, Any]):
        """Ingest lab responses and sync the wave.

        Reads a CSV path or in-memory table, attaches responses to the loaded
        :class:`~doekit.orchestration.experiment.Experiment`, evaluates if
        needed, and writes artifacts via :meth:`sync`.

        Parameters
        ----------
        source : str, Path, DataFrame, or mapping
            CSV file path or tabular responses. Non-factor columns become
            response variables; ``run_id`` is ignored when present.

        Returns
        -------
        Experiment
            Updated experiment with responses and evaluation.
        """
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
        """Persist experiment state into the wave directory.

        Writes configuration, run sheet, responses, evaluation, fit, comparison,
        and optional HTML report. Updates ``manifest.json`` status, inputs,
        outputs, and ``metadata/checksums.json``.

        Parameters
        ----------
        exp : Experiment
            Experiment whose state is serialized to disk.
        write_report : bool, default False
            When True, render ``reports/index.html``.
        comparison : mapping or object, optional
            Comparison result (``to_dict()`` or plain dict) for
            ``results/comparison.json``.
        next_runs : mapping or object, optional
            Next-run proposal for ``results/next_runs.json``.
        thresholds : mapping of str to float, optional
            Quality gates merged with :data:`~doekit.presentation.workspace.conclusions.DEFAULT_THRESHOLDS`.
        seed : int, optional
            Recorded in ``metadata/provenance.json``.

        Returns
        -------
        dict
            Updated manifest (``status``, ``inputs``, ``outputs``, ``thresholds``).

        Raises
        ------
        ValueError
            When the derived status is not in :data:`~doekit.presentation.workspace.paths.WAVE_STATUSES`.
        """
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
        """Generate automatic conclusions for this wave.

        Builds ``doekit.AutomaticConclusions/1`` from computed facts and
        threshold gates, writes ``automatic-conclusions/conclusions.json`` and
        ``conclusions.md``, and sets manifest status to ``concluded``.

        Parameters
        ----------
        exp : Experiment, optional
            Experiment to conclude; loaded from disk when omitted.
        thresholds : mapping of str to float, optional
            Gate thresholds; manifest or defaults apply when omitted.
        lang : str, default ``"en"``
            Language for narrative strings (``"en"`` or ``"es"``).
        write_html : bool, default False
            When True, sync an HTML report before concluding.
        comparison : mapping or object, optional
            Comparison facts for the process gate; read from disk when omitted.

        Returns
        -------
        dict
            ``doekit.AutomaticConclusions/1`` payload (JSON-safe).
        """
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
    """On-disk research project containing sequential DoE waves.

    Layout: ``PROJECT.json``, ``README.md``, and ``waves/wave_NNN/`` children.
    Use :meth:`create` / :meth:`open` or the module helpers :func:`project` and
    :func:`open_project`.

    Attributes
    ----------
    path : Path
        Absolute path to the project root.
    """

    def __init__(self, path: PathLike):
        """Open an existing project directory.

        Parameters
        ----------
        path : str or Path
            Project root containing ``PROJECT.json``.

        Raises
        ------
        FileNotFoundError
            When ``PROJECT.json`` is missing.
        """
        self.path = Path(path).resolve()
        meta = self.path / "PROJECT.json"
        if not meta.exists():
            raise FileNotFoundError(f"not an experiment project (missing PROJECT.json): {self.path}")
        self._meta = _read_json(meta)

    @property
    def name(self) -> str:
        """Human-readable project name from ``PROJECT.json``."""
        return self._meta["name"]

    @property
    def slug(self) -> str:
        """Filesystem slug derived from :attr:`name`."""
        return self._meta["slug"]

    @classmethod
    def create(
        cls,
        name: str,
        root: PathLike = "experiments",
        *,
        description: str = "",
    ) -> "ExperimentProject":
        """Create a new experiment project on disk.

        Creates ``<root>/experiment_project_<slug>/`` with ``PROJECT.json``,
        ``README.md``, and an empty ``waves/`` directory. Returns the existing
        project unchanged when the directory already exists.

        Parameters
        ----------
        name : str
            Human-readable project name (must contain alphanumeric characters).
        root : str or Path, default ``"experiments"``
            Parent directory for the project folder.
        description : str, default ``""``
            Free-text description stored in ``PROJECT.json``.

        Returns
        -------
        ExperimentProject
            Handle to the created or existing project.

        Examples
        --------
        >>> import doekit as ed
        >>> proj = ed.ExperimentProject.create("Screening A", root="experiments")
        >>> proj.slug
        'screening-a'
        """
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
        """Open an existing project directory.

        Parameters
        ----------
        path : str or Path
            Project root containing ``PROJECT.json``.

        Returns
        -------
        ExperimentProject
        """
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
        """List waves sorted by directory name.

        Returns
        -------
        list of Wave
            One handle per ``waves/wave_NNN/`` with a valid ``manifest.json``.
        """
        waves_dir = self.path / "waves"
        if not waves_dir.exists():
            return []
        paths = sorted(
            p for p in waves_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )
        return [Wave(p, project=self) for p in paths]

    def latest_wave(self) -> Optional[Wave]:
        """Most recent wave, or ``None`` when the project has no waves."""
        ws = self.waves()
        return ws[-1] if ws else None

    def get_wave(self, wave_id: str) -> Wave:
        """Open a wave by id (e.g. ``"wave_001"``).

        Parameters
        ----------
        wave_id : str
            Wave folder name under ``waves/``.

        Returns
        -------
        Wave
        """
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
        """Create the next wave and write INPUT artifacts from ``exp``.

        Allocates ``waves/wave_NNN/``, initializes ``manifest.json``, and calls
        :meth:`Wave.sync`. Links ``parent_wave`` to the latest wave when omitted.

        Parameters
        ----------
        exp : Experiment
            Experiment whose design and configuration seed the new wave.
        parent_wave : str, optional
            Prior wave id for lineage; defaults to the latest wave.
        thresholds : mapping of str to float, optional
            Quality gates stored in the manifest and sync artifacts.
        seed : int, optional
            Recorded in wave provenance metadata.

        Returns
        -------
        Wave
            Handle to the newly created wave directory.
        """
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
    """Open an existing experiment project.

    Alias for :meth:`ExperimentProject.open`.

    Parameters
    ----------
    path : str or Path
        Project root containing ``PROJECT.json``.

    Returns
    -------
    ExperimentProject
    """
    return ExperimentProject.open(path)


def project(name: str, root: PathLike = "experiments", **kwargs) -> ExperimentProject:
    """Create or open ``experiments/experiment_project_<slug>/``.

    Opens the project when ``PROJECT.json`` already exists; otherwise creates
    it via :meth:`ExperimentProject.create`.

    Parameters
    ----------
    name : str
        Human-readable project name.
    root : str or Path, default ``"experiments"``
        Parent directory for the project folder.
    **kwargs
        Forwarded to :meth:`ExperimentProject.create` (e.g. ``description``).

    Returns
    -------
    ExperimentProject

    Examples
    --------
    >>> import doekit as ed
    >>> proj = ed.project("My study")
    >>> proj.name
    'My study'
    """
    root = Path(root)
    path = root / project_dirname(name)
    if (path / "PROJECT.json").exists():
        return ExperimentProject.open(path)
    return ExperimentProject.create(name, root=root, **kwargs)
