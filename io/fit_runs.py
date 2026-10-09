"""Fit-run records under ``surface_pricer/output`` (plain JSON, no database).

Every ``fit`` run writes one directory::

    output/
      index.json                    # newest-first list of runs
      latest.json                   # {"run": "<directory name>"}
      MO_20260929_150000/
        surface.json                # EDSSabrSurface.to_dict()
        manifest.json               # metadata (underlying / date / spot / expiries ...)
        report.txt
        smile_*.png / term_structure.png
        overrides.json              # only when hand overrides were used

The layout is deliberately simple; :func:`record_fit_run`, :func:`list_runs`
and :func:`resolve_run` are the only entry points, so a database backend can
replace the files later without touching the callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.daycount import to_datetime

RUN_INDEX_NAME = "index.json"
LATEST_NAME = "latest.json"
MANIFEST_NAME = "manifest.json"
SURFACE_NAME = "surface.json"


def default_output_root() -> Path:
    """``surface_pricer/output`` (next to the package root)."""
    return Path(__file__).resolve().parent.parent / "output"


def fit_run_root(output_root=None, *, index=None) -> Path:
    """``<output_root>/vol_fit`` - and ``.../<bare index>`` when one is given.

    One output root, three folders: ``vol_fit/`` here, ``ir_curve/`` and
    ``borrow_curve/`` in :mod:`surface_pricer.io.curve_runs`.  Redirecting the root
    therefore redirects a whole test (or a whole book) in one flag.

    The runs of an index live in their own folder (2026-10): ``vol_fit/000852/``
    holds every 000852 run next to *its* ``index.json`` / ``latest.json``.  The
    index is then visible in the path - which is what a run's file name alone could
    never say - and "discard / compare one index's surface history" is a folder.
    Runs written before that, directly under ``vol_fit/``, are still read.
    """
    root = Path(output_root) if output_root else default_output_root()
    root = root / "vol_fit"
    key = bare_code(index) if index is not None else ""
    return root / key if key else root


@dataclass
class FitRun:
    """One stored fit run: a directory holding ``surface.json`` + ``manifest.json``."""

    name: str
    directory: Path
    surface_path: Path
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def underlying(self) -> str:
        """The **index** the run prices - the space the barriers and the spot live in."""
        return str(self.manifest.get("underlying", "") or "")

    @property
    def option_underlying(self) -> str:
        """The option venue the surface was fitted from (``MO`` for a 000852 run).

        A run is named by its index because that is what a contract references; the
        same index may be fitted from different venues (000905 from 510500's
        options).  Falls back to :attr:`underlying` for runs recorded before the two
        were split.
        """
        return str(self.manifest.get("option_underlying") or self.underlying)

    @property
    def valuation_datetime(self) -> Optional[str]:
        value = self.manifest.get("valuation_datetime")
        return str(value) if value else None

    @property
    def spot(self) -> Optional[float]:
        value = self.manifest.get("spot")
        return float(value) if value is not None else None

    @property
    def rate(self) -> float:
        return float(self.manifest.get("rate", 0.0) or 0.0)

    @property
    def borrow(self) -> Optional[float]:
        """The flat borrow the run recorded, if any.

        Usually ``None``: the fit prices its forwards off the borrow *curve* (whose
        run the manifest names), so there is no flat number to fall back to - and
        ``--borrow`` unset then means a zero flat borrow, not "the run's".
        """
        value = self.manifest.get("borrow")
        return float(value) if value is not None else None

    @property
    def expiries(self) -> List[str]:
        return list(self.manifest.get("expiries", []) or [])

    @property
    def created_at(self) -> str:
        return str(self.manifest.get("created_at", "") or "")

    def surface_payload(self) -> Dict[str, Any]:
        return json.loads(self.surface_path.read_text(encoding="utf-8"))

    def describe(self) -> str:
        parts = [self.name]
        if self.underlying:
            venue = self.option_underlying
            parts.append(
                self.underlying
                if not venue or venue == self.underlying
                else "{} ({})".format(self.underlying, venue)
            )
        if self.valuation_datetime:
            parts.append(str(self.valuation_datetime))
        if self.spot is not None:
            parts.append("spot={:.4f}".format(self.spot))
        if self.expiries:
            parts.append("{} expiries".format(len(self.expiries)))
        return " | ".join(parts)


# ------------------------------------------------------------------ writing
def record_fit_run(
    output_dir,
    *,
    surface,
    underlying: str,
    valuation_datetime,
    spot: float,
    settings=None,
    metrics: Optional[Dict[str, Any]] = None,
    overrides=None,
    rate: float = 0.0,
    calendar_name: Optional[str] = None,
    trading_days_per_year: Optional[float] = None,
    holiday_weight: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> FitRun:
    """Write ``surface.json`` + ``manifest.json`` and refresh the run index."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = surface.to_dict()
    (directory / SURFACE_NAME).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    manifest: Dict[str, Any] = {
        "name": directory.name,
        "underlying": str(underlying or "").upper(),
        "valuation_datetime": to_datetime(valuation_datetime).isoformat(sep=" "),
        "spot": float(spot),
        "rate": float(rate),
        "calendar": calendar_name,
        "trading_days_per_year": trading_days_per_year,
        "holiday_weight": holiday_weight,
        "created_at": datetime.now().isoformat(timespec="microseconds"),
        "surface_file": SURFACE_NAME,
        "expiries": [
            to_datetime(value).date().isoformat() for value in payload.get("expiry_dates", [])
        ],
        "n_expiries": len(payload.get("expiry_dates", [])),
        "settings": _settings_summary(settings),
        "hand_overrides": overrides.to_dict() if overrides is not None else None,
        "metrics": metrics or {},
        "files": [],
    }
    if extra:
        manifest.update(extra)
    manifest["files"] = sorted(
        item.name
        for item in directory.iterdir()
        if item.is_file() and item.name != MANIFEST_NAME
    ) + [MANIFEST_NAME]
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    run = FitRun(
        name=manifest["name"],
        directory=directory,
        surface_path=directory / SURFACE_NAME,
        manifest=manifest,
    )
    _refresh_index(directory.parent, run)
    return run


def _settings_summary(settings) -> Dict[str, Any]:
    if settings is None:
        return {}
    summary: Dict[str, Any] = {}
    for name in (
        "weight_mode",
        "forward_source",
        "max_iterations",
        "trading_days_per_year",
        "holiday_weight",
        "param_scaling_floor",
    ):
        if hasattr(settings, name):
            summary[name] = getattr(settings, name)
    overrides = getattr(settings, "override_config", None)
    if overrides is not None:
        summary["extend_synthetic_tenors"] = bool(overrides.extend_synthetic_tenors)
        summary["n_pinned_expiries"] = len(overrides.overrides)
    return summary


def _refresh_index(output_root: Path, run: FitRun) -> None:
    entries = [
        item for item in _read_index_at(output_root) if item.get("name") != run.name
    ]
    entries.append(_index_entry(run))
    entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    (output_root / RUN_INDEX_NAME).write_text(
        json.dumps({"runs": entries}, indent=2, default=str), encoding="utf-8"
    )
    # The pointer is per underlying: this run becomes "latest" for **its** index and
    # every other index keeps the pointer it had (2026-10).
    latest = _read_latest_map_at(output_root)
    latest[bare_code(run.underlying) or run.name] = run.name
    (output_root / LATEST_NAME).write_text(
        json.dumps(dict(sorted(latest.items())), indent=2), encoding="utf-8"
    )


def _index_entry(run: FitRun) -> Dict[str, Any]:
    return {
        "name": run.name,
        "underlying": run.underlying,
        "valuation_datetime": run.manifest.get("valuation_datetime"),
        "spot": run.manifest.get("spot"),
        "created_at": run.manifest.get("created_at"),
        "directory": run.name,
        "n_expiries": run.manifest.get("n_expiries"),
    }


# ------------------------------------------------------------------ reading
def bare_code(value: Any) -> str:
    """``000852.SH`` -> ``000852``; ``MO`` -> ``MO`` (case / suffix insensitive).

    The key ``latest.json`` is filed under: two spellings of the same index must
    land on the same entry (a run records ``000852.SH``, a payload may say either).
    """
    return str(value or "").strip().upper().split(".")[0]


def read_latest_map(output_root=None) -> Dict[str, str]:
    """``{bare index code: run name}`` - ``latest.json`` is **per underlying**.

    Two indices are two surfaces, so "the latest run" without saying which one is
    meaningless - it is how a 000852 quote once ended up priced on a 000300 surface.
    The old single-pointer shape (``{"run": ...}``) is still read: it can only mean
    the run's own underlying, which is what it maps to.

    This reads a **folder's** pointer; a run's own index folder
    (``vol_fit/000852/latest.json``, 2026-10) is what the app writes now - see
    :func:`latest_run_path`.
    """
    return _read_latest_map_at(fit_run_root(output_root))


def latest_run_path(output_root=None, underlying: Any = "") -> Optional[Path]:
    """The newest stored run directory for ``underlying`` (``None`` when unknown).

    Looks in the index's own folder first (``vol_fit/<index>/latest.json``), then at
    the flat pointer of runs written before the folders existed.  Nothing is
    guessed: an index that is in neither is simply unknown here - the callers
    (:func:`resolve_run`) turn that into the error that names what there is.
    """
    root = fit_run_root(output_root)
    keys = _underlying_keys(underlying)
    for key, directory in _latest_entries(root).items():
        if key in keys:
            return directory
    return None


def _pointer_folders(root: Path) -> List[Path]:
    """Folders that may hold a pointer: the index folders first, then the root."""
    folders: List[Path] = []
    if root.is_dir():
        folders.extend(
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and not _is_run_dir(child)
        )
    folders.append(root)
    return folders


def _latest_entries(root: Path) -> Dict[str, Path]:
    """``{bare index: run directory}`` from every ``latest.json`` under a runs root."""
    entries: Dict[str, Path] = {}
    for folder in _pointer_folders(root):
        for key, name in _read_latest_map_at(folder).items():
            entries.setdefault(key, folder / str(name))
    return entries


def _read_latest_map_at(root: Path) -> Dict[str, str]:
    """Read the pointer from a **runs directory** (already ``.../vol_fit``)."""
    path = root / LATEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    if len(payload) == 1 and "run" in payload:  # the pre-2026-10 shape
        name = str(payload.get("run") or "")
        manifest = root / name / MANIFEST_NAME
        if not name or not manifest.is_file():
            return {}
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        key = bare_code(data.get("underlying"))
        return {key: name} if key else {}
    return {str(key): str(value) for key, value in payload.items() if value}


def read_index(output_root=None) -> List[Dict[str, Any]]:
    """The run index, read from ``<output_root>/vol_fit/index.json``."""
    return _read_index_at(fit_run_root(output_root))


def _read_index_at(root: Path) -> List[Dict[str, Any]]:
    """Read ``index.json`` from a **runs directory** (already ``.../vol_fit``)."""
    path = root / RUN_INDEX_NAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("runs", [])
    return [item for item in payload if isinstance(item, dict)]


def list_runs(output_root=None) -> List[FitRun]:
    """Stored runs of every index, newest first (falls back to scanning folders)."""
    return _list_runs_at(fit_run_root(output_root))


def _list_runs_at(root: Path) -> List[FitRun]:
    """Every run under a runs root: its own (flat) plus one level of index folders."""
    runs = _runs_in(root)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and not _is_run_dir(child):
                runs.extend(_runs_in(child))
    runs.sort(key=lambda run: (str(run.created_at), run.name), reverse=True)
    return runs


def _is_run_dir(path: Path) -> bool:
    """A run directory is the one that holds a ``surface.json``."""
    return (path / SURFACE_NAME).is_file()


def _runs_in(root: Path) -> List[FitRun]:
    """The runs stored **directly** in one folder (index first, then a directory scan)."""
    runs: List[FitRun] = []
    for item in _read_index_at(root):
        directory = root / str(item.get("directory") or item.get("name") or "")
        if _is_run_dir(directory):
            runs.append(_load_run(directory))
    if runs:
        return runs
    if root.is_dir():
        for directory in sorted(root.iterdir(), reverse=True):
            if directory.is_dir() and _is_run_dir(directory):
                runs.append(_load_run(directory))
    return runs


def _underlying_keys(underlying: Any) -> List[str]:
    """The lookup keys for ``underlying``: one code, or several spellings in order.

    Both a payload's index (``000852.SH``) and a block's option venue (``MO``) can
    name the same run, and a run recorded before the two were split is filed under
    whichever the caller gave - so a caller may hand over both, most specific first.
    """
    values = [underlying] if isinstance(underlying, str) else list(underlying or ())
    keys: List[str] = []
    for value in values:
        key = bare_code(value)
        if key and key not in keys:
            keys.append(key)
    return keys


def resolve_run(
    spec: str = "latest", output_root=None, underlying: Any = ""
) -> FitRun:
    """Resolve ``latest`` / a run name (unique prefix ok) / a directory / a file.

    ``output_root`` is the *output root* - the run folders live in its ``vol_fit/``
    (see :func:`fit_run_root`).  ``latest`` reads ``latest.json``, which is filed
    **per underlying** (2026-10): ``underlying`` picks the entry, so a 000852 quote
    can never silently move onto a 000300 surface (nor onto a run of another index
    that happens to be newer).  A missing pointer or a missing index is an error,
    never "the first run in the folder".
    """
    root = fit_run_root(output_root)
    text = str(spec if spec is not None else "latest").strip()

    path = Path(text)
    if path.exists():
        if path.is_file():
            if path.name == SURFACE_NAME:
                return _run_from_surface_path(path)
            raise ValueError("{} is not a surface.json file".format(path))
        if (path / SURFACE_NAME).is_file():
            return _load_run(path)
        raise ValueError("{} does not contain a {}".format(path, SURFACE_NAME))

    lowered = text.lower()
    if lowered in {"", "latest", "last"}:
        entries = _latest_entries(root)
        keys = _underlying_keys(underlying)
        target = next((entries[key] for key in keys if key in entries), None)
        if target is None and keys:
            # No pointer for this index - but the **folders** still know: "latest for
            # 000852" is the newest stored run of 000852, in its own folder, which is
            # a question about one index only (never a cross-index guess).  This is
            # what makes an index folder whose pointer is missing - or one written
            # before the folders existed - still work.
            for key in keys:
                runs = _runs_in(root / key)
                if runs:
                    target = runs[0].directory
                    break
            if target is None:
                for candidate in _list_runs_at(root):
                    if bare_code(candidate.underlying) in keys:
                        target = candidate.directory
                        break
        if keys and target is None:
            raise ValueError(
                "no fit run for {} under {}: it knows {}.  Run "
                "'python -m surface_pricer fit --underlying ... --index ...' "
                "for it first, or name a run directory".format(
                    " / ".join(keys),
                    root,
                    ", ".join(sorted(entries)) or "(nothing)",
                )
            )
        if not keys:
            if not entries:
                raise ValueError(
                    "no fit run registered under {} ({} is missing); run "
                    "'python -m surface_pricer fit' first, or name a run directory".format(
                        root, LATEST_NAME
                    )
                )
            if len(entries) == 1:
                # nothing to go on and one candidate: that is what "latest" means
                target = next(iter(entries.values()))
            else:
                raise ValueError(
                    "{} lists several underlyings ({}): say which one - pass the "
                    "payload's underlying, or name the run".format(
                        root / LATEST_NAME, ", ".join(sorted(entries))
                    )
                )
        if target is None or not _is_run_dir(target):
            raise ValueError(
                "latest points at {!r}, which is not a stored run under {} - run "
                "'python -m surface_pricer fit' first, or name a run directory".format(
                    target, root
                )
            )
        return _load_run(target)

    candidate = root / text
    if candidate.is_dir() and (candidate / SURFACE_NAME).is_file():
        return _load_run(candidate)

    runs = _list_runs_at(root)
    for run in runs:
        if run.name.lower() == lowered:
            return run
    matches = [run for run in runs if run.name.lower().startswith(lowered)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            "ambiguous fit run {!r}; matches {}".format(spec, [run.name for run in matches])
        )
    raise ValueError(
        "unknown fit run {!r}; available: {}".format(spec, [run.name for run in runs] or "(none)")
    )


def _load_run(directory: Path) -> FitRun:
    manifest: Dict[str, Any] = {}
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("name", directory.name)
    return FitRun(
        name=str(manifest["name"]),
        directory=directory,
        surface_path=directory / SURFACE_NAME,
        manifest=manifest,
    )


def _run_from_surface_path(surface_path: Path) -> FitRun:
    directory = surface_path.parent
    if (directory / MANIFEST_NAME).is_file():
        return _load_run(directory)
    payload = json.loads(surface_path.read_text(encoding="utf-8"))
    return FitRun(
        name=directory.name,
        directory=directory,
        surface_path=surface_path,
        manifest={
            "name": directory.name,
            "valuation_datetime": payload.get("init_date"),
            "spot": payload.get("init_spot"),
            "calendar": payload.get("calendar"),
            "expiries": [
                to_datetime(value).date().isoformat()
                for value in payload.get("expiry_dates", [])
            ],
        },
    )


__all__ = [
    "FitRun",
    "LATEST_NAME",
    "MANIFEST_NAME",
    "RUN_INDEX_NAME",
    "SURFACE_NAME",
    "bare_code",
    "default_output_root",
    "fit_run_root",
    "latest_run_path",
    "list_runs",
    "read_index",
    "read_latest_map",
    "record_fit_run",
    "resolve_run",
]
