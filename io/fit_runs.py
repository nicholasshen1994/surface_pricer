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


@dataclass
class FitRun:
    """One stored fit run: a directory holding ``surface.json`` + ``manifest.json``."""

    name: str
    directory: Path
    surface_path: Path
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def underlying(self) -> str:
        return str(self.manifest.get("underlying", "") or "")

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
            parts.append(self.underlying)
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
    entries = [item for item in read_index(output_root) if item.get("name") != run.name]
    entries.append(_index_entry(run))
    entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    (output_root / RUN_INDEX_NAME).write_text(
        json.dumps({"runs": entries}, indent=2, default=str), encoding="utf-8"
    )
    (output_root / LATEST_NAME).write_text(
        json.dumps({"run": run.name}, indent=2), encoding="utf-8"
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
def read_index(output_root=None) -> List[Dict[str, Any]]:
    root = Path(output_root) if output_root else default_output_root()
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
    """Stored runs, newest first (falls back to scanning the directory)."""
    root = Path(output_root) if output_root else default_output_root()
    runs: List[FitRun] = []
    for item in read_index(root):
        directory = root / str(item.get("directory") or item.get("name") or "")
        if (directory / SURFACE_NAME).is_file():
            runs.append(_load_run(directory))
    if runs:
        return runs
    if root.is_dir():
        for directory in sorted(root.iterdir(), reverse=True):
            if directory.is_dir() and (directory / SURFACE_NAME).is_file():
                runs.append(_load_run(directory))
    return runs


def resolve_run(spec: str = "latest", output_root=None) -> FitRun:
    """Resolve ``latest`` / a run name (unique prefix ok) / a directory / a file."""
    root = Path(output_root) if output_root else default_output_root()
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
        latest_path = root / LATEST_NAME
        if latest_path.is_file():
            try:
                name = json.loads(latest_path.read_text(encoding="utf-8")).get("run")
            except json.JSONDecodeError:
                name = None
            if name and (root / str(name) / SURFACE_NAME).is_file():
                return _load_run(root / str(name))
        runs = list_runs(root)
        if runs:
            return runs[0]
        raise ValueError(
            "no fit run found under {}; run 'python -m surface_pricer fit' first".format(root)
        )

    candidate = root / text
    if candidate.is_dir() and (candidate / SURFACE_NAME).is_file():
        return _load_run(candidate)

    runs = list_runs(root)
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
    "default_output_root",
    "list_runs",
    "read_index",
    "record_fit_run",
    "resolve_run",
]
