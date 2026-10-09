"""Timestamped curve runs: every build writes a **new** file, ``latest`` picks one.

Same convention as the fit runs (:mod:`surface_pricer.io.fit_runs`), one folder
per curve kind under the output root - and for a per-index kind one folder per
**index** inside it::

    output/ir_curve/ir_curve_20261008_101541.json      # one CNY curve for everything
    output/ir_curve/latest.json                        # {"ir_curve": "..."}
    output/borrow_curve/000852/borrow_curve_20261008_101542.json
    output/borrow_curve/000852/latest.json             # {"borrow_curve": "..."}
    output/borrow_curve/510500/borrow_curve_20261009_101758.json

The **borrow** curve is per index (2026-10): two indices are two borrow curves -
one is implied from 000852's futures and options, another from 510500's own chain
- so each index gets its own folder, ``latest.json`` inside it names one run, and
``--borrow-curve latest`` is resolved **against the index being priced**
(:func:`resolve_curve_path`'s ``index``).  The index is then visible in the path,
which is what a file name like ``borrow_curve_20261009_101758.json`` could never
say.  An index with no folder is an error naming the ones there are, never another
index's curve.  The interest-rate curve is one CNY curve for everything, so it
stays a flat folder (``index`` is ignored for it), and borrow runs written before
the folders existed are still found through the flat pointer's map
(``{"000852": ..., "510500": ...}``) or its single name.

A build never overwrites what a valuation already used, and a valuation says
``latest`` (the newest for that index) or names one file - nothing is searched
for: those two spellings plus ``none`` (the flat ``--rate`` / ``--borrow``) are
the whole contract.  The run index (``index.json``) is the human-readable history.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from .fit_runs import bare_code, default_output_root

IR_CURVE = "ir_curve"
BORROW_CURVE = "borrow_curve"
CURVE_KINDS = (IR_CURVE, BORROW_CURVE)
#: Curve kinds filed **per index** (a folder each): one CNY rate curve for
#: everything is not, a borrow curve implied from one index's futures/options is.
PER_INDEX_KINDS = (BORROW_CURVE,)
LATEST_NAME = "latest.json"
INDEX_NAME = "index.json"
#: The one spelling that switches a curve off (the flat ``--rate`` / ``--borrow``).
NO_CURVE = "none"

_BUILD_COMMAND = {
    IR_CURVE: "python -m surface_pricer build-ir-curve",
    BORROW_CURVE: "python -m surface_pricer build-borrow-curve",
}


def _check_kind(kind: str) -> str:
    text = str(kind or "").strip().lower()
    if text not in CURVE_KINDS:
        raise ValueError(
            "unknown curve kind {!r}: use one of {}".format(kind, ", ".join(CURVE_KINDS))
        )
    return text


def curve_root(
    kind: str,
    output_root: Union[str, Path, None] = None,
    *,
    index: Any = None,
) -> Path:
    """``<output_root>/<kind>`` - and ``.../<bare index>`` for a per-index kind.

    A per-index kind (the borrow curve) keeps one folder per index, so the index is
    in the path instead of buried in a file name - ``borrow_curve/000852/`` holds
    every 000852 borrow run next to *its* ``latest.json`` / ``index.json``.  The one
    CNY rate curve is not per index, so an ``index`` passed for it is ignored.
    """
    kind = _check_kind(kind)
    root = (Path(output_root) if output_root else default_output_root()) / kind
    if kind in PER_INDEX_KINDS and index is not None:
        key = bare_code(index)
        if key:
            return root / key
    return root


def curve_file_name(kind: str, stamp: str) -> str:
    """``ir_curve_20261008_101541.json`` - the run file name for a stamp."""
    return "{}_{}.json".format(_check_kind(kind), stamp)


def curve_run_name(path: Union[str, Path]) -> str:
    """The run label of a curve file (its file name without the extension)."""
    return Path(path).stem


# ------------------------------------------------------------------ writing
def record_curve_run(
    kind: str,
    pillars: Any,
    *,
    output_root: Union[str, Path, None] = None,
    stamp: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    index: Any = None,
) -> Path:
    """Write ``pillars`` as a new stamped run and point ``latest.json`` at it.

    ``pillars`` only has to expose ``to_json(path)`` (both curve pillar classes
    do).  ``details`` lands in the run index next to the file name - the apps pass
    the valuation date / source so the index can be read without opening every
    curve.

    ``index`` says whose curve this is.  A **per-index kind** requires it and files
    the run in that index's folder (``borrow_curve/000852/``), where ``latest.json``
    is a single name - the folder already says which index it is.  The
    one-for-everything interest-rate curve takes no index and keeps the flat folder.
    """
    kind = _check_kind(kind)
    if kind in PER_INDEX_KINDS and not bare_code(index):
        raise ValueError(
            "{} runs are filed per index: pass index=<the index they belong to>, "
            "e.g. index='000852.SH'".format(kind)
        )
    folder = curve_root(kind, output_root, index=index)
    folder.mkdir(parents=True, exist_ok=True)

    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    target = folder / curve_file_name(kind, stamp)
    pillars.to_json(str(target))
    _write_latest(folder, kind, target)
    _refresh_index(folder, kind, target, details)
    return target


def _write_latest(root: Path, kind: str, target: Path) -> None:
    """Point a folder's ``latest.json`` at ``target`` (one name inside its folder)."""
    (root / LATEST_NAME).write_text(
        json.dumps({kind: target.name}, indent=2), encoding="utf-8"
    )


def _read_pointer(path: Path) -> Dict[str, Any]:
    """``latest.json`` as a dict (missing / unreadable -> empty, to be rewritten)."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _refresh_index(
    root: Path,
    kind: str,
    target: Path,
    details: Optional[Mapping[str, Any]],
) -> None:
    entries = [
        item for item in _read_index_at(root, kind) if item.get("file") != target.name
    ]
    entry: Dict[str, Any] = {
        "file": target.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if details:
        entry.update({str(key): value for key, value in details.items()})
    entries.append(entry)
    # newest first; the file name breaks ties (two runs inside one second)
    entries.sort(
        key=lambda item: (str(item.get("created_at", "")), str(item.get("file", ""))),
        reverse=True,
    )
    (root / INDEX_NAME).write_text(
        json.dumps({kind: entries}, indent=2, default=str), encoding="utf-8"
    )


# ------------------------------------------------------------------ reading
def read_curve_index(
    kind: str,
    output_root: Union[str, Path, None] = None,
    *,
    index: Any = None,
) -> List[Dict[str, Any]]:
    """The run index of one curve kind (newest first), empty when there is none."""
    return _read_index_at(curve_root(kind, output_root, index=index), kind)


def _read_index_at(root: Path, kind: str) -> List[Dict[str, Any]]:
    """Read ``index.json`` from a **curve folder** (already ``.../<kind>``)."""
    path = root / INDEX_NAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = payload.get(_check_kind(kind)) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def list_curve_runs(
    kind: str,
    output_root: Union[str, Path, None] = None,
    *,
    index: Any = None,
) -> List[Path]:
    """Stored curve runs, newest first (index first, then a folder scan).

    For a per-index kind this is the index's own folder (plus the flat folder, for
    runs written before the folders existed).
    """
    kind = _check_kind(kind)
    runs: List[Path] = []
    seen: set = set()
    for folder in _pointer_folders(
        curve_root(kind, output_root), curve_root(kind, output_root, index=index)
    ):
        _collect_runs_at(folder, kind, runs, seen)
    return runs


def _collect_runs_at(root: Path, kind: str, runs: List[Path], seen: set) -> None:
    """Append one folder's runs to ``runs`` (index first, then a ``*.json`` scan)."""
    for item in _read_index_at(root, kind):
        candidate = root / str(item.get("file") or "")
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            runs.append(candidate)
    if not root.is_dir():
        return
    reserved = {LATEST_NAME, INDEX_NAME}
    for candidate in sorted(root.glob("*.json"), reverse=True):
        if candidate.name in reserved or candidate in seen:
            continue
        seen.add(candidate)
        runs.append(candidate)


def _pointer_folders(flat: Path, folder: Path) -> List[Path]:
    """The folders to read, most specific first (the index folder, then the root)."""
    return [folder, flat] if folder != flat else [flat]


def _pointer_name(value: Any, index: Any) -> Optional[str]:
    """The run a pointer value names for ``index`` (``None`` = not this index's)."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, Mapping):
        for key in _index_keys(index):
            name = value.get(key)
            if name:
                return str(name)
    return None


def latest_curve_path(
    kind: str,
    output_root: Union[str, Path, None] = None,
    *,
    index: Any = None,
) -> Path:
    """The file ``latest.json`` points at; a missing run is an error, never a guess.

    A **per-index** kind is read from that index's own folder
    (``borrow_curve/000852/latest.json``) - the index is in the path, so one index's
    curve can never be served to another.  The flat pointer is still read for runs
    written before the folders existed: its single name, or the map shape
    (``{"000852": "...", "510500": "..."}``) it briefly had.  Either way an index
    with no entry is an error that lists the indices there are.
    """
    kind = _check_kind(kind)
    flat = curve_root(kind, output_root)
    for folder in _pointer_folders(flat, curve_root(kind, output_root, index=index)):
        value = _read_pointer(folder / LATEST_NAME).get(kind)
        name = _pointer_name(value, index)
        if name is None:
            continue
        target = folder / name
        if target.is_file():
            return target
        raise ValueError(
            "{} points at {} which is not there any more".format(
                folder / LATEST_NAME, name
            )
        )
    if kind in PER_INDEX_KINDS and _index_keys(index):
        raise ValueError(
            "no {} run for {} under {}: it knows {}.  Build one with '{}' for that "
            "venue first (or pass 'none' for the flat rate)".format(
                kind,
                " / ".join(_index_keys(index)),
                flat,
                ", ".join(_known_indexes(flat, kind)) or "(nothing)",
                _BUILD_COMMAND[kind],
            )
        )
    raise ValueError(
        "no {} run under {}; build one with '{}'".format(kind, flat, _BUILD_COMMAND[kind])
    )


def _known_indexes(flat: Path, kind: str) -> List[str]:
    """Indices the flat folder can name: its pointer's keys plus its subfolders."""
    known = set()
    value = _read_pointer(flat / LATEST_NAME).get(kind)
    if isinstance(value, Mapping):
        known.update(str(key) for key in value)
    if flat.is_dir():
        known.update(child.name for child in flat.iterdir() if child.is_dir())
    return sorted(known)


def _index_keys(index: Any) -> List[str]:
    """The lookup keys for an index: one code, or several spellings, in order."""
    values = [index] if isinstance(index, str) else list(index or ())
    keys: List[str] = []
    for value in values:
        key = bare_code(value)
        if key and key not in keys:
            keys.append(key)
    return keys


def resolve_curve_path(
    kind: str,
    spec: Union[str, Path, None],
    *,
    output_root: Union[str, Path, None] = None,
    index: Any = None,
) -> Optional[Path]:
    """Turn a ``--ir-curve`` / ``--borrow-curve`` value into a file path.

    ``None`` / ``""`` / ``"none"`` -> ``None`` (no curve: the flat rate / borrow).
    ``"latest"`` -> the run ``latest.json`` points at **for ``index``** (per-index
    curve kinds; the interest-rate curve ignores it).  Anything else is a **path,
    taken as given** (relative to the cwd) - a missing file is an error naming it,
    not a reason to go looking elsewhere.
    """
    text = str(spec if spec is not None else "").strip()
    if not text or text.lower() == NO_CURVE:
        return None
    if text.lower() == "latest":
        return latest_curve_path(kind, output_root, index=index)
    path = Path(text).expanduser()
    if path.is_file():
        return path
    raise ValueError(
        "{} not found: pass a path (as given), 'latest' for the newest run under "
        "{}, or 'none' for the flat rate".format(path, curve_root(kind, output_root))
    )


def curve_details(pillars: Any) -> Dict[str, Any]:
    """A few ``pillars`` fields for the run index (missing ones are skipped)."""
    details: Dict[str, Any] = {}
    for key in ("curve_name", "valuation_date", "source", "observed", "extended"):
        value = getattr(pillars, key, None)
        if value is None:
            continue
        details[key] = value.date().isoformat() if hasattr(value, "date") else value
    return details


__all__ = [
    "BORROW_CURVE",
    "CURVE_KINDS",
    "IR_CURVE",
    "INDEX_NAME",
    "LATEST_NAME",
    "NO_CURVE",
    "PER_INDEX_KINDS",
    "curve_details",
    "curve_file_name",
    "curve_root",
    "curve_run_name",
    "latest_curve_path",
    "list_curve_runs",
    "read_curve_index",
    "record_curve_run",
    "resolve_curve_path",
]
