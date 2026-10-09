"""Shared helpers for the command line entry points."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

DEFAULT_ENV_FILES = (
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / ".env",
)


def load_env_files(explicit: Optional[str] = None) -> None:
    """Load key=value pairs from .env files without overriding real env vars."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(DEFAULT_ENV_FILES)
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        parse_env_file(path)


def parse_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = value


def env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def print_progress(message: str) -> None:
    print(message, flush=True)


def null_reporter(message: str) -> None:
    return None


def reporter_for(quiet: bool) -> Callable[[str], None]:
    return null_reporter if quiet else print_progress


# --------------------------------------------------------------- JSON payloads
def load_payload(path: str) -> Any:
    """Read a JSON payload; ``-`` / ``stdin`` reads it from standard input.

    Shared by the entry points that accept a resolved contract (``--spec``), so
    the stdin convention is the same everywhere.
    """
    if str(path).strip() in {"-", "stdin"}:
        return json.loads(sys.stdin.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


#: The package directory (``surface_pricer/``): what a quick block's own relative
#: *output* paths are resolved against (see :func:`quick_output_path`).
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def dump_payload(path: str, payload: Any) -> None:
    """Write a JSON payload; ``-`` / ``stdout`` prints it instead.

    The write is confirmed on **stderr** (so ``--json`` stays parseable) and the
    note carries the **absolute** path: a relative path means whatever the process
    working directory happens to be - the IDE's guess when the app is launched
    with the Run button - so "wrote test.json" without a directory is not
    something a reader can find.
    """
    text = json.dumps(payload, indent=2)
    if str(path).strip() in {"-", "stdout"}:
        print(text)
        return
    target = Path(str(path)).expanduser()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    kind = payload.get("kind") if isinstance(payload, Mapping) else None
    print(
        "note: wrote {}{}".format(
            target.resolve(), "" if not kind else " ({})".format(kind)
        ),
        file=sys.stderr,
    )


def quick_output_path(value: Optional[str]) -> Optional[str]:
    """Resolve a ``QUICK_DEFAULTS`` output path against the **package**, not the cwd.

    The quick block is project configuration, while the working directory belongs
    to whoever launched the process (an IDE Run button, a shell in some folder), so
    a relative path there - ``"output/test.json"`` - is read from the package
    directory: the artifact lands with the fit runs, which is where a reader looks
    for it.  Absolute
    paths and ``-``/``stdout`` are honoured as written, and a missing parent folder
    is created (that is what "output/" means here).
    """
    text = str(value if value is not None else "").strip()
    if not text or text in {"-", "stdout"}:
        return value
    target = Path(text).expanduser()
    if target.is_absolute():
        return str(target)
    resolved = PACKAGE_DIR / target
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def payload_text(payload: Any, max_items: int = 6, indent: int = 2) -> str:
    """Pretty JSON of a resolved payload, with long arrays elided.

    A daily-monitored snowball carries one monitoring date and level per business
    day, so the faithful payload is ~500 lines; arrays longer than ``max_items``
    keep their head and tail.  What was dropped is reported by :func:`echo_payload`
    (on stderr, so the JSON stays clean), and ``--spec-out FILE`` always writes the
    payload in full.
    """
    def trim(value: Any) -> Any:
        if isinstance(value, list) and len(value) > max_items:
            return value[:2] + ["... {} more ...".format(len(value) - 3)] + value[-1:]
        if isinstance(value, dict):
            return {key: trim(item) for key, item in value.items()}
        return value

    return json.dumps(trim(payload), indent=indent, default=str)


def elided_arrays(payload: Any, max_items: int = 6) -> List[str]:
    """Dotted paths of the arrays :func:`payload_text` would shorten."""
    found: List[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, list):
            if len(value) > max_items:
                found.append("{} ({} entries)".format(path or "list", len(value)))
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, "{}.{}".format(path, key) if path else str(key))

    walk(payload, "")
    return found


def echo_payload(payload: Any, max_items: int = 6) -> None:
    """Print the resolved payload, noting on stderr anything elided for width."""
    print(payload_text(payload, max_items=max_items))
    dropped = elided_arrays(payload, max_items=max_items)
    if dropped:
        print(
            "note: {} elided for width - use --spec-out FILE for the payload in full".format(
                ", ".join(dropped)
            ),
            file=sys.stderr,
        )


# Backwards compatible aliases (the pre-restructure private names).
_load_env_files = load_env_files
_parse_env_file = parse_env_file
_env = env
_print_progress = print_progress
_null_reporter = null_reporter


__all__ = [
    "DEFAULT_ENV_FILES",
    "PACKAGE_DIR",
    "dump_payload",
    "echo_payload",
    "elided_arrays",
    "env",
    "load_env_files",
    "load_payload",
    "null_reporter",
    "parse_env_file",
    "payload_text",
    "print_progress",
    "quick_output_path",
    "reporter_for",
]
