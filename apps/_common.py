"""Shared helpers for the command line entry points."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

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


# Backwards compatible aliases (the pre-restructure private names).
_load_env_files = load_env_files
_parse_env_file = parse_env_file
_env = env
_print_progress = print_progress
_null_reporter = null_reporter


__all__ = [
    "DEFAULT_ENV_FILES",
    "env",
    "load_env_files",
    "null_reporter",
    "parse_env_file",
    "print_progress",
    "reporter_for",
]
