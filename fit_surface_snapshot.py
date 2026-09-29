"""Deprecated entry point kept for backwards compatibility.

Use the unified launcher instead::

    python -m surface_pricer fit [options]

The implementation lives in :mod:`surface_pricer.apps.fit_surface`.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # ``python surface_pricer/fit_surface_snapshot.py`` has no package context,
    # so the relative imports below cannot resolve; hand control to the package
    # module instead (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from surface_pricer.fit_surface_snapshot import main

    raise SystemExit(main())

from .apps._common import (
    _env,
    _load_env_files,
    _null_reporter,
    _parse_env_file,
    _print_progress,
)
from .apps.fit_surface import _build_override_config, _parse_args, main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
