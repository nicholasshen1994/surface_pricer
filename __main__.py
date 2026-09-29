"""Unified launcher: ``python -m surface_pricer <command> [options]``.

Commands::

    fit                 fetch a listed-option snapshot, fit the surface, write report/plots
    price               value a term sheet (JSON/CSV/XLSX) against an offline surface
    price-tool          quote NPV + Greeks of one option on a stored fit run
    build-ir-curve      bootstrap the CNY FR007 curve from data/interest_rate.csv
    build-borrow-curve  imply the borrow curve from futures + listed options
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # ``python surface_pricer/__main__.py`` has no package context, so the
    # relative imports below cannot resolve; hand control to the package module
    # instead (``python -m surface_pricer`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from surface_pricer.__main__ import main

    raise SystemExit(main())

import sys
from typing import List, Optional

USAGE = """usage: python -m surface_pricer <command> [options]

commands:
  fit                  fetch a snapshot, fit the EDS SABR surface (apps/fit_surface.py)
  price                value a term sheet against an offline surface (apps/price_trades.py)
  price-tool           quote NPV + Greeks of one option on a stored fit run (price_tool.py)
  build-ir-curve       bootstrap the CNY FR007 curve from the rate export
  build-borrow-curve   imply the borrow curve from futures + listed options

run '<command> --help' for the command specific options.
"""


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    command, rest = args[0], args[1:]
    if command in {"fit", "fit-surface", "fit_surface"}:
        from .apps.fit_surface import main as fit_main

        return fit_main(rest)
    if command in {"price", "price-trades", "price_trades"}:
        from .apps.price_trades import main as price_main

        return price_main(rest)
    if command in {"price-tool", "price_tool", "quote", "pricer"}:
        from .price_tool import main as price_tool_main

        # no arguments -> use price_tool.QUICK_DEFAULTS (edit them there)
        return price_tool_main(rest, quick=not rest)
    if command in {"build-ir-curve", "build_ir_curve", "ir-curve"}:
        from .apps.build_ir_curve import main as ir_main

        return ir_main(rest)
    if command in {"build-borrow-curve", "build_borrow_curve", "borrow-curve"}:
        from .apps.build_borrow_curve import main as borrow_main

        return borrow_main(rest)

    print("unknown command {!r}\n".format(command))
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
