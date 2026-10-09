"""Unified launcher: ``python -m surface_pricer <command> [options]``.

Commands::

    fit                 fetch a listed-option snapshot, fit the surface, write report/plots
    build-json          write a resolved contract payload (JSON) from a quick block
    price-json          price a resolved contract payload (JSON) and report its Greeks
    autocall-pricer     solve the coupon that prices an autocall at a target NPV
    slide               spot ladder of a resolved contract payload (--slide of price-json)
    build-ir-curve      bootstrap the CNY FR007 curve from data/interest_rate.csv
    build-borrow-curve  imply the borrow curve from futures + listed options
    fetch-contracts     pull listed-option terms from Wind into data/option_contracts.json
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
  build-json           write a resolved contract payload from a quick block (apps/build_json.py)
  price-json           price a resolved contract payload and report its Greeks (apps/price_json.py)
  autocall-pricer      solve the coupon for a target NPV (apps/autocall_pricer.py)
  slide                spot ladder of a resolved contract payload (apps/price_json.py --slide)
  build-ir-curve       bootstrap the CNY FR007 curve from the rate export
  build-borrow-curve   imply the borrow curve from futures + listed options
  fetch-contracts      pull listed-option terms from Wind into data/option_contracts.json

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
    if command in {"build-json", "build_json", "make-json"}:
        from .apps.build_json import main as build_json_main

        return build_json_main(rest)
    if command in {"price-json", "price_json", "price-spec"}:
        from .apps.price_json import main as price_json_main

        # no arguments -> use price_json.QUICK_DEFAULTS (edit them there)
        return price_json_main(rest, quick=not rest)
    if command in {"autocall-pricer", "autocall_pricer", "solve-coupon"}:
        from .apps.autocall_pricer import main as autocall_pricer_main

        # no arguments -> use autocall_pricer.QUICK_DEFAULTS (edit them there)
        return autocall_pricer_main(rest, quick=not rest)
    if command in {"slide", "spot-slide", "spot_slide"}:
        from .apps.price_json import main as price_json_main

        # the ladder is a mode of price-json: same payload, same market, one rung
        # per spot.  No arguments -> the quick block, with the ladder mode kept.
        return price_json_main(["--slide", *rest], quick=not rest)
    if command in {"build-ir-curve", "build_ir_curve", "ir-curve"}:
        from .apps.build_ir_curve import main as ir_main

        return ir_main(rest)
    if command in {"build-borrow-curve", "build_borrow_curve", "borrow-curve"}:
        from .apps.build_borrow_curve import main as borrow_main

        # no arguments -> use build_borrow_curve.QUICK_DEFAULTS (edit them there)
        return borrow_main(rest, quick=not rest)
    if command in {"fetch-contracts", "fetch_contracts", "contracts"}:
        from .apps.fetch_contracts import main as fetch_contracts_main

        return fetch_contracts_main(rest)

    print("unknown command {!r}\n".format(command))
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
