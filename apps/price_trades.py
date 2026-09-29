"""Value a term sheet against an offline fitted surface.

Run from the repository root::

    cd C:\\Code\\edslib
    python -m surface_pricer price --terms trades.json --surface surface.json
    python -m surface_pricer price --terms trades.csv --surface surface.json \\
        --rate 0.015 --borrow 0.01 --out surface_pricer/output/portfolio_demo

``--surface`` accepts the ``surface.json`` written by ``surface_pricer fit``
(spot / valuation date / calendar are taken from it unless overridden); pass
``--market`` instead to use a full market payload accepted by
:func:`surface_pricer.io.serialization.market_from_dict`.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # ``python surface_pricer/apps/price_trades.py`` has no package context, so
    # the relative imports below cannot resolve; hand control to the package
    # module instead (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.price_trades import main

    raise SystemExit(main())

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from ..core.market import MarketState
from ..io.serialization import market_from_dict, market_from_surface
from ..portfolio.report import format_summary, write_csv, write_json
from ..portfolio.terms import load_terms
from ..portfolio.valuation import value_portfolio
from ..pricing.results import RiskSettings
from ._common import load_env_files, reporter_for

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    load_env_files(args.env_file)

    try:
        terms = load_terms(args.terms)
    except (OSError, ValueError) as error:
        print("ERROR: cannot read the term sheet: {}".format(error))
        return 2
    try:
        market = _build_market(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("ERROR: cannot build the market state: {}".format(error))
        return 2

    reporter = reporter_for(args.quiet)
    reporter(
        "valuing {} trade(s) at {} | spot={:.4f}".format(
            len(terms), market.valuation_date.date().isoformat(), market.spot
        )
    )
    started = time.perf_counter()
    portfolio = value_portfolio(
        terms,
        market,
        settings=RiskSettings(),
        with_risk=not args.no_risk,
        bucketed_vega_pillars=[] if args.no_buckets else None,
        bucketed_delta_pillars=[] if args.no_buckets else None,
    )
    elapsed = time.perf_counter() - started

    print(format_summary(portfolio))
    output_dir = (
        Path(args.out)
        if args.out
        else DEFAULT_OUTPUT_ROOT
        / "portfolio_{}".format(market.valuation_date.strftime("%Y%m%d_%H%M%S"))
    )
    csv_path = write_csv(portfolio, str(output_dir / "trades.csv"))
    json_path = write_json(portfolio, str(output_dir / "trades.json"))
    print("outputs: {} | {}".format(csv_path, json_path))
    print(
        "elapsed: {:.2f}s ({:.3f}s per trade, risk={})".format(
            elapsed, elapsed / max(len(terms), 1), "off" if args.no_risk else "on"
        )
    )
    return 0


def _build_market(args: argparse.Namespace) -> MarketState:
    if args.market:
        payload = json.loads(Path(args.market).read_text(encoding="utf-8"))
        return market_from_dict(payload, calendar_file=args.calendar_file)
    if not args.surface:
        raise ValueError("either --market or --surface is required")
    payload = json.loads(Path(args.surface).read_text(encoding="utf-8"))
    return market_from_surface(
        payload,
        spot=args.spot,
        rate=float(args.rate),
        borrow=float(args.borrow),
        valuation_date=args.valuation_date,
        calendar_file=args.calendar_file,
    )


def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="surface_pricer price")
    parser.add_argument("--terms", required=True, help="term sheet (JSON / CSV / XLSX)")
    parser.add_argument("--market", default=None, help="full market payload (JSON)")
    parser.add_argument(
        "--surface",
        default=None,
        help="surface.json produced by 'surface_pricer fit' (spot/date/calendar taken from it)",
    )
    parser.add_argument("--calendar-file", default=None, help="JSON file with calendar holidays")
    parser.add_argument("--spot", type=float, default=None, help="override the surface spot")
    parser.add_argument("--rate", type=float, default=0.0, help="flat risk-free rate")
    parser.add_argument("--borrow", type=float, default=0.0, help="flat borrow rate")
    parser.add_argument("--valuation-date", default=None, help="override the surface valuation date")
    parser.add_argument("--out", default=None, help="output directory (default: surface_pricer/output/portfolio_<ts>)")
    parser.add_argument("--no-risk", action="store_true", help="price only, skip the Greek bumps")
    parser.add_argument("--no-buckets", action="store_true", help="skip bucketed vega / rhoQ / delta")
    parser.add_argument("--env-file", default=None, help="additional .env file")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
