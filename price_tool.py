"""Query NPV and Greeks for one option against a stored fit run.

Examples::

    python -m surface_pricer.price_tool --fit latest --strike 7500 --tenor 3M --call
    python -m surface_pricer.price_tool --fit latest --strike 7500 --tenor 3M --put --json
    python -m surface_pricer.price_tool --fit MO_20260929_150000 --strike 100 \
        --strike-type percentage --tenor 6M --put --notional 1000000
    python -m surface_pricer.price_tool --list-runs
    python -m surface_pricer.price_tool --fit latest --interactive

The tool reads the ``surface.json`` (+ ``manifest.json``) written by
``python -m surface_pricer fit`` under ``surface_pricer/output``.  ``--fit``
accepts ``latest``, a run name (a unique prefix is enough), a run directory or
a path to a ``surface.json``.

Quick run: edit the ``QUICK_DEFAULTS`` block below and execute the file (or
``python -m surface_pricer.price_tool``) **without arguments** - e.g. the IDE
"Run" button.  Any command line argument disables the block and the normal CLI
rules apply.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # ``python surface_pricer/price_tool.py`` has no package context, so the
    # relative imports below cannot resolve; hand control to the package module
    # instead (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from surface_pricer.price_tool import main

    raise SystemExit(main())

import argparse
import json
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

from .core.daycount import shift_tenor, to_datetime
from .core.market import MarketState
from .io.curve_files import curve_valuation_date, load_borrow_curve, load_ir_curve
from .io.fit_runs import FitRun, list_runs, resolve_run
from .io.serialization import market_from_surface
from .pricing.contracts import VanillaContract
from .pricing.greeks import calculate_greeks
from .pricing.results import PricingResult, RiskSettings
from .reporting.quote_report import format_quote, quote_to_dict


# ---------------------------------------------------------------------------
# Quick-run defaults
#
# Edit this block, then run the file (or ``python -m surface_pricer.price_tool``)
# with **no arguments** - e.g. with the IDE "Run" button - to get one quote
# straight away.  Passing any command line argument disables the block and the
# usual CLI rules (and defaults) apply again, so scripts are unaffected.
# ---------------------------------------------------------------------------
QUICK_DEFAULTS: Dict[str, Any] = {
    "fit": "latest",            # 'latest' / run-name prefix / directory / surface.json
    "output_root": None,        # None -> surface_pricer/output
    "strike": 7900.0,
    "strike_type": "absolute",  # absolute / percentage / spot_percentage / fwd_percentage
    "tenor": None,              # 3M / 1Y / 90D ... (used when ``expiry`` is None)
    "expiry": "2026-10-21",             # explicit YYYY-MM-DD, wins over ``tenor``
    "option_type": "call",       # call / put
    "notional": 5000.0,
    "spot": None,               # None -> spot stored in the fit run
    "rate": None,               # None -> rate stored in the fit run
    "borrow": 0.0,
    "ir_curve": None,           # ir_curve.json from build-ir-curve; replaces the flat rate
    "borrow_curve": "surface_pricer/output/borrow_curve.json",       # borrow_curve.json from build-borrow-curve; replaces --borrow
    "valuation_date": None,     # None -> valuation date of the fit run
    "calendar_file": None,
    "no_buckets": False,
    "json": False,
    "list_runs": False,
    "interactive": False,
}


def _apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Overwrite the parsed arguments with :data:`QUICK_DEFAULTS`."""
    for key, value in QUICK_DEFAULTS.items():
        setattr(args, key, value)
    option_type = str(args.option_type or "call").strip().lower()
    args.option_type = option_type
    args.call = option_type == "call"
    args.put = option_type == "put"
    return args


def _no_cli_arguments() -> bool:
    return len(sys.argv) <= 1


def main(argv: Optional[Iterable[str]] = None, *, quick: bool = False) -> int:
    use_quick = quick or (argv is None and _no_cli_arguments())
    args = _parse_args(argv)
    if use_quick:
        args = _apply_quick_defaults(args)

    if args.list_runs:
        runs = list_runs(args.output_root)
        if not runs:
            print("no fit run found; run 'python -m surface_pricer fit' first")
            return 0
        for run in runs:
            print(run.describe())
        return 0

    try:
        run = resolve_run(args.fit, output_root=args.output_root)
    except ValueError as error:
        print("ERROR: {}".format(error))
        return 2

    try:
        market = _market_from_run(run, args)
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as error:
        print("ERROR: cannot build the market state from {}: {}".format(run.name, error))
        return 2

    if args.interactive:
        return _interactive(run, market, args)

    if args.strike is None:
        print("ERROR: --strike is required (or pass --interactive / --list-runs)")
        return 2
    if not args.tenor and not args.expiry:
        print("ERROR: --tenor or --expiry is required")
        return 2
    try:
        return _quote_once(run, market, args)
    except ValueError as error:
        print("ERROR: {}".format(error))
        return 2


# ------------------------------------------------------------------ one quote
def _quote_once(run: FitRun, market: MarketState, args: argparse.Namespace) -> int:
    contract, info = _build_contract(run, market, args)
    result = _calculate(contract, market, args)
    if args.json:
        print(
            json.dumps(
                quote_to_dict(result, run=run.name, contract=info),
                indent=2,
                default=str,
            )
        )
    else:
        print(format_quote(result, run=run.describe(), contract=info))
    return 0


def _interactive(run: FitRun, market: MarketState, args: argparse.Namespace) -> int:
    print("fit run : {}".format(run.describe()))
    print("input   : <strike> <tenor|expiry> <call|put> [notional]   ('q' or blank to quit)")
    while True:
        try:
            line = input("quote> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"q", "quit", "exit"}:
            break
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            print("need: <strike> <tenor|expiry> <call|put> [notional]")
            continue
        try:
            strike = float(parts[0])
        except ValueError:
            print("strike must be a number: {!r}".format(parts[0]))
            continue
        option_type = parts[2].strip().lower()
        if option_type not in {"c", "call", "p", "put"}:
            print("option type must be call or put: {!r}".format(parts[2]))
            continue
        local = _clone_args(args)
        local.strike = strike
        local.put = option_type in {"p", "put"}
        local.call = not local.put
        local.option_type = "put" if local.put else "call"
        local.notional = float(parts[3]) if len(parts) > 3 else float(args.notional)
        if _looks_like_date(parts[1]):
            local.expiry, local.tenor = parts[1], None
        else:
            local.tenor, local.expiry = parts[1], None
        try:
            contract, info = _build_contract(run, market, local)
            result = _calculate(contract, market, local)
        except ValueError as error:
            print("ERROR: {}".format(error))
            continue
        print()
        print(format_quote(result, contract=info))
        print()
    return 0


def _clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


# ------------------------------------------------------------------ contracts
def _build_contract(
    run: FitRun,
    market: MarketState,
    args: argparse.Namespace,
) -> Tuple[VanillaContract, Dict[str, Any]]:
    expiry = _resolve_expiry(market, args)
    option_type = _option_type(args)
    contract = VanillaContract(
        expiry=expiry,
        strike=float(args.strike),
        option_type=option_type,
        notional=float(args.notional),
        strike_type=str(args.strike_type),
    )
    info: Dict[str, Any] = {
        "option_type": option_type,
        "strike": float(args.strike),
        "strike_type": str(args.strike_type),
        "tenor": args.tenor,
        "expiry": to_datetime(expiry).date().isoformat(),
        "notional": float(args.notional),
    }
    return contract, info


def _resolve_expiry(market: MarketState, args: argparse.Namespace):
    if args.expiry:
        value = to_datetime(args.expiry)
    elif args.tenor:
        value = to_datetime(shift_tenor(market.valuation_date, args.tenor, market.calendar))
    else:
        raise ValueError("tenor or expiry is required")
    if value <= market.valuation_date:
        raise ValueError(
            "expiry {} is not after the valuation date {}".format(
                value.date(), market.valuation_date.date()
            )
        )
    return value


def _option_type(args: argparse.Namespace) -> str:
    if getattr(args, "put", False):
        return "put"
    if getattr(args, "call", False):
        return "call"
    return str(args.option_type or "call").strip().lower()


def _calculate(
    contract: VanillaContract,
    market: MarketState,
    args: argparse.Namespace,
) -> PricingResult:
    return calculate_greeks(
        contract,
        market,
        settings=RiskSettings(),
        bucketed_vega_pillars=[] if args.no_buckets else None,
        bucketed_delta_pillars=[] if args.no_buckets else None,
    )


# --------------------------------------------------------------------- market
def _market_from_run(run: FitRun, args: argparse.Namespace) -> MarketState:
    payload = run.surface_payload()
    spot = args.spot if args.spot is not None else run.spot
    rate = args.rate if args.rate is not None else run.rate
    valuation_date = args.valuation_date or run.valuation_datetime

    rate_curve = load_ir_curve(args.ir_curve) if getattr(args, "ir_curve", None) else None
    borrow_curve = (
        load_borrow_curve(args.borrow_curve)
        if getattr(args, "borrow_curve", None)
        else None
    )
    _warn_curve_anchor(rate_curve, valuation_date, "--ir-curve")
    _warn_curve_anchor(borrow_curve, valuation_date, "--borrow-curve")

    return market_from_surface(
        payload,
        spot=spot,
        rate=rate,
        borrow=float(args.borrow or 0.0),
        valuation_date=valuation_date,
        calendar_file=args.calendar_file,
        rate_curve=rate_curve,
        borrow_curve=borrow_curve,
    )


def _warn_curve_anchor(curve, valuation_date, flag: str) -> None:
    """Warn (stderr, so ``--json`` stays clean) on a curve/run date mismatch."""
    if curve is None:
        return
    anchor = curve_valuation_date(curve)
    if anchor is None:
        return
    reference = to_datetime(valuation_date).date()
    if (anchor - reference).days != 0:
        print(
            "WARNING: {} was built for {} but the valuation date is {}".format(
                flag, anchor.isoformat(), reference.isoformat()
            ),
            file=sys.stderr,
        )


def _looks_like_date(text: str) -> bool:
    value = str(text).strip()
    if "-" not in value:
        return False
    try:
        to_datetime(value)
    except (ValueError, TypeError):
        return False
    return True


# ------------------------------------------------------------------------ CLI
def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="surface_pricer price-tool",
        description=(
            "Quote NPV and Greeks of one option on a stored fit run "
            "(surface_pricer/output)."
        ),
    )
    parser.add_argument(
        "--fit",
        default="latest",
        help="fit run: 'latest' (default), a run name (prefix ok), a directory or a surface.json",
    )
    parser.add_argument("--output-root", default=None, help="fit output root (default surface_pricer/output)")
    parser.add_argument("--list-runs", action="store_true", help="list stored fit runs and exit")
    parser.add_argument("--strike", type=float, default=None, help="strike (absolute or percentage)")
    parser.add_argument(
        "--strike-type",
        default="absolute",
        choices=["absolute", "percentage", "spot_percentage", "fwd_percentage"],
        help="how to interpret --strike (default absolute)",
    )
    parser.add_argument("--tenor", default=None, help="expiry tenor, e.g. 3M / 1Y / 90D")
    parser.add_argument("--expiry", default=None, help="explicit expiry date (YYYY-MM-DD)")
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--call", action="store_true", help="call option (default)")
    direction.add_argument("--put", action="store_true", help="put option")
    parser.add_argument("--option-type", default="call", choices=["call", "put"], help="alternative to --call/--put")
    parser.add_argument("--notional", type=float, default=1.0, help="contract notional (default 1.0)")
    parser.add_argument("--spot", type=float, default=None, help="override the run spot")
    parser.add_argument("--rate", type=float, default=None, help="override the run rate")
    parser.add_argument("--borrow", type=float, default=0.0, help="flat borrow rate (default 0)")
    parser.add_argument(
        "--ir-curve",
        default=None,
        help="ir_curve.json from 'build-ir-curve' (replaces the flat rate)",
    )
    parser.add_argument(
        "--borrow-curve",
        default=None,
        help="borrow_curve.json from 'build-borrow-curve' (replaces the flat borrow)",
    )
    parser.add_argument("--valuation-date", default=None, help="override the run valuation date")
    parser.add_argument("--calendar-file", default=None, help="JSON file with calendar holidays")
    parser.add_argument("--no-buckets", action="store_true", help="skip bucketed vega / rhoQ / delta")
    parser.add_argument("--json", action="store_true", help="print the quote as JSON")
    parser.add_argument("--interactive", action="store_true", help="read strike/tenor/type from stdin in a loop")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["QUICK_DEFAULTS", "main"]


if __name__ == "__main__":
    sys.exit(main())
