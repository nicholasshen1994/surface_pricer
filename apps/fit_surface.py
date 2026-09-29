"""Fetch a listed-option snapshot, fit the EDS SABR surface and write charts.

Run from the repository root::

    cd C:\\Code\\edslib
    python -m surface_pricer fit                       # configured defaults
    python -m surface_pricer fit --underlying IO --index 000300.SH
    python -m surface_pricer fit --pin "2027-06-18:atm_vol=0.215,skew=0.04"

Without arguments the run uses the defaults configured below.  The gateway
password is read from ``surface_pricer/.env`` first (falling back to the real
environment variables ``QUOTE_GATEWAY_PASSWORD`` / ``CICC_QUOTE_PASSWORD``).

Supported underlyings: MO / IO / HO (CFFEX index options) and
510500 / 588000 / 159915 (ETF options).
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # ``python surface_pricer/apps/fit_surface.py`` has no package context, so
    # the relative imports below cannot resolve; hand control to the package
    # module instead (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.fit_surface import main

    raise SystemExit(main())

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..fitting.overrides import OverrideConfig, parse_pin_spec
from ..fitting.pipeline import fit_surface
from ..fitting.settings import FitSettings
from ..io.curve_files import load_borrow_curve, load_ir_curve
from ..io.fit_runs import record_fit_run
from ..marketdata.gateway import QuoteGatewaySnapshotClient
from ..marketdata.providers import QuoteApiDataProvider
from ..reporting.fit_report import format_fit_report
from ..reporting.plots import plot_fit_result
from ._common import (
    DEFAULT_ENV_FILES,
    env,
    load_env_files,
    reporter_for,
)

# ---------------------------------------------------------------------------
# Default run configuration - edit here, CLI flags override any of it.
# ---------------------------------------------------------------------------
DEFAULT_UNDERLYING = "MO"  # MO / IO / HO / 510500 / 588000 / 159915
DEFAULT_INDEX: Optional[str] = None  # e.g. "000852.SH"; None uses the registry
DEFAULT_RATE = 0.015
DEFAULT_WEIGHT_MODE = "vega"  # vega / atm_vega / equal / spread
DEFAULT_MAX_ITERATIONS = 300
DEFAULT_REPORT_ROWS = 0  # per-expiry quote rows appended to the report

DEFAULT_HOST = "10.43.1.9"
DEFAULT_PORT = 7070
DEFAULT_USER = "eq_eds_trd"

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    load_env_files(args.env_file)

    password = env("QUOTE_GATEWAY_PASSWORD", "CICC_QUOTE_PASSWORD")
    if not password:
        print(
            "ERROR: missing quote gateway password.\n"
            "       Write QUOTE_GATEWAY_PASSWORD=<password> into {}\n"
            "       (a template is already there) or export the environment "
            "variable.".format(DEFAULT_ENV_FILES[0])
        )
        return 2
    host = env("QUOTE_GATEWAY_HOST", "CICC_QUOTE_API_HOST") or DEFAULT_HOST
    port = int(env("QUOTE_GATEWAY_PORT", "CICC_QUOTE_API_PORT") or DEFAULT_PORT)
    user = env("QUOTE_GATEWAY_USER", "CICC_QUOTE_USER") or DEFAULT_USER

    underlying = str(args.underlying).strip().upper()
    try:
        override_config = _build_override_config(args)
    except (OSError, ValueError) as error:
        print("ERROR: invalid override configuration: {}".format(error))
        return 2
    settings = FitSettings(
        weight_mode=args.weight_mode,
        max_iterations=int(args.maxiter),
        override_config=override_config,
    )
    reporter = reporter_for(args.quiet)
    if override_config is not None:
        reporter(
            "hand overrides | {} pinned expiry(ies) | synthetic tenors {}".format(
                len(override_config.overrides),
                "on" if override_config.extend_synthetic_tenors else "off",
            )
        )

    try:
        rate_curve = load_ir_curve(args.ir_curve) if args.ir_curve else None
        borrow_curve = load_borrow_curve(args.borrow_curve) if args.borrow_curve else None
    except ValueError as error:
        print("ERROR: cannot load curve: {}".format(error))
        return 2
    if rate_curve is not None or borrow_curve is not None:
        reporter(
            "curves     | ir={} | borrow={}".format(
                args.ir_curve or "flat {:.4f}".format(float(args.rate)),
                args.borrow_curve or "(none)",
            )
        )

    started = time.perf_counter()
    reporter("connecting to quote gateway {}:{} as {} ...".format(host, port, user))
    with QuoteGatewaySnapshotClient(host, port, user, password) as client:
        provider = QuoteApiDataProvider(
            client,
            rate=float(args.rate),
            rate_curve=rate_curve,
            borrow_curve=borrow_curve,
        )
        if args.index:
            spec = replace(provider.spec_for(underlying), index_ticker=str(args.index))
            provider = QuoteApiDataProvider(
                client,
                rate=float(args.rate),
                rate_curve=rate_curve,
                borrow_curve=borrow_curve,
                spec_overrides={underlying: spec},
            )
        reporter("loading {} option chain + spot snapshot ...".format(underlying))
        load_started = time.perf_counter()
        snapshot = provider.load(underlying)
        reporter("snapshot loaded in {:.2f}s".format(time.perf_counter() - load_started))
        result = fit_surface(snapshot, settings, progress=reporter)
    reporter("fit stage finished in {:.1f}s".format(time.perf_counter() - started))

    stamp = result.valuation_datetime.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT
    output_dir = output_root / "{}_{}".format(underlying, stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = format_fit_report(result, max_smile_rows_per_expiry=int(args.report_rows))
    print(report)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    if override_config is not None:
        override_config.to_json(str(output_dir / "overrides.json"))
    if not args.no_plot:
        reporter("writing charts ...")
        written = plot_fit_result(
            result,
            output_dir,
            title_prefix=underlying,
            show=bool(args.show),
        )
        for path in written:
            print("plot: {}".format(path))

    # Store the run (surface.json + manifest.json + index/latest) so that
    # ``surface_pricer.price_tool`` can pick it up by name.
    run = record_fit_run(
        output_dir,
        surface=result.surface,
        underlying=underlying,
        valuation_datetime=result.valuation_datetime,
        spot=result.spot,
        settings=settings,
        metrics=result.metrics,
        overrides=override_config,
        rate=float(args.rate),
        calendar_name=getattr(result.surface.calendar, "name", None),
        trading_days_per_year=result.surface.trading_days_per_year,
        holiday_weight=result.surface.holiday_weight,
        extra={
            "host": host,
            "user": user,
            "scenario": "snapshot",
            "ir_curve": args.ir_curve,
            "borrow_curve": args.borrow_curve,
        },
    )
    print("fit run  : {}".format(run.describe()))
    print("outputs  : {}".format(run.directory))
    print("total elapsed: {:.1f}s".format(time.perf_counter() - started))
    return 0


def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="surface_pricer fit")
    parser.add_argument("--underlying", default=DEFAULT_UNDERLYING)
    parser.add_argument("--index", default=DEFAULT_INDEX, help="override the default index level ticker")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument(
        "--ir-curve",
        default=None,
        help="ir_curve.json from 'build-ir-curve' (replaces --rate)",
    )
    parser.add_argument(
        "--borrow-curve",
        default=None,
        help="borrow_curve.json from 'build-borrow-curve' (adds borrow to the forwards)",
    )
    parser.add_argument(
        "--weight-mode",
        default=DEFAULT_WEIGHT_MODE,
        choices=["vega", "atm_vega", "equal", "spread"],
    )
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--report-rows", type=int, default=DEFAULT_REPORT_ROWS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--env-file", default=None, help="additional .env file with gateway credentials")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--override-file",
        default=None,
        help="JSON file with hand overrides (format: surface_pricer/fitting/overrides.py)",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=None,
        metavar="EXPIRY:FIELD=VALUE[,FIELD=VALUE]",
        help=(
            'pin one expiry, e.g. --pin "2027-06-18:atm_vol=0.215,skew=0.04"; '
            "repeat the flag for several expiries"
        ),
    )
    parser.add_argument(
        "--extend-tenors",
        dest="extend_tenors",
        action="store_true",
        default=False,
        help="add synthetic June/December third-Friday tenors beyond the listed expiries",
    )
    parser.add_argument(
        "--no-extend-tenors",
        dest="extend_tenors",
        action="store_false",
        help="disable the synthetic tenor extension (default)",
    )
    parser.add_argument(
        "--synthetic-end-tenor",
        default=None,
        help="horizon of the synthetic extension, default 3Y",
    )
    parser.add_argument(
        "--synthetic-months",
        default=None,
        metavar="M1,M2",
        help="calendar months of the synthetic tenors, default 6,12",
    )
    parser.add_argument(
        "--synthetic-week",
        type=int,
        default=None,
        help="which occurrence of the weekday inside the month, default 3 (third Friday)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_override_config(args: argparse.Namespace) -> Optional[OverrideConfig]:
    """Merge ``--override-file`` / ``--pin`` / synthetic flags into one config.

    Returns ``None`` when nothing was requested, so a plain run keeps the exact
    historical code path.
    """
    config = (
        OverrideConfig.from_json(args.override_file)
        if args.override_file
        else OverrideConfig()
    )
    payload = config.to_dict()
    if args.pin:
        payload["overrides"] = list(payload["overrides"]) + [
            parse_pin_spec(text).to_dict() for text in args.pin
        ]
    if args.extend_tenors:
        payload["extend_synthetic_tenors"] = True
    if args.synthetic_end_tenor:
        payload["synthetic_end_tenor"] = str(args.synthetic_end_tenor)
    if args.synthetic_months:
        try:
            payload["synthetic_months"] = [
                int(part)
                for part in str(args.synthetic_months).replace(";", ",").split(",")
                if part.strip()
            ]
        except ValueError as error:
            raise ValueError(
                "--synthetic-months must be a comma separated list of months, e.g. 6,12"
            ) from error
    if args.synthetic_week is not None:
        payload["synthetic_week"] = int(args.synthetic_week)
    merged = OverrideConfig.from_dict(payload)
    if not merged.overrides and not merged.extend_synthetic_tenors:
        return None
    return merged


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
