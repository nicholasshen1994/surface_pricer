"""Imply the borrow curve from futures + listed options.

``python -m surface_pricer build-borrow-curve`` pulls one listed-option
snapshot (CFFEX index underlyings), takes forwards from the futures (put/call
parity as fallback), implies borrow rates against the CNY FR007 curve built by
``build-ir-curve`` and extends the tail with the edslib OU model up to 3Y.

The resulting JSON is consumed by the pricing layer through
:class:`surface_pricer.core.curves.PiecewiseRateCurve`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

from ..core.borrow_curve import (
    DEFAULT_EXTENSION_YEARS,
    BorrowCurvePillars,
    build_borrow_curve,
    extend_borrow_tail,
)
from ..core.daycount import DateHelperBusinessCalendar
from ..core.ir_curve import IRCurvePillars, build_fr007_curve
from ..marketdata.forwards import forwards_from_snapshot
from ..marketdata.gateway import QuoteGatewaySnapshotClient
from ..marketdata.providers import QuoteApiDataProvider
from ..marketdata.rate_inputs import SAMPLE_CSV, parse_interest_rate_csv
from ._common import env, load_env_files, reporter_for
from .fit_surface import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_USER

DEFAULT_IR_CURVE = Path(__file__).resolve().parent.parent / "output" / "ir_curve.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "borrow_curve.json"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    load_env_files(args.env_file)

    password = env("QUOTE_GATEWAY_PASSWORD", "CICC_QUOTE_PASSWORD")
    if not password:
        print(
            "ERROR: missing quote gateway password.\n"
            "       Put QUOTE_GATEWAY_PASSWORD=<password> into surface_pricer/.env "
            "or export the environment variable."
        )
        return 2
    host = env("QUOTE_GATEWAY_HOST", "CICC_QUOTE_API_HOST") or DEFAULT_HOST
    port = int(env("QUOTE_GATEWAY_PORT", "CICC_QUOTE_API_PORT") or DEFAULT_PORT)
    user = env("QUOTE_GATEWAY_USER", "CICC_QUOTE_USER") or DEFAULT_USER

    reporter = reporter_for(args.quiet)
    rate_curve, rate_pillars = _load_rate_curve(args, reporter)
    reporter(
        "rate curve | {} | valuation {} | {} pillar(s)".format(
            rate_pillars.curve_name,
            rate_pillars.valuation_date.isoformat(),
            len(rate_pillars.tenors),
        )
    )

    underlying = str(args.underlying).strip().upper()
    calendar = DateHelperBusinessCalendar("SHX")
    reporter("connecting to quote gateway {}:{} as {} ...".format(host, port, user))
    with QuoteGatewaySnapshotClient(host, port, user, password) as client:
        provider = QuoteApiDataProvider(client, rate=0.0)
        if args.index:
            from dataclasses import replace

            spec = replace(provider.spec_for(underlying), index_ticker=str(args.index))
            provider = QuoteApiDataProvider(
                client, rate=0.0, spec_overrides={underlying: spec}
            )
        snapshot = provider.load(underlying)

    reporter(
        "snapshot   | {} | valuation {} | spot {:.4f} | {} option quotes | {} future expiry(ies)".format(
            underlying,
            snapshot.valuation_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            snapshot.spot,
            len(snapshot.option_records),
            len(snapshot.future_price_by_expiry),
        )
    )

    forwards, sources = forwards_from_snapshot(snapshot, rate_curve)
    if not forwards:
        print("ERROR: no forwards available (no futures and no usable parity pairs)")
        return 2

    pillars = build_borrow_curve(
        snapshot.valuation_datetime.date(),
        snapshot.spot,
        forwards,
        rate_curve,
        forward_source=sources,
        calendar_name="SHX",
        min_days_to_expiry=int(args.min_days_to_expiry),
    )
    if not args.no_extend:
        pillars = extend_borrow_tail(
            pillars,
            extension_years=int(args.horizon_years),
            calendar=calendar,
        )

    lines = [
        "borrow   : {} | valuation {} | {} observed + {} tail pillar(s)".format(
            underlying,
            pillars.valuation_date.isoformat(),
            pillars.observed,
            pillars.extended,
        ),
        "forward source: {}".format(
            ", ".join(
                "{}={}".format(key, pillars.forward_source.get(key, "?"))
                for key in sorted(pillars.forwards)
            )
            or "-"
        ),
        "pillars  :",
    ]
    observed = set(pillars.forward_source)
    for pillar_date, rate in zip(pillars.pillar_dates, pillars.rates):
        label = pillar_date.isoformat()
        if label in observed:
            kind = pillars.forward_source.get(label, "observed")
        else:
            kind = "ou-tail"
        lines.append(
            "  {:>10} {:>6}d  borrow {:>8.4f}%  [{}]".format(
                label,
                (pillar_date - pillars.valuation_date).days,
                rate * 100.0,
                kind,
            )
        )
    if pillars.ou:
        lines.append(
            "ou model : kappa={kappa:.3f} mu={mu:.6f} f0={f0:.6f} anchored {} ({})".format(
                pillars.ou.get("anchor_date", "-"),
                "calibrated" if pillars.ou.get("calibrated") else "prior",
                **pillars.ou,
            )
        )
    for note in pillars.notes:
        lines.append("note     : {}".format(note))
    print("\n".join(lines))

    target = Path(args.out) if args.out else DEFAULT_OUTPUT
    pillars.to_json(str(target))
    print("written  : {}".format(target))
    return 0


def _load_rate_curve(
    args: argparse.Namespace,
    reporter,
) -> Tuple[object, IRCurvePillars]:
    """Use the cached curve JSON when present, otherwise bootstrap it."""
    path = Path(args.ir_curve)
    if path.is_file() and not args.rebuild_ir_curve:
        pillars = IRCurvePillars.from_json(path)
        reporter("rate curve | loaded {}".format(path))
    else:
        inputs = parse_interest_rate_csv(
            args.rate_file,
            drop_constant_columns=not args.keep_constant_columns,
        )
        pillars = build_fr007_curve(
            inputs.valuation_date,
            inputs.ir_swap,
            fr007=inputs.fr007,
            source=str(args.rate_file),
        )
        reporter("rate curve | bootstrapped from {}".format(args.rate_file))
    return pillars.to_piecewise_curve(), pillars


def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="surface_pricer build-borrow-curve")
    parser.add_argument("--underlying", default="MO", help="MO / IO / HO (CFFEX index options)")
    parser.add_argument("--index", default=None, help="override the index level ticker")
    parser.add_argument(
        "--ir-curve",
        default=str(DEFAULT_IR_CURVE),
        help="curve JSON produced by build-ir-curve (default: output/ir_curve.json)",
    )
    parser.add_argument(
        "--rebuild-ir-curve",
        action="store_true",
        help="ignore the cached curve JSON and bootstrap from the rate file",
    )
    parser.add_argument("--rate-file", default=str(SAMPLE_CSV), help="desk rate export fallback")
    parser.add_argument(
        "--keep-constant-columns",
        action="store_true",
        help="do not drop constant rate columns when bootstrapping",
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=DEFAULT_EXTENSION_YEARS,
        help="OU tail extension horizon in years (default 3)",
    )
    parser.add_argument(
        "--min-days-to-expiry",
        type=int,
        default=3,
        help="skip forwards closer than this many days (default 3, edslib convention)",
    )
    parser.add_argument("--no-extend", action="store_true", help="skip the OU tail extension")
    parser.add_argument("--out", default=None, help="output JSON (default: output/borrow_curve.json)")
    parser.add_argument("--env-file", default=None, help="additional .env file with credentials")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["main"]
