"""Imply the borrow curve from futures + listed options.

``python -m surface_pricer build-borrow-curve`` pulls one listed-option
snapshot (CFFEX index underlyings), takes forwards from the futures (put/call
parity as fallback), implies borrow rates against the CNY FR007 curve built by
``build-ir-curve`` and extends the tail with the edslib OU model up to 3Y.

The resulting JSON is consumed by the pricing layer through
:class:`surface_pricer.core.curves.PiecewiseRateCurve`.

The curve is filed **per index** (``borrow_curve/latest.json`` is a map): a MO run
lands under ``000852``, a 510500 run under ``510500``, and a valuation follows the
entry of *its* index - see :mod:`surface_pricer.io.curve_runs`.

Quick run: edit the ``QUICK_DEFAULTS`` block below and execute the file (or
``python -m surface_pricer build-borrow-curve``) **without arguments** - e.g. with
the IDE "Run" button; the block picks which underlying to build.  Any command line
argument disables the block and the usual CLI rules apply again.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # IDE "Run" launches the file directly, with no package context; put the
    # repository root on the path and hand control to the package module
    # (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.build_borrow_curve import main

    raise SystemExit(main())

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from ..core.borrow_curve import (
    DEFAULT_EXTENSION_YEARS,
    BorrowCurvePillars,
    build_borrow_curve,
    extend_borrow_tail,
)
from ..core.daycount import DateHelperBusinessCalendar
from ..core.ir_curve import IRCurvePillars, build_fr007_curve
from ..io.curve_runs import (
    BORROW_CURVE,
    IR_CURVE,
    LATEST_NAME,
    curve_details,
    curve_root,
    record_curve_run,
    resolve_curve_path,
)
from ..marketdata.forwards import forwards_from_snapshot
from ..marketdata.gateway import QuoteGatewaySnapshotClient
from ..marketdata.providers import QuoteApiDataProvider
from ..marketdata.rate_inputs import SAMPLE_CSV, parse_interest_rate_csv
from ._common import env, load_env_files, reporter_for
from .fit_surface import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_USER

# ---------------------------------------------------------------------------
# Quick-run defaults
#
# Edit this block, then run the file (or ``python -m surface_pricer
# build-borrow-curve``) with **no arguments** - e.g. with the IDE "Run" button.
# Passing any command line argument disables the block and the usual CLI rules
# (and defaults) apply again, so scripts are unaffected.
#
# The whole flag surface is listed here: the keys are argparse **dest** names,
# i.e. underscores - ``"output_root"``, never ``"output-root"`` - and
# :func:`_apply_quick_defaults` refuses an unknown one instead of silently
# setting an attribute nobody reads.
# ---------------------------------------------------------------------------
QUICK_DEFAULTS: Dict[str, Any] = {
    # ---- 拟合哪一个标的（曲线按**指数**归档）-----------------------------------
    "underlying": "MO",           # MO / IO / HO（CFFEX 指数期权）或 510500 / 588000 / 159915（ETF 期权）
    "index": None,                # 覆盖指数代码（None -> registry）；给 ETF 期权拟合别的指数时写，如 "000905.SH"
    # ---- 利率曲线（隐含 borrow 的基准）----------------------------------------
    "ir_curve": "latest",         # latest（最新一次 build-ir-curve）/ 具体文件 / "none"（从 --rate-file 重新 bootstrap）
    "rebuild_ir_curve": False,    # True = 忽略缓存曲线，强制从 CSV 重新 bootstrap
    "rate_file": str(SAMPLE_CSV), # 利率导出兜底 CSV（ir_curve 为 none / rebuild 时使用）
    "keep_constant_columns": False,
    # ---- borrow 曲线构造 -----------------------------------------------------
    "horizon_years": DEFAULT_EXTENSION_YEARS,  # OU 尾部外推年数
    "min_days_to_expiry": 3,      # 跳过距今太近的远期（天，edslib 口径）
    "no_extend": False,           # True = 不做 OU 尾部外推
    # ---- 输出与运行 ----------------------------------------------------------
    "out": None,                  # 写死某个文件（跳过 run 记账）；None = 正常按指数归档到 borrow_curve/
    "output_root": None,          # 输出根（None -> surface_pricer/output）
    "env_file": None,             # 额外的 .env（凭据）
    "quiet": False,
}


def _apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Overwrite the parsed arguments with :data:`QUICK_DEFAULTS`.

    The keys are argparse **dest** names (underscores: ``"output_root"``).  A
    mistyped one - ``"output-root"`` - would otherwise just set an attribute
    nobody reads, i.e. the block would look configured and quietly do nothing, so
    it is refused with the spelled-out hint instead.
    """
    unknown = [key for key in QUICK_DEFAULTS if not hasattr(args, key)]
    if unknown:
        raise ValueError(
            "QUICK_DEFAULTS has unknown key(s): {} - these are argparse dest names "
            "('output_root', not 'output-root')".format(", ".join(sorted(unknown)))
        )
    for key, value in QUICK_DEFAULTS.items():
        setattr(args, key, value)
    return args


def _no_cli_arguments() -> bool:
    """True when the process was started with no arguments (IDE Run button)."""
    return len(sys.argv) <= 1


def main(argv: Optional[Iterable[str]] = None, *, quick: bool = False) -> int:
    args = _parse_args(argv)

    if quick or (argv is None and _no_cli_arguments()):
        try:
            args = _apply_quick_defaults(args)
        except ValueError as error:  # a typo in the block must not pass silently
            print("ERROR: {}".format(error))
            return 2

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
    try:
        rate_curve, rate_pillars = _load_rate_curve(args, reporter)
    except ValueError as error:
        print("ERROR: {}".format(error))
        return 2
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
        # the snapshot carries the same curve the borrow is implied against, so
        # anything rate-dependent inside the provider sees the real one
        provider = QuoteApiDataProvider(client, rate_curve=rate_curve)
        spec = provider.spec_for(underlying)
        if args.index:
            from dataclasses import replace

            spec = replace(spec, index_ticker=str(args.index))
            provider = QuoteApiDataProvider(
                client, rate_curve=rate_curve, spec_overrides={underlying: spec}
            )
        snapshot = provider.load(underlying)
    # the borrow curve belongs to the **index** the venue references (MO -> 000852):
    # that is the entry this run is filed under, exactly like the fit runs' pointer
    index = str(getattr(spec, "index_ticker", None) or underlying)

    reporter(
        "snapshot   | {} | index {} | valuation {} | spot {:.4f} | {} option quotes "
        "| {} future expiry(ies)".format(
            underlying,
            index,
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

    if args.out:
        target = Path(args.out)
        pillars.to_json(str(target))
        print("written  : {}".format(target))
        return 0
    try:
        target = record_curve_run(
            BORROW_CURVE,
            pillars,
            output_root=args.output_root,
            details=curve_details(pillars),
            index=index,
        )
    except OSError as error:
        print("ERROR: cannot write the curve run: {}".format(error))
        return 2
    print("written  : {}".format(target))
    print(
        "latest   : {}".format(
            curve_root(BORROW_CURVE, args.output_root, index=index) / LATEST_NAME
        )
    )
    return 0


def _load_rate_curve(
    args: argparse.Namespace,
    reporter,
) -> Tuple[object, IRCurvePillars]:
    """The IR curve to imply borrow against: a stored run, or a fresh bootstrap.

    ``--ir-curve latest`` (default) or a path loads a curve run; ``none`` - or
    ``--rebuild-ir-curve`` - bootstraps from the rate file instead.  A run that
    cannot be found is an error: silently bootstrapping would imply borrow off a
    different curve than the one the caller asked for.
    """
    path = resolve_curve_path(IR_CURVE, args.ir_curve, output_root=args.output_root)
    if path is not None and not args.rebuild_ir_curve:
        pillars = IRCurvePillars.from_json(path)
        reporter("rate curve | loaded {} ({})".format(path.name, path.parent))
        return pillars.to_piecewise_curve(), pillars
    if args.rebuild_ir_curve:
        reporter("rate curve | rebuilding from {} (--rebuild-ir-curve)".format(args.rate_file))
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
        default="latest",
        help=(
            "ir_curve run to imply borrow against: 'latest' (default) / a path / "
            "'none' (bootstrap from --rate-file)"
        ),
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
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "write exactly this file, no run bookkeeping (default: a new stamped "
            "run under <output-root>/borrow_curve/, with latest.json pointing at it)"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="output root (default: surface_pricer/output; runs live in its borrow_curve/)",
    )
    parser.add_argument("--env-file", default=None, help="additional .env file with credentials")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["main"]
