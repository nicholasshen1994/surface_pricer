"""Price a resolved contract payload and report the Greeks that were asked for.

Examples::

    python -m surface_pricer price-json --spec contract.json --greeks delta,gamma_cash
    python -m surface_pricer price-json --spec quote.json --greeks all --json
    python -m surface_pricer price-json --spec - --method pde < contract.json

``--spec`` takes what ``build-json`` writes - ``autocall_schedule`` /
``vanilla_spec``, the ``--product`` block of its ``QUICK_DEFAULTS`` (§15) - or
one of the whole ``--json`` quotes, which wrap the payload under ``contract``.  The
payload's ``kind`` picks the pricer, ``--fit`` supplies the market, and the Greeks
come from ``--greeks`` or, when the flag is absent, from a ``"greeks"`` list in
the payload itself.  Nothing is resolved from a term sheet: what the JSON says is
what gets priced.

``--slide`` turns the same run into a **spot ladder**: the trade is repriced on
every rung (``--slide-range`` / ``--slide-step`` / ``--slide-spots``) and the
requested Greeks are reported per rung, which is how a book's delta / gamma is
read across the barrier region.  ``--csv`` writes the ladder for a spreadsheet.

Quick run: edit the ``QUICK_DEFAULTS`` block below and execute the file (or
``python -m surface_pricer price-json``) **without arguments** - e.g. the IDE
"Run" button; ``python -m surface_pricer slide`` with no arguments uses the same
block and keeps its ladder mode.  Any command line argument disables the block
and the usual CLI rules apply again.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # IDE "Run" launches the file directly, with no package context; put the
    # repository root on the path and hand control to the package module
    # (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.price_json import main

    raise SystemExit(main())

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..io.fit_runs import list_runs, resolve_run
from ..pricing.exotics.autocall import AutocallSchedule
from ..pricing.exotics.autocall.mc import AutocallMonteCarlo
from ..pricing.exotics.autocall.pde import AutocallPDE
from ..pricing.results import RiskSettings
from ..pricing.risk.diff import ALL_GREEK_NAMES, BUCKET_NAMES, GREEK_CONVENTION, parse_greeks
from ..pricing.risk.slide import DEFAULT_SPAN, DEFAULT_STEP, run_slide, spot_ladder
from ..pricing.vanilla import VanillaPricer, VanillaSpec, calculate_greeks_spec
from ..reporting.autocall_report import autocall_to_dict, format_autocall
from ..reporting.quote_report import format_quote, num, quote_to_dict
from ..reporting.slide_report import format_slide, slide_csv, slide_to_dict
from ._common import echo_payload, load_payload, quick_output_path
from ._market import (
    market_from_run,
    note_table_cache,
    risk_settings,
    table_cache_for,
    warn_underlying_mismatch,
)


# ---------------------------------------------------------------------------
# Quick-run defaults
#
# Edit this block, then run the file (or ``python -m surface_pricer price-json``)
# with **no arguments** - e.g. with the IDE "Run" button.  Passing any command
# line argument disables the block and the usual CLI rules (and defaults) apply
# again, so scripts are unaffected.  ``python -m surface_pricer slide`` with no
# arguments reads the block too, but keeps its ladder mode.
#
# The whole CLI surface is listed here - every ``--help`` flag, in the order the
# parser groups them.  The keys are argparse **dest** names, i.e. underscores -
# ``"output_root"``, never ``"output-root"`` - and :func:`_apply_quick_defaults`
# refuses an unknown one instead of silently setting an attribute nobody reads.
# ---------------------------------------------------------------------------
QUICK_DEFAULTS: Dict[str, Any] = {
    # ---- 载荷与 fit run ------------------------------------------------------
    "payload": r"C:\Code\edslib\surface_pricer\output\autocall.json",  # 载荷的**绝对路径**：按原样读，不做任何查找，读不到就报错；"-" 读 stdin
    "fit": "latest",             # 用哪次拟合：'latest' / run 名（唯一前缀即可）/ run 目录 / surface.json
    "output_root": r"C:\Code\edslib\surface_pricer\output",  # 输出根（绝对路径最稳）：fit run 在它的 vol_fit/，曲线在 ir_curve/ 与 borrow_curve/；None -> 包内 output
    "list_runs": False,          # True = 只列出已存 fit run 再退出（模式开关，一般从命令行给）
    # ---- 风险：要算什么 ------------------------------------------------------
    "greeks": "delta_cash,gamma_cash",             # 逗号分隔：all / none / delta / delta_cash / delta_n / gamma / gamma_cash / vega / theta / rho / rhoq / buckets / volga / vanna（后两个是二阶交叉差分，雪球与香草都支持、**不进 all**、需显式点名）；None -> 用载荷自带的 "greeks" 列表，都没有就只算 NPV。默认 none = 只出 NPV，要希腊值再打开
    "method": "pde",              # 雪球引擎：pde（默认，确定性，希腊值首选）/ mc
    "paths": None,                # MC 路径数（None -> 引擎默认；生产按 §12.6 的收敛表给）；价格与希腊值共用这一份路径（2026-10 起没有单独的 greek_paths）
    "seed": None,                 # MC 种子（None -> 确定性 Sobol 默认）
    "pde_nodes": None,            # PDE 现货网格节点数（None -> 引擎默认 601；试算可用 201）
    "pde_theta": None,            # PDE 时间格式权重：1.0 全隐式 / 0.5 Crank-Nicolson（None -> 默认）
    "theta_days": None,           # theta 的 bump 天数（None -> 1 天）
    "full_bucket_grid": False,    # True = 分桶按曲线每个 pillar 各 bump 一次（更细更慢，2026-10 之前的口径）
    "local_vol_cache": True,      # True = 局部波动率表走磁盘缓存（**按指数分档**：output/local_vol/<指数>/lv_<hash>.json）：命中直接读表（文件里记着生成它的曲面/曲线/时间网格/现货锚与生成时间 —— ir/borrow/vol/spot 四项任一变化即另一个 key）；现货锚与文件不一致（或文件被改过）视为未命中并重建；False = 每次重建。greeks / slide 共用同一张表（锚点=基准 spot）
    # ---- 市场覆盖（None / "none" = 用 run 的对应值）---------------------------
    "spot": None,                 # 覆盖 run 的现货（也是梯子的基准档；障碍锚 spot0 不动，§14）
    "rate": None,                 # 覆盖平坦利率（ir_curve 为 "none" 时生效）
    "borrow": None,               # 平坦融券率（只在 borrow_curve 为 "none" 时生效）；None = 不覆盖（用 run 的，一般没有 -> 0）
    "ir_curve": "latest",         # 利率曲线：latest（最新一次 build-ir-curve）/ 具体文件路径 / "none"（用 --rate 平值）
    "borrow_curve": "latest",     # 融券曲线：latest / 具体文件路径 / "none"（用 --borrow 平值）
    "valuation_date": None,       # 覆盖估值日（None -> run 的估值日）
    "calendar_file": None,        # 带节假日的日历 JSON（None -> run 的日历）
    # ---- 当日判定口径（§12.3 第 8 条）---------------------------------------
    "trigger_basis": "effective",  # "contractual"（默认）= 真实条款线判今天（EOD / 入账）；"effective" = 位移后生效线判今天（盘中：spot 在两线之间时仍算未敲入，希腊值停在模型状态）
    # ---- 现价梯子（slide，§14）---------------------------------------------
    "slide": False,               # True = 铺 spot 梯子（等价于 slide 子命令）；只有 slide 子命令无参运行时这一项被忽略（模式由子命令给）
    "slide_range": None,          # 梯子半幅，'0.30' / '30%'（None -> 30%）
    "slide_step": None,           # 梯子步长，'0.05' / '5%'（None -> 5%）
    "slide_spots": None,          # 显式绝对档位，逗号分隔（写成 list/tuple 也行，会自动拼）；给了就忽略 range/step
    "progress": False,            # True = 每档往 stderr 打进度（长梯子建议开）
    # ---- 输出 ---------------------------------------------------------------
    "json": False,                # True = 打机器可读 JSON（含 contract / effective / greeks，可整份回喂）
    "csv": "slide.csv",                  # 梯子的 CSV（"-" 打到 stdout）；相对路径按包目录解析，None = 不写
}


def _apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Overwrite the parsed arguments with :data:`QUICK_DEFAULTS`.

    The keys are argparse **dest** names (underscores: ``"output_root"``).  A
    mistyped one - ``"output-root"`` - would otherwise just set an attribute
    nobody reads, i.e. the block would look configured and quietly do nothing,
    so it is refused with the spelled-out hint instead.  ``csv`` is resolved
    against the package (:func:`...apps._common.quick_output_path`), because the
    cwd of an IDE Run button is anyone's guess.
    """
    unknown = [key for key in QUICK_DEFAULTS if not hasattr(args, key)]
    if unknown:
        raise ValueError(
            "QUICK_DEFAULTS has unknown key(s): {} - these are argparse dest names "
            "('output_root', not 'output-root')".format(", ".join(sorted(unknown)))
        )
    for key, value in QUICK_DEFAULTS.items():
        if key == "slide" and args.slide:
            # the ``slide`` subcommand injects --slide: the mode belongs to the
            # caller, the block only fills in what to price
            continue
        setattr(args, key, value)
    if isinstance(args.slide_spots, (list, tuple)):
        # the block is nicer to edit as a list; the CLI wants one comma string
        args.slide_spots = ",".join(str(value) for value in args.slide_spots)
    args.csv = quick_output_path(args.csv)
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

    if args.list_runs:
        runs = list_runs(args.output_root)
        if not runs:
            print("no fit run found; run 'python -m surface_pricer fit' first")
            return 0
        for run in runs:
            print(run.describe())
        return 0

    if not args.payload:
        print("ERROR: a contract payload is required, e.g. price-json contract.json")
        return 2

    # The path is taken literally: no cwd / package / output fallback.  A wrong
    # path fails here, with the path it actually tried, instead of silently
    # pricing some other file that happened to share the name.
    try:
        payload = load_payload(args.payload)
    except (OSError, json.JSONDecodeError) as error:
        print("ERROR: cannot read {}: {}".format(args.payload, error))
        return 2
    if not isinstance(payload, Mapping):
        print("ERROR: {} does not hold a JSON object".format(source))
        return 2

    try:
        selection = parse_greeks(
            args.greeks if args.greeks is not None else _payload_greeks(payload)
        )
    except ValueError as error:  # bad --greeks selection
        print("ERROR: {}".format(error))
        return 2

    # ``--fit latest`` is per underlying: the payload says which index it is written
    # for, and that is the pointer it follows (a 000852 quote can no longer land on
    # a 000300 surface because that run happened to be newer).
    wanted = _payload_underlying(payload)
    try:
        run = resolve_run(args.fit, output_root=args.output_root, underlying=wanted)
    except ValueError as error:
        print("ERROR: {}".format(error))
        return 2
    warn_underlying_mismatch(run, wanted)

    try:
        market = market_from_run(run, args)
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as error:
        print("ERROR: cannot build the market state from {}: {}".format(run.name, error))
        return 2

    kind = _kind(payload)
    settings = risk_settings(args, selection)

    try:
        if args.slide:
            return _slide(payload, market, settings, run, args)
        if kind == "vanilla_spec":
            return _price_vanilla(payload, market, settings, run, args)
        if kind == "autocall_schedule":
            return _price_autocall(payload, market, settings, run, args)
        print(_unknown_kind(kind, payload))
        return 2
    except (ValueError, OSError) as error:
        print("ERROR: {}".format(error))
        return 2


def _payload_underlying(payload: Mapping[str, Any]) -> str:
    """The index a payload is written for (a quote wraps it under ``contract``)."""
    inner: Mapping[str, Any] = payload
    wrapped = payload.get("contract")
    if isinstance(wrapped, Mapping):
        inner = wrapped
    return str(inner.get("underlying") or "").strip()


def _unknown_kind(kind: str, payload: Mapping[str, Any]) -> str:
    inner = payload.get("contract")
    if not isinstance(inner, Mapping):
        inner = payload
    if not str(inner.get("kind", "") or "").strip():
        return (
            "ERROR: the payload has no 'kind': it must say 'vanilla_spec' or "
            "'autocall_schedule' (a hand-written file has to declare which pricer "
            "reads it; see the design doc, section 13)"
        )
    return (
        "ERROR: unknown payload kind {!r}: expected 'vanilla_spec' or "
        "'autocall_schedule' (see the design doc, section 13)".format(kind)
    )


# ------------------------------------------------------------------- pricers
def _price_vanilla(
    payload: Mapping[str, Any],
    market: Any,
    settings: RiskSettings,
    run: Any,
    args: argparse.Namespace,
) -> int:
    spec = VanillaSpec.from_dict(payload, market)
    if settings.greeks:
        result = calculate_greeks_spec(spec, market, settings)
    else:  # NPV only: a single analytic valuation, no bumps at all
        result = VanillaPricer(market).price_spec(spec)

    if args.json:
        print(
            json.dumps(
                quote_to_dict(result, run=run.name, spec=spec), indent=2, default=str
            )
        )
    else:
        echo_payload(spec.to_dict())
        print()
        print(format_quote(result, run=run.describe(), spec=spec))
    return 0


def _price_autocall(
    payload: Mapping[str, Any],
    market: Any,
    settings: RiskSettings,
    run: Any,
    args: argparse.Namespace,
) -> int:
    schedule = AutocallSchedule.from_dict(
            payload, market, trigger_basis=getattr(args, "trigger_basis", None)
        )
    table_cache = table_cache_for(args, market, run)
    engine = _engine(args, table_cache)
    if settings.greeks:
        result = engine.greeks_schedule(schedule, market, settings)
    else:
        result = engine.price_schedule(schedule, market, settings)
    note_table_cache(table_cache)

    if args.json:
        print(json.dumps(autocall_to_dict(schedule, result), indent=2, default=str))
    else:
        echo_payload(schedule.to_dict())
        print()
        print(
            format_autocall(schedule, result, market, with_greeks=bool(settings.greeks))
        )
    return 0


# ------------------------------------------------------------------ spot slide
def _slide(
    payload: Mapping[str, Any],
    market: Any,
    settings: RiskSettings,
    run: Any,
    args: argparse.Namespace,
) -> int:
    """A spot ladder: the same trade repriced on every rung (``--slide``).

    Every rung is a **full repricing on the bumped market state** - the trade never
    moves (an absolute strike, barriers anchored on ``spot0``), so the ladder reads
    "what the book is worth there, and what its Greeks are there".  Only the
    parallel Greeks make sense in a ladder; the bucketed ones are dropped with a
    note.  The Greeks come from ``--greeks`` exactly as in a single quote.
    """
    kind = _kind(payload)
    if kind not in ("vanilla_spec", "autocall_schedule"):
        print(_unknown_kind(kind, payload))
        return 2
    csv_target = None if args.csv is None else str(args.csv).strip()
    if csv_target in {"-", "stdout"} and args.json:
        print("ERROR: --json and '--csv -' both write to stdout; pick one")
        return 2
    if args.slide_spots is not None and (
        args.slide_range is not None or args.slide_step is not None
    ):
        # explicit rungs win *or* the span does - never both (it used to be a
        # silent "spots win", which reads as if the range had been applied)
        print(
            "ERROR: --slide-spots and --slide-range/--slide-step are mutually "
            "exclusive: give the explicit rungs or the span, not both"
        )
        return 2

    greeks = _slide_selection(settings)
    settings = replace(settings, greeks=greeks)
    # one table for the whole ladder (the rungs differ only in spot, and the anchor
    # is pinned to the base: every rung asks for the same coefficients)
    table_cache = (
        table_cache_for(args, market, run) if kind == "autocall_schedule" else None
    )
    span = _fraction(args.slide_range)
    step = _fraction(args.slide_step)
    try:
        ladder = spot_ladder(
            market.spot, span=span, step=step, spots=_spots(args.slide_spots)
        )
    except ValueError as error:
        print("ERROR: {}".format(error))
        return 2

    if kind == "vanilla_spec":
        spec = VanillaSpec.from_dict(payload, market)
        contract = spec.to_dict()
        contract_line = _vanilla_line(spec)
        method = "analytic"

        def price_at(moved):
            spec_at = spec.rebased(moved)
            if greeks:
                return calculate_greeks_spec(spec_at, moved, settings)
            return VanillaPricer(moved).price_spec(spec_at)

    else:
        schedule = AutocallSchedule.from_dict(
            payload, market, trigger_basis=getattr(args, "trigger_basis", None)
        )
        contract = schedule.to_dict()
        contract_line = _autocall_line(schedule)
        method = str(args.method or "pde").strip().lower()
        engine = _engine(args, table_cache)

        def price_at(moved):
            schedule_at = schedule.rebased(moved)
            if greeks:
                return engine.greeks_schedule(schedule_at, moved, settings)
            return engine.price_schedule(schedule_at, moved, settings)

    def on_rung(index: int, level: float) -> None:
        if args.progress:
            print(
                "slide rung {}/{}: spot={}".format(index + 1, len(ladder), num(level, 8)),
                file=sys.stderr,
                flush=True,
            )

    rows = run_slide(
        market, price_at=price_at, greeks=greeks, ladder=ladder, on_rung=on_rung
    )
    if table_cache is not None:
        note_table_cache(table_cache)

    base_spot = float(market.spot)
    explicit = args.slide_spots is not None
    lines = [
        "slide      : {} rungs{} | greeks={}".format(
            len(ladder),
            (
                ""
                if explicit
                else " | span=+-{:.2%} | step={:.2%}".format(
                    DEFAULT_SPAN if span is None else span,
                    DEFAULT_STEP if step is None else step,
                )
            ),
            ",".join(greeks) if greeks else "npv only",
        ),
        "fit run    : {}".format(run.describe()),
        contract_line,
        "market     : base spot={} | method={}".format(num(base_spot, 8), method),
    ]
    conventions = [
        "{}: {}".format(name, GREEK_CONVENTION[name])
        for name in greeks
        if name in GREEK_CONVENTION
    ]
    if conventions:  # one per line: the units matter and the lines stay readable
        lines.append("convention : {}".format(conventions[0]))
        lines.extend(" " * len("convention : ") + item for item in conventions[1:])

    if csv_target is not None and csv_target not in {"-", "stdout"}:
        Path(csv_target).write_text(slide_csv(rows, greeks), encoding="utf-8")
    if args.json:
        print(
            json.dumps(
                slide_to_dict(
                    rows,
                    run=run.name,
                    contract=contract,
                    base_spot=base_spot,
                    greeks=greeks,
                    span=None if explicit else (DEFAULT_SPAN if span is None else span),
                    step=None if explicit else (DEFAULT_STEP if step is None else step),
                    method=method,
                ),
                indent=2,
                default=str,
            )
        )
    elif csv_target in {"-", "stdout"}:
        print(slide_csv(rows, greeks), end="")
    else:
        print(format_slide(rows, greeks, lines=lines))
        if csv_target is not None:
            print()
            print("slide      : wrote {}".format(csv_target))
    return 0


def _slide_selection(settings: RiskSettings) -> Tuple[str, ...]:
    """The Greeks a ladder reports, in report order: the parallel ones only.

    A bucketed Greek is a bump pair *per bucket*: inside a ladder that is a table
    per rung for no extra insight, so the buckets are dropped with a note (a plain
    quote still has them).
    """
    selected = parse_greeks(settings.greeks)
    dropped = [name for name in selected if name in BUCKET_NAMES]
    if dropped:
        print(
            "note: a slide reports parallel Greeks only; dropped {} - use a plain "
            "quote for the buckets".format(", ".join(dropped)),
            file=sys.stderr,
        )
    return tuple(
        name for name in ALL_GREEK_NAMES if name in selected and name not in dropped
    )


def _fraction(text: Optional[str]) -> Optional[float]:
    """``0.3`` / ``30%`` -> ``0.3`` (a bare number above 1 is read as a percentage)."""
    if text is None:
        return None
    value = str(text).strip()
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    number = float(value)
    return number / 100.0 if abs(number) > 1.0 else number


def _spots(text: Optional[str]) -> Optional[Tuple[float, ...]]:
    """``"6900, 7500"`` -> absolute spot levels (``None`` when not given)."""
    if text is None:
        return None
    values = tuple(float(item) for item in str(text).replace(",", " ").split())
    if not values:
        raise ValueError("--slide-spots is empty")
    return values


def _vanilla_line(spec: VanillaSpec) -> str:
    return "contract   : vanilla_spec | {} | strike={} | expiry={} | notional={}".format(
        spec.underlying or "-",
        num(spec.strike, 8),
        spec.expiry_date.date().isoformat(),
        num(spec.notional, 8),
    )


def _autocall_line(schedule: AutocallSchedule) -> str:
    """The contract block of a ladder: the barriers are what the rungs move against."""
    ko = schedule.ko_levels[0] if schedule.ko_levels else None
    ki = schedule.ki_levels[0] if schedule.ki_levels else None
    return (
        "contract   : autocall_schedule | {} | notional={} | spot0={} | {} obs"
        " | ko={} | ki={}".format(
            schedule.underlying or "-",
            num(schedule.notional, 8),
            num(schedule.spot0, 8),
            len(schedule.observation_dates),
            num(ko, 8),
            num(ki, 8),
        )
    )


# --------------------------------------------------------------------- payload
def _kind(payload: Mapping[str, Any]) -> str:
    """Which pricer the payload asks for (a quote wraps it under ``contract``).

    ``kind`` is **required** (2026-10): the field-sniffing this used to do
    (``spot0`` + ``observations`` -> a snowball) meant a hand-written file could be
    priced as something it never said it was.  A blank string comes back when the
    key is missing - every caller reports that as an error.
    """
    inner: Mapping[str, Any] = payload
    wrapped = payload.get("contract")
    if isinstance(wrapped, Mapping):
        inner = wrapped
    return str(inner.get("kind", "")).strip().lower()


def _payload_greeks(payload: Mapping[str, Any]) -> Optional[Any]:
    """A ``"greeks"`` selection stored next to the contract, if there is one.

    Only a string (``"delta,vega"``) or a list counts: a ``--json`` *quote* also
    has a ``greeks`` key, but there it is the result table (a mapping), which must
    not be read as a selection.
    """
    for candidate in (payload, payload.get("contract")):
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get("greeks")
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    return None


def _engine(args: argparse.Namespace, table_cache: Any = None):
    """The engine for an autocall payload (the shift is already in the JSON)."""
    if str(args.method).strip().lower() in ("pde", "fd", "fdm"):
        return AutocallPDE(local_vol_cache=table_cache)
    return AutocallMonteCarlo(local_vol_cache=table_cache)


# ------------------------------------------------------------------------- CLI
def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="surface_pricer price-json",
        description=(
            "Price a resolved contract payload (vanilla_spec / autocall_schedule) "
            "on a stored fit run and report the Greeks asked for."
        ),
    )
    parser.add_argument(
        "payload",
        nargs="?",
        default=None,
        metavar="FILE",
        help=(
            "resolved contract JSON (build-json output, or a whole --json quote). "
            "The path is read exactly as given - an absolute path is the reliable "
            "spelling; nothing is searched for.  '-' reads stdin"
        ),
    )
    parser.add_argument("--fit", default="latest", help="'latest' / run name / directory / surface.json")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "output root: None -> surface_pricer/output; the fit runs live in its "
            "vol_fit/, the curve runs in its ir_curve/ and borrow_curve/"
        ),
    )
    parser.add_argument("--list-runs", action="store_true", help="list stored fit runs and exit")

    risk = parser.add_argument_group("risk")
    risk.add_argument(
        "--greeks",
        default=None,
        metavar="LIST",
        help=(
            "Greeks to compute, comma separated: all / none / delta / delta_cash / "
            "gamma / gamma_cash / vega / theta / rho / rhoq / buckets / volga / "
            "vanna (the last two are second-order cross differences - opt-in on both "
            "the autocall and the vanilla side, and never part of 'all'); default: "
            "the payload's own 'greeks' list, else none"
        ),
    )
    risk.add_argument("--method", default="pde", help="autocall engine: pde (default) / mc")
    risk.add_argument("--paths", type=int, default=None, help="MC paths (price and greeks)")
    risk.add_argument("--seed", type=int, default=None)
    risk.add_argument("--pde-nodes", type=int, default=None)
    risk.add_argument("--pde-theta", type=float, default=None)
    risk.add_argument("--theta-days", type=int, default=None, help="theta bump in days")
    risk.add_argument(
        "--no-local-vol-cache",
        dest="local_vol_cache",
        action="store_false",
        default=True,
        help=(
            "rebuild the local-vol table every run instead of using "
            "output/local_vol/ (a hit loads the coefficients, a miss builds and "
            "stores them together with the surface, the curves and the build time)"
        ),
    )
    risk.add_argument(
        "--full-bucket-grid",
        action="store_true",
        help=(
            "bump one curve pillar per bucket instead of the coarse trade-aware "
            "grid (slower, finer: the pre-2026-10 behaviour)"
        ),
    )

    market = parser.add_argument_group("market")
    market.add_argument("--spot", type=float, default=None, help="override the run spot")
    market.add_argument("--rate", type=float, default=None, help="override the run rate")
    market.add_argument(
        "--borrow",
        type=float,
        default=None,
        help="flat borrow rate, used when --borrow-curve is 'none' (default: the run's, else 0)",
    )
    market.add_argument(
        "--ir-curve",
        default="latest",
        help="ir_curve run: 'latest' (default) / a path / 'none' (flat --rate)",
    )
    market.add_argument(
        "--borrow-curve",
        default="latest",
        help=(
            "borrow_curve run: 'latest' (default) = the newest run for the "
            "payload's index / a path / 'none' (flat --borrow)"
        ),
    )
    market.add_argument("--valuation-date", default=None, help="override the run valuation date")
    market.add_argument("--calendar-file", default=None, help="JSON file with calendar holidays")

    determination = parser.add_argument_group("same-day determination")
    determination.add_argument(
        "--trigger-basis",
        choices=("contractual", "effective"),
        default=None,
        help=(
            "which barrier today's fixing is read against: contractual (default; "
            "EOD, raw terms, what the ledger uses) or effective (intraday: the "
            "post-shift line the engines price, so a spot through the raw barrier "
            "but not the shifted one stays not knocked in)"
        ),
    )

    slide = parser.add_argument_group("slide")
    slide.add_argument(
        "--slide",
        action="store_true",
        help=(
            "price a spot ladder instead of one quote: every rung is the same trade "
            "repriced on that spot, with the --greeks selection at each rung"
        ),
    )
    # note: argparse formats help strings, so a literal '%' is written '%%'
    slide.add_argument(
        "--slide-range",
        default=None,
        metavar="X",
        help="half-range of the ladder ('0.30' / '30%%'; default 30%%)",
    )
    slide.add_argument(
        "--slide-step",
        default=None,
        metavar="X",
        help="ladder step ('0.05' / '5%%'; default 5%%)",
    )
    slide.add_argument(
        "--slide-spots",
        default=None,
        metavar="LIST",
        help="explicit absolute spot levels, comma separated (wins over range/step)",
    )
    slide.add_argument(
        "--progress",
        action="store_true",
        help="report every rung on stderr while the ladder runs (it is slow)",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true", help="print the quote as JSON")
    output.add_argument(
        "--csv",
        default=None,
        metavar="FILE",
        help="write the ladder as CSV ('-' prints it instead of the table)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["QUICK_DEFAULTS", "main"]


if __name__ == "__main__":
    sys.exit(main())
