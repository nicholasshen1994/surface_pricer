"""Build a resolved contract payload - the JSON that ``price-json`` eats.

Generating a payload and pricing it are two jobs, and this app does only the
first: it resolves the terms in the ``QUICK_DEFAULTS`` block below into the
payload the pricing layer reads (``autocall_schedule`` / ``vanilla_spec``, design
doc section 13) and writes it into a folder.  ``price-json`` / ``slide`` do all
the pricing - nothing is valued here, so no surface is fitted and a payload comes
out in milliseconds where a quote takes minutes.

Edit the block and run the app with **no arguments** (or
``python -m surface_pricer build-json``) - e.g. from the IDE Run button - to write
the payload.  The command line only carries what a script wants to vary (product,
fit run, output location, the market overrides): the terms belong in the block, so
there is exactly one place to look at what is being generated.

The autocall block adds four fields to the plain snowball terms:

* ``guaranteed_period`` - whole months with **no observation**.  The first
  observation is the first grid date after ``start + guaranteed_period`` months,
  and it still accrues the coupon of the whole lock-up (accrual runs from the
  start date), so a 3-month guarantee pays three months of coupon the first time
  it can knock out - not one period.
* ``stepdown_size`` - the **per-observation** knock-out step: ``0.005`` means the
  KO drops 0.5% of the anchor at every observation (``ko_i = ko - (i-1) x step``).
  It is written as explicit per-observation levels and the packaged stepwise KO
  rule is switched off, so the term sheet's step and the house rule never stack.
* ``coupon`` - one annual rate, one per observation, or a **step schedule keyed by
  observation number**: ``{1: 0.19, 11: 0.10}`` = 0.19 from the 1st observation
  on, 0.10 from the 11th on.
* ``ki_strike`` - the strike of the knock-in loss leg (a ratio of the anchor).

Examples::

    python -m surface_pricer build-json                                    # the block
    python -m surface_pricer build-json --product vanilla --output-name var.json
    python -m surface_pricer build-json --fit MO_20260928_150000 --output-dir out2
    python -m surface_pricer build-json --json                             # also print it
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # IDE "Run" launches the file directly, with no package context; put the
    # repository root on the path and hand control to the package module
    # (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.build_json import main

    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.daycount import add_tenor, month_grid, shift_tenor, to_date, to_datetime
from ..io.fit_runs import list_runs, resolve_run
from ..marketdata.registry import get_underlying_spec
from ..pricing.exotics.autocall import AutocallContract, build_schedule
from ..pricing.rules import load_shift_config
from ..pricing.vanilla import VanillaContract, resolve_spec
from ..reporting.quote_report import num, pct
from ._common import dump_payload, quick_output_path
from ._market import market_from_run


# ---------------------------------------------------------------------------
# What to generate
#
# Edit this block and run the app with no arguments.  ``product`` picks which of
# the two blocks below is used; the keys inside a block are the contract's own
# fields (see the module docstring for the four autocall-specific ones), so the
# payload is exactly what is written here.
#
# Relative paths in this block - ``output_dir`` - are read from the **package**
# directory, not from the working directory (an IDE Run button's cwd is anyone's
# guess); an absolute path is honoured as written.
# ---------------------------------------------------------------------------
QUICK_DEFAULTS: Dict[str, Any] = {
    "product": "autocall",   # autocall / vanilla - which block below is used
    "fit": "latest",         # 'latest' / run name / directory / surface.json
    "output_dir": "output",  # where the payload goes (relative -> the package)
    "output_name": None,     # None -> "<product>.json"; '-' prints to stdout
    "json": False,           # True -> print the payload instead of the summary
    "spot": None,              # None -> the fit run's spot (the **current** index)
    "valuation_date": None,    # None -> the fit run's valuation date
    "autocall": {
        "underlying": '000852.SH',           # None -> the run's underlying
        "start": '2026-07-07',        # the trade's inception (None -> the valuation date)
        "tenor": "2Y",                # 3M / 1Y / 90D ... (used when expiry is None)
        "expiry": None,               # explicit YYYY-MM-DD, wins over tenor
        "obs_freq": "M",              # M / Q / S / A(Y) - grid stepped back from expiry
        "guaranteed_period": 2,       # months with no observation (0 = observe from month 1)
        "ko": 0.95,                   # knock-out, ratio of the anchor
        "stepdown_size": 0.005,         # per-observation KO step (0.005 = -0.5% each time)
        "ki": 0.65,                   # knock-in, ratio of the anchor
        "ki_frequency": "daily",     # daily / expiry / observation_dates
        "ki_strike": 1.0,             # strike of the KI loss leg, ratio of the anchor
        "ki_gearing": 1.0,            # loss multiplier
        "protection": 0.0,            # protected principal ratio
        "settlement_days": 0,         # payment lag: payment = observation + days
        "coupon": {1: 0.1296},        # rate / list / {**名义期数**: rate} —— 期数从起息日按月数，guaranteed_period **不改变编号**
        "rebate": 0.1296,               # annual rate of the no-KO / no-KI leg (None -> last coupon)
        "day_count": "act/365f",      # act/365f / act/360 / act/act
        "notional": 195200000,
        "start_spot": 8301.8,         # the inception index level the barriers are anchored on
        "no_shift": False,            # True -> write the raw terms, no barrier shift
        "ko_boundary": "inclusive",   # touching the KO level knocks out (">=" vs ">")
        "ki_boundary": "inclusive",   # touching the KI level knocks in ("<=" vs "<")
    },
    # "vanilla": {
    #     "underlying": None,           # None -> the run's underlying
    #     "strike": 7800.0,
    #     "strike_type": "absolute",    # absolute / percentage / fwd_percentage
    #     "tenor": "3M",                # used when expiry is None
    #     "expiry": None,               # explicit YYYY-MM-DD, wins over tenor
    #     "option_type": "call",        # call / put
    #     "notional": 1.0,
    # },
}

#: Observation-grid frequency -> months, counted back from the expiry.
_OBS_MONTHS = {"M": 1, "Q": 3, "S": 6, "A": 12, "Y": 12}


# ------------------------------------------------------------------------- main
def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_runs:
        runs = list_runs(args.output_root)
        if not runs:
            print("no fit run found; run 'python -m surface_pricer fit' first")
            return 0
        for run in runs:
            print(run.describe())
        return 0

    product = str(args.product or QUICK_DEFAULTS["product"]).strip().lower()
    if product not in ("autocall", "vanilla"):
        print(
            "ERROR: unknown product {!r}: expected autocall or vanilla".format(product)
        )
        return 2
    if args.fit is None:
        args.fit = QUICK_DEFAULTS["fit"]
    if args.spot is None:
        args.spot = QUICK_DEFAULTS.get("spot")
    if args.valuation_date is None:
        args.valuation_date = QUICK_DEFAULTS.get("valuation_date")

    terms = _terms(product)
    try:
        run = resolve_run(
            args.fit,
            output_root=args.output_root,
            underlying=[_index_key(terms), terms.get("underlying")],
        )
        market = market_from_run(run, args)
        if product == "autocall":
            payload, schedule, rolled = _autocall_payload(terms, market, run, args)
        else:
            payload, schedule, rolled = _vanilla_payload(terms, market, run)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("ERROR: {}".format(error))
        return 2

    target = _output_path(args, product)
    if target in {"-", "stdout"}:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    try:
        dump_payload(target, payload)
    except OSError as error:
        print("ERROR: cannot write {}: {}".format(target, error))
        return 2
    if _print_json(args):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            _summary(
                payload,
                product,
                terms,
                target,
                schedule,
                calendar_name=getattr(market.calendar, "name", None),
                rolled=rolled,
            )
        )
    return 0


def _terms(product: str) -> Dict[str, Any]:
    """The block of the chosen product (a copy: the block itself is never edited)."""
    block = QUICK_DEFAULTS.get(product)
    if not isinstance(block, Mapping):
        raise ValueError("QUICK_DEFAULTS[{}] must be a block of terms".format(product))
    return dict(block)


def _index_key(terms: Mapping[str, Any]) -> str:
    """The index a block prices - the key ``--fit latest`` is filed under.

    A block may name the *option venue* (``MO``) or the index itself
    (``000852.SH``); the run is filed under the cash index, because that is where
    the barriers, the spot and the payload live.  An unknown code is passed
    through unchanged (the caller hands both spellings to :func:`resolve_run`).
    """
    venue = str(terms.get("underlying") or "").strip()
    if not venue:
        return ""
    try:
        spec = get_underlying_spec(venue)
    except KeyError:
        return venue
    return spec.index_ticker or venue


# ------------------------------------------------------------------- autocall
def _autocall_payload(
    terms: Mapping[str, Any], market: Any, run: Any, args: argparse.Namespace
) -> Tuple[Dict[str, Any], Any, int]:
    """Resolve the autocall block into an ``autocall_schedule`` payload.

    Returns the payload, the effective schedule behind it and how many grid dates
    had to be rolled off a holiday.  The payload keeps the term-sheet levels plus
    the shift rule (what a reader edits), while the schedule carries the levels
    the engines actually see - the summary prints the latter.
    """
    start = (
        to_datetime(terms["start"])
        if terms.get("start")
        else to_datetime(market.valuation_date)
    )
    expiry = _autocall_expiry(terms, start, market)
    observations, rolled = _autocall_observations(terms, start, expiry, market)
    periods = len(observations)

    ko = float(terms["ko"])
    step = float(terms.get("stepdown_size") or 0.0)
    levels = tuple(ko - index * step for index in range(periods))
    if levels[-1] <= 0.0:
        raise ValueError(
            "stepdown_size={} takes the knock-out down to {:.4%} at the last "
            "observation - check ko and the number of periods".format(step, levels[-1])
        )

    start_spot = terms.get("start_spot")
    if start_spot is None:
        if to_date(start) != to_date(market.valuation_date):
            raise ValueError(
                "this payload starts on {} but is generated on {}: state start_spot, "
                "the inception spot the barriers are anchored on".format(
                    to_date(start).isoformat(), to_date(market.valuation_date).isoformat()
                )
            )
        start_spot = float(market.spot)

    contract = AutocallContract(
        underlying=str(terms.get("underlying") or run.underlying),
        start_date=start,
        expiry_date=expiry,
        observation_dates=observations,
        ko_levels=levels,
        ki_level=float(terms["ki"]),
        ki_frequency=str(terms.get("ki_frequency") or "daily"),
        ko_boundary=str(terms.get("ko_boundary") or "inclusive"),
        ki_boundary=str(terms.get("ki_boundary") or "inclusive"),
        ki_strike=float(terms.get("ki_strike") or 1.0),
        annual_coupon=_coupon_rates(
            terms.get("coupon", 0.0),
            periods,
            # the ladder is written in nominal periods: a guaranteed lock-up hides
            # observations from the contract but not numbers from the term sheet
            offset=_guarantee_offset(terms, start, expiry),
        ),
        rebate=terms.get("rebate"),
        day_count=str(terms.get("day_count") or "act/365f"),
        notional=float(terms.get("notional", 1.0)),
        protected_principal=float(terms.get("protection") or 0.0),
        ki_gearing=float(terms.get("ki_gearing") or 1.0),
        settlement_days=int(terms.get("settlement_days") or 0),
        start_spot=float(start_spot),
        # the term sheet's own step replaces the house knock-out rule rather than
        # adding to it: the levels above already carry it
        shift_override={"ko_shift": {"mode": "none"}} if step else None,
    )
    schedule = build_schedule(
        contract,
        market,
        shift_config=load_shift_config(args.shift_config),
        contractual=bool(terms.get("no_shift")),
    )
    return schedule.to_dict(), schedule, rolled


def _business_day(value: Any, market: Any):
    """A contractual date on the calendar: a holiday rolls to the next open day."""
    moment = to_datetime(value)
    calendar = getattr(market, "calendar", None)
    if calendar is None:
        return moment
    return to_datetime(calendar.next_business_day(moment))


def _autocall_expiry(terms: Mapping[str, Any], start: Any, market: Any):
    if terms.get("expiry"):
        expiry = _business_day(terms["expiry"], market)
    elif terms.get("tenor"):
        expiry = to_datetime(shift_tenor(start, str(terms["tenor"]), market.calendar))
    else:
        raise ValueError("the autocall block needs a tenor or an expiry")
    if expiry <= start:
        raise ValueError(
            "expiry {} must be after the start date {}".format(
                to_date(expiry).isoformat(), to_date(start).isoformat()
            )
        )
    return expiry


def _autocall_observations(
    terms: Mapping[str, Any], start: Any, expiry: Any, market: Any
) -> Tuple[Tuple[Any, ...], int]:
    """The observation grid: periodic, minus the guarantee, on the calendar.

    Every date is rolled **forward to the next business day** - an observation
    cannot fall on a holiday, and a monthly grid on the 9th lands on the Spring
    Festival sooner or later.  The guarantee is applied to the **contractual**
    dates first: an observation on the boundary of the lock-up is not observed just
    because a holiday would have pushed it out of the window.  The second element
    counts the dates that had to move, for the summary; the payload only carries
    the rolled ones.
    """
    calendar = getattr(market, "calendar", None)
    months = _observation_months(terms)
    grid = month_grid(expiry, months, after=start)

    guaranteed = int(terms.get("guaranteed_period") or 0)
    if guaranteed < 0:
        raise ValueError("guaranteed_period must not be negative")
    if guaranteed:
        cutoff = add_tenor(start, "{}M".format(guaranteed))
        grid = tuple(day for day in grid if day > cutoff)
    if not grid:
        raise ValueError(
            "no observation left: {} grid from {} to {} with guaranteed_period={} "
            "months".format(
                terms.get("obs_freq"),
                to_date(start).isoformat(),
                to_date(expiry).isoformat(),
                guaranteed,
            )
        )
    if calendar is None:
        return tuple(to_datetime(day) for day in grid), 0
    rolled = sum(1 for day in grid if not calendar.is_business_day(day))
    moved = tuple(sorted({calendar.next_business_day(day) for day in grid}))
    return tuple(to_datetime(day) for day in moved), rolled


def _observation_months(terms: Mapping[str, Any]) -> int:
    """The observation grid's step in months (``M`` -> 1, ``Q`` -> 3, ``S``/``A`` ...)."""
    return _OBS_MONTHS.get(str(terms.get("obs_freq") or "Q").strip().upper(), 3)


def _guarantee_offset(terms: Mapping[str, Any], start: Any, expiry: Any) -> int:
    """How many periods the guaranteed lock-up hides from the contract.

    The coupon ladder is counted on the **full** grid (:func:`_coupon_rates`), so
    this is the bridge between the two numberings: observation ``j`` is nominal
    period ``j + offset``.  Counting nominal is what keeps ``{1: 0.10, 13: None}``
    meaning "the first twelve months" on a trade that only starts observing in
    month 4 (2026-10) - a term sheet is written against the trade, not against the
    grid that survived the lock-up.
    """
    guaranteed = int(terms.get("guaranteed_period") or 0)
    if guaranteed <= 0:
        return 0
    grid = month_grid(expiry, _observation_months(terms), after=start)
    cutoff = add_tenor(start, "{}M".format(guaranteed))
    return sum(1 for day in grid if day <= cutoff)


def _coupon_rates(value: Any, periods: int, *, offset: int = 0) -> Tuple[float, ...]:
    """The annual coupon as one rate per **observation**, keyed by nominal period.

    A scalar is repeated; a list holds one rate per period of the **full** grid; a
    mapping is a **step schedule keyed by nominal period** - ``{1: 0.19, 11: 0.10}``
    = 0.19 from the 1st period on, 0.10 from the 11th on.  A schedule has to start at
    period 1, so the first periods are never left without a coupon by accident.

    ``offset`` is how many periods the guarantee hides (:func:`_guarantee_offset`):
    the keys keep counting the periods the trade *would* have had, so the number
    does not move when a lock-up is added, and a step that lands inside the lock-up
    simply applies from the first observed period on (the last key at or before it
    wins).  The list spelling counts the same way - it must cover the full grid.
    """
    if isinstance(value, Mapping):
        schedule = {int(key): float(rate) for key, rate in value.items()}
        if not schedule:
            raise ValueError("the coupon schedule is empty")
        keys = sorted(schedule)
        if keys[0] != 1:
            raise ValueError(
                "the coupon schedule must start at period 1, not {}".format(keys[0])
            )
        if keys[-1] > periods + offset:
            raise ValueError(
                "the coupon schedule names period {} but the contract has {} "
                "period(s) - guaranteed_period does not change the numbering".format(
                    keys[-1], periods + offset
                )
            )
        return tuple(
            schedule[max(key for key in keys if key <= index + offset)]
            for index in range(1, periods + 1)
        )
    if isinstance(value, (int, float)):
        return (float(value),) * periods
    rates = tuple(float(item) for item in value)
    if len(rates) != periods + offset:
        raise ValueError(
            "the coupon list has {} entries but the contract has {} period(s) - a "
            "guaranteed_period does not shorten the list either".format(
                len(rates), periods + offset
            )
        )
    return rates[offset:]


# -------------------------------------------------------------------- vanilla
def _vanilla_payload(
    terms: Mapping[str, Any], market: Any, run: Any
) -> Tuple[Dict[str, Any], Any, int]:
    """Resolve the vanilla block into a ``vanilla_spec`` payload."""
    if terms.get("expiry"):
        raw = to_datetime(terms["expiry"])
    elif terms.get("tenor"):
        raw = to_datetime(
            shift_tenor(market.valuation_date, str(terms["tenor"]), market.calendar)
        )
    else:
        raise ValueError("the vanilla block needs a tenor or an expiry")
    expiry = _business_day(raw, market)  # a holiday expiry rolls to the next open day
    rolled = 0 if to_date(expiry) == to_date(raw) else 1
    if expiry <= market.valuation_date:
        raise ValueError(
            "expiry {} is not after the valuation date {}".format(
                to_date(expiry).isoformat(), to_date(market.valuation_date).isoformat()
            )
        )
    contract = VanillaContract(
        expiry=expiry,
        strike=float(terms["strike"]),
        option_type=str(terms.get("option_type") or "call"),
        notional=float(terms.get("notional", 1.0)),
        strike_type=str(terms.get("strike_type") or "absolute"),
        underlying=str(terms.get("underlying") or run.underlying),
    )
    # resolving needs the market only for a relative strike; nothing is priced
    return resolve_spec(contract, market).to_dict(), None, rolled


# ---------------------------------------------------------------------- output
def _output_path(args: argparse.Namespace, product: str) -> str:
    """Where the payload goes; a relative path lands in the package, not the cwd."""
    name = (
        args.output_name
        if args.output_name is not None
        else QUICK_DEFAULTS.get("output_name")
    )
    name = "{}".format(name if name is not None else "{}.json".format(product))
    if name.strip() in {"-", "stdout"}:
        return name.strip()
    directory = (
        args.output_dir
        if args.output_dir is not None
        else QUICK_DEFAULTS.get("output_dir")
    )
    target = Path(name).expanduser()
    if not target.is_absolute() and target.parent == Path("."):
        # only a bare file name takes the block's folder; a path with a directory
        # is what the caller asked for
        base = Path(str(directory if directory is not None else ".")).expanduser()
        target = base / target
    return str(quick_output_path(str(target)))


def _print_json(args: argparse.Namespace) -> bool:
    if args.json is None:
        return bool(QUICK_DEFAULTS.get("json"))
    return bool(args.json)


# --------------------------------------------------------------------- summary
def _summary(
    payload: Mapping[str, Any],
    product: str,
    terms: Mapping[str, Any],
    target: str,
    schedule: Any = None,
    *,
    calendar_name: Optional[str] = None,
    rolled: int = 0,
) -> str:
    """A few verifiable lines about what was written (not a pricing report).

    The knock-out line shows the **effective** levels - the shift applied, exactly
    what the engines will see - because that is the number a reader wants to check
    against the term sheet; the payload keeps the pre-shift terms plus the rule.
    The dates line says which calendar the observation grid was rolled on, and how
    many grid dates had to move, so "顺延" is visible rather than implied.
    """
    if product == "vanilla":
        return "\n".join(
            [
                "payload    : {}".format(target),
                "contract   : vanilla_spec | {} | notional={}".format(
                    payload.get("underlying") or "-", num(payload.get("notional"), 8)
                ),
                "terms      : {} | {} | strike={} ({})".format(
                    str(payload.get("expiry_date"))[:10],
                    payload.get("option_type"),
                    num(payload.get("strike"), 8),
                    payload.get("strike_type"),
                ),
            ]
        )

    observations = list(payload.get("observations") or ())
    rates = [float(item.get("coupon_rate", 0.0)) for item in observations]
    levels = [
        float(level)
        for level in (getattr(schedule, "ko_levels", None) or ())
    ] or [float(item["ko"]) for item in observations]
    knock_in = dict(payload.get("knock_in") or {})
    anchor = float(payload.get("spot0") or 0.0)
    guaranteed = int(terms.get("guaranteed_period") or 0)
    return "\n".join(
        [
            "payload    : {}".format(target),
            "contract   : autocall_schedule | {} | notional={}".format(
                payload.get("underlying") or "-", num(payload.get("notional"), 8)
            ),
            "terms      : {} -> {} | {} observation(s){} | accrual={} | spot0={}".format(
                str(payload.get("start_date"))[:10],
                str(payload.get("expiry_date"))[:10],
                len(observations),
                "" if not guaranteed else " | guaranteed={}M".format(guaranteed),
                payload.get("day_count") or "act/365f",
                num(anchor, 8),
            ),
            "dates      : {}".format(_dates_text(calendar_name, rolled)),
            "knock-out  : {} ({}) -> {} ({}) | {}".format(
                num(levels[0], 8),
                pct(levels[0] / anchor if anchor else None),
                num(levels[-1], 8),
                pct(levels[-1] / anchor if anchor else None),
                _shift_text(payload),
            ),
            "knock-in   : {} ({}) | {} | loss strike {} ({}) | gearing {} | protection {}".format(
                num(knock_in.get("level"), 8),
                pct(float(knock_in.get("level", 0.0)) / anchor if anchor else None),
                knock_in.get("frequency"),
                num(knock_in.get("strike"), 8),
                pct(float(knock_in.get("strike", 0.0)) / anchor if anchor else None),
                num(knock_in.get("gearing"), 4),
                pct(knock_in.get("protected_principal")),
            ),
            "coupon     : {} | rebate={} (annual)".format(
                _coupon_text(rates), pct(payload.get("rebate"))
            ),
        ]
    )


def _dates_text(calendar_name: Optional[str], rolled: int) -> str:
    """What the observation grid's dates went through (see ``month_grid``)."""
    if not calendar_name:
        return "no calendar - dates as computed"
    if rolled:
        return "{} calendar | {} grid date(s) fell on a holiday and moved to the next business day".format(
            calendar_name, rolled
        )
    return "{} calendar | every observation is a business day".format(calendar_name)


def _shift_text(payload: Mapping[str, Any]) -> str:
    """The knock-out rule the levels above went through (empty when there is none)."""
    ko = dict(dict(payload.get("shift") or {}).get("ko") or {})
    mode = str(ko.get("mode") or "none")
    if mode == "none":
        return "shift=none, levels as written"
    return "shift={} {} ({})".format(mode, pct(ko.get("value")), ko.get("source"))


def _coupon_text(rates: Sequence[float]) -> str:
    """The per-observation rates compressed into runs: ``19.0000% (#1-10) -> ...``."""
    if not rates:
        return "-"
    parts: List[str] = []
    first = 1
    for index in range(1, len(rates) + 1):
        if index < len(rates) and rates[index] == rates[first - 1]:
            continue
        span = "#{}".format(first) if first == index else "#{}-{}".format(first, index)
        parts.append("{} {}".format(pct(rates[first - 1]), span))
        first = index + 1
    return " -> ".join(parts)


# ------------------------------------------------------------------------- CLI
def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="surface_pricer build-json",
        description=(
            "Write a resolved contract payload (autocall_schedule / vanilla_spec) "
            "from the QUICK_DEFAULTS block; price it with price-json."
        ),
    )
    parser.add_argument(
        "--product",
        default=None,
        choices=["autocall", "vanilla"],
        help="which block of QUICK_DEFAULTS to generate (default: the block's own)",
    )
    parser.add_argument(
        "--fit", default=None, help="'latest' / run name / directory / surface.json"
    )
    parser.add_argument(
        "--output-root", default=None, help="where the fit runs live (None -> surface_pricer/output)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where the payload goes (relative -> the package folder, not the cwd)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="file name (default: '<product>.json'; '-' prints the payload to stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=None,
        help="print the payload as well (the block's 'json' key makes it the default)",
    )
    parser.add_argument(
        "--shift-config",
        default=None,
        help="barrier-shift config (default: the packaged config/barrier_shift.json)",
    )
    parser.add_argument("--list-runs", action="store_true", help="list stored fit runs and exit")

    market = parser.add_argument_group("market")
    market.add_argument("--spot", type=float, default=None, help="override the run spot (the anchor)")
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
            "borrow_curve run: 'latest' (default) = the newest run for the index "
            "being built / a path / 'none' (flat --borrow)"
        ),
    )
    market.add_argument("--valuation-date", default=None, help="override the run valuation date")
    market.add_argument("--calendar-file", default=None, help="JSON file with calendar holidays")

    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["QUICK_DEFAULTS", "main"]


if __name__ == "__main__":
    sys.exit(main())
