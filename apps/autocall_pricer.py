"""Solve the coupon that prices an autocallable at a target NPV.

Examples::

    python -m surface_pricer autocall-pricer                       # the block below
    python -m surface_pricer autocall-pricer --target 0.95 --coupon-max 0.30
    python -m surface_pricer autocall-pricer --terms 2y-desk.json --target 1.00 --greeks delta_cash

The **terms** are the ``build_json`` autocall block's own fields - the barrier
levels as ratios of the anchor, the observation grid, the step-down, the shift
switch, ... - so a desk edits one vocabulary in both apps.  Edit the block below,
or point ``--terms FILE`` at a JSON object with the same keys (the file wins over
the block, key by key).  Two of them default the way a *quote* wants rather than
the way a trade sheet reads:

* ``notional = 1`` - the answer is a rate, and ``--target`` is a fraction of
  notional (``0.95`` = 95%);
* ``start_spot`` defaults to the **valuation spot** (and ``start`` to the
  valuation date), because a new trade is struck where the index is today; the
  report says which of the two it used.

What is solved is **one coupon rate** - the rate of the segment the term sheet
marks as the unknown:

```python
"coupon": {1: None},                 # flat: the whole schedule is the unknown
"coupon": {1: 0.10, 13: None},       # 2Y monthly: 10% pays the first 12 periods,
                                     #   solve the rate of periods 13..24
"coupon": {1: None, 13: 0.10},       # ... or the other way round: solve the head
```

The vocabulary is the **step schedule** ``build_json`` already uses (``{nominal
period: annual rate}``, a scalar for every period, a list for one rate per period).
A period is counted from the start date **by month**, guarantee or not: a
``guaranteed_period`` hides observations from the contract, not numbers from the
term sheet, so the same ladder describes the same trade either way (2026-10).
Exactly one entry is written as ``null`` - that segment runs to the next entry (or
to the last period).  One ``null`` because one bisection solves one rate; two of
them would need a multi-dimensional solve, so they are refused.  A schedule with no
``null`` has nothing to solve and is refused too, and a segment that sits entirely
inside the lock-up is refused as well (the trade observes nothing there, so the
rate would not move the price).

* each trial rate rebuilds the terms - the barrier **shift** included, so a house
  rule that is sized off the coupon stays consistent instead of being frozen at the
  first guess;
* the **rebate** (the no-KO / no-KI leg) defaults to the coupon of the **last
  period**, so it follows the tail when the tail is solved and stays on the fixed
  tail when the *head* is solved; ``rebate_gap`` is an **additive spread on that
  last coupon**, in annual-rate units (``-0.005`` = 50bp under it, ``0`` = follow
  it exactly), and an explicit ``rebate`` in the terms wins over both;
* the engine values every trial exactly like ``price-json`` does, on the same fit
  run, the same curves and the **same local-vol table** (one table for the whole
  search: the coefficients do not depend on the coupon);
* the rate is bracketed (``--coupon-min`` / ``--coupon-max``, of the **solved
  segment**) and bisected on the NPV per notional, so a target that cannot be
  reached is reported as such ("even a 100% coupon only reaches X") instead of
  returning a made-up rate;
* ``--method pde`` is the default because the search wants a **deterministic**
  value: every bisection step compares two NPVs, and Monte Carlo noise makes those
  comparisons ambiguous (with ``mc`` the common random numbers at least pair them).

The solved contract is written as a normal ``autocall_schedule`` payload
(``--out``, the block's ``autocall_priced.json``), which is the point - price it
again, take its Greeks, or slide it with the existing tools::

    python -m surface_pricer price-json output/autocall_priced.json --greeks all
    python -m surface_pricer slide     output/autocall_priced.json --slide-range 20% --csv
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # IDE "Run" launches the file with no package context; put the repository root
    # on the path and hand control to the package module (``python -m ...`` never
    # takes this branch).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.autocall_pricer import main

    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from scipy.optimize import brentq

from ..io.fit_runs import resolve_run
from ..pricing.exotics.autocall import accrual
from ..pricing.exotics.autocall.mc import AutocallMonteCarlo
from ..pricing.exotics.autocall.pde import AutocallPDE
from ..pricing.risk.diff import parse_greeks
from ..reporting.quote_report import num
from ._common import dump_payload, load_env_files, quick_output_path
from ._market import market_from_run, note_table_cache, risk_settings, table_cache_for

# The term sheet is resolved by build_json's own code: one implementation of
# "terms -> contract -> schedule" (grid rolling, coupon accrual, the shift rule)
# means the solver cannot disagree with the generator about what the contract is.
from .build_json import _autocall_payload, _guarantee_offset, _index_key

# ---------------------------------------------------------------------------
# Quick-run defaults
#
# Edit this block, then run the file (or ``python -m surface_pricer
# autocall-pricer``) with **no arguments** - e.g. with the IDE "Run" button; any
# command line argument disables the block and the CLI defaults apply instead (the
# same values, spelled on the flags).  The keys are argparse **dest** names
# (underscores) and a mistyped one is refused by :func:`_apply_quick_defaults`
# rather than silently doing nothing; ``autocall`` is the **terms block** (read by
# :func:`_terms`, checked against :data:`TERM_KEYS`), not a flag.
# ---------------------------------------------------------------------------
QUICK_DEFAULTS: Dict[str, Any] = {
    # ---- 要求解什么 ----------------------------------------------------------
    "target": 0.96,              # 目标 NPV，**占名义本金的比例**：1.0 = 平价，0.95 = 95%
    "coupon_min": 0.0,           # 搜索下界（年化）
    "coupon_max": 1.0,           # 搜索上界（100% 年化；够不到会报错提示调大）
    "coupon_tol": 1e-5,          # 票息收敛容差（年化，1e-6 = 0.0001%）
    "rebate_gap": 0.0,           # rebate 相对**最后一段票息**的**年化加差**：-0.005 = 低 50bp（10% -> 9.5%）、0 = 完全跟随（只有条款没写 rebate 时生效）
    # ---- 市场：fit run + 曲线 -------------------------------------------------
    "fit": "latest",             # 'latest' / run 名（唯一前缀即可）/ run 目录 / surface.json
    "output_root": None,         # 输出根（None -> surface_pricer/output）
    "spot": None,                # 覆盖估值现货（None -> run 的 spot；start_spot 默认跟它）
    "rate": None,                # 覆盖平坦利率（ir_curve 为 "none" 时生效）
    "borrow": None,              # 平坦融券率（borrow_curve 为 "none" 时生效）
    "ir_curve": "latest",        # 利率曲线：latest / 路径 / "none"
    "borrow_curve": "latest",    # 融券曲线：latest（按当前指数）/ 路径 / "none"
    "valuation_date": None,      # 覆盖估值日（None -> run 的估值日）
    "calendar_file": None,       # 带节假日的日历 JSON（None -> run 的日历）
    "shift_config": None,        # 位移规则 JSON（None -> 打包配置；条款里的 no_shift 优先）
    # ---- 引擎 ---------------------------------------------------------------
    "method": "pde",             # pde（默认：确定性，反解首选）/ mc
    "pde_nodes": None,           # PDE 现货网格节点数（None -> 引擎默认 601）
    "pde_theta": None,           # PDE 时间格式权重（None -> 默认）
    "paths": None,               # MC 路径数（method=mc 时）
    "seed": None,                # MC 种子
    "local_vol_cache": True,     # True = 局部波动率表走磁盘缓存（按指数分档；整场搜索共用一张表）
    # ---- 输出 ---------------------------------------------------------------
    "terms": None,               # 条款 JSON（与 autocall 块同样的键；逐键覆盖块）
    "out": "",  # 求出的 payload（相对路径按**包目录**；None / "" 不写；"-" 打 stdout）
    "json": False,               # True = 打机器可读 JSON（coupon / npv / 条款 / 希腊值）
    "greeks": "none",            # 解完顺带算的希腊值（逗号分隔，如 delta_cash,gamma_cash）；none = 只报 NPV
    "progress": True,           # True = 每次试算往 stderr 打一行
    "env_file": None,
    # ---- 合约条款（= build_json 的 autocall 块；coupon 由本 app 反解，不在这里）----
    "autocall": {
        "underlying": "000852.SH",   # 指数代码或期权产地（MO）；None -> run 的标的
        "start": None,               # 起息日（None -> 估值日；非估值日时必须给 start_spot）
        "tenor": "2Y",               # 3M / 1Y / 90D ...（expiry 为 None 时生效）
        "expiry": None,              # 显式到期日 YYYY-MM-DD，优先于 tenor
        "obs_freq": "M",             # M / Q / S / A(Y)：自到期日回推的观察网格
        "guaranteed_period": 2,      # 观察封闭期（月；0 = 第一期就观察）
        "ko": 1,                  # 敲出线（锚的比例）
        "stepdown_size": 0.005,        # 每次观察的敲出递减（0.005 = 每次 -0.5%）
        "ki": 0.65,                  # 敲入线（锚的比例）
        "ki_frequency": "expiry",     # daily / expiry / observation_dates
        "ki_strike": 1.0,            # 敲入亏损腿的行权价（锚的比例）
        "ki_gearing": 1.0,           # 亏损倍数
        "protection": 0.0,           # 保本比例
        "settlement_days": 0,        # 结算滞后（天）
        "day_count": "act/365f",     # act/365f / act/360 / act/act
        # 票息阶梯：{**名义期数**: 年化利率}（标量 = 全体同率；列表 = 每期一个）。
        # 期数从**起息日按月**数，guaranteed_period **不改变编号**（它只是让合约
        # 少观察几期），所以同一份阶梯在有没有锁定期时都描述同一个交易：
        #   整条都是未知（解一个平坦票息） : {1: None}
        #   2Y 月度：前 12 期 10%、后 12 期求解 : {1: 0.10, 13: None}
        #   反过来：求前段、后段固定 10%      : {1: None, 13: 0.10}
        "coupon": {1: None},
        # 无敲出/无敲入腿的年化利率；None -> 跟随**最后一个期数**的票息
        # （所以解尾段时 rebate 跟着走，解前段时它留在固定的尾段上）
        "rebate": None,
        "notional": 1.0,             # **默认 1**：答案就是比率（target 也按名义本金的比例）
        "start_spot": None,          # 障碍锚定的起息现货；None -> 估值现货（start = 估值日时）
        "no_shift": False,            # True = 不施加位移，按条款原样
        "ko_boundary": "inclusive",  # 触碰（含）即敲出
        "ki_boundary": "inclusive",  # 触碰（含）即敲入
    },
}

#: The term-sheet keys a trial may carry - the block's own keys.  A typo in
#: ``--terms`` (``stepdown`` for ``stepdown_size``) would otherwise be silently
#: ignored by the schedule builder, i.e. the solved coupon would describe a trade
#: nobody asked for.
TERM_KEYS = frozenset(QUICK_DEFAULTS["autocall"])


def _apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Overwrite the parsed arguments with :data:`QUICK_DEFAULTS`.

    ``autocall`` is skipped: it is the terms block (read by :func:`_terms`), not a
    flag, and the other keys are checked against the namespace so a mistyped one
    cannot quietly do nothing.
    """
    unknown = [
        key
        for key in QUICK_DEFAULTS
        if key != "autocall" and not hasattr(args, key)
    ]
    if unknown:
        raise ValueError(
            "QUICK_DEFAULTS has unknown key(s): {} - these are argparse dest names "
            "('output_root', not 'output-root')".format(", ".join(sorted(unknown)))
        )
    for key, value in QUICK_DEFAULTS.items():
        if key == "autocall":
            continue
        setattr(args, key, value)
    return args


def _no_cli_arguments() -> bool:
    """True when the process was started with no arguments (IDE Run button)."""
    return len(sys.argv) <= 1


# ------------------------------------------------------------------------- main
def main(argv: Optional[Iterable[str]] = None, *, quick: bool = False) -> int:
    args = _parse_args(argv)

    if quick or (argv is None and _no_cli_arguments()):
        try:
            args = _apply_quick_defaults(args)
        except ValueError as error:  # a typo in the block must not pass silently
            print("ERROR: {}".format(error))
            return 2

    load_env_files(args.env_file)
    try:
        terms = _terms(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("ERROR: {}".format(error))
        return 2

    try:
        run = resolve_run(
            args.fit,
            output_root=args.output_root,
            underlying=[_index_key(terms), terms.get("underlying")],
        )
        market = market_from_run(run, args)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("ERROR: {}".format(error))
        return 2

    notional = float(terms.get("notional", 1.0) or 1.0)
    if notional <= 0.0:
        print("ERROR: notional must be positive, got {}".format(notional))
        return 2

    # one table for the whole search: the coefficients are a property of the
    # market, not of the coupon - a sweep would otherwise pay for the table once
    # per trial
    table_cache = table_cache_for(args, market, run)
    engine = _engine(args, table_cache)
    try:
        solved = _solve(args, terms, market, run, engine, notional)
    except ValueError as error:
        print("ERROR: {}".format(error))
        return 2
    finally:
        note_table_cache(table_cache)

    schedule = solved["schedule"]
    target = float(args.target)
    anchor = terms.get("start_spot")
    payload = schedule.to_dict()
    start_index, end_index = solved["segment"]
    dates = schedule.observation_dates
    # the segment is numbered on the **nominal** grid; the dates it actually moves
    # are the observed periods it covers
    observed_from = max(1, start_index - solved["offset"])
    observed_to = min(solved["periods"], end_index - solved["offset"])
    first_date = dates[observed_from - 1].date().isoformat()
    last_date = dates[observed_to - 1].date().isoformat()

    lines = [
        "run        : {}".format(run.describe()),
        "terms      : {} | notional={} | start_spot={} ({}) | {} obs{} | ko={} | ki={}".format(
            schedule.underlying or "-",
            num(notional, 8),
            num(schedule.spot0, 8),
            "the valuation spot" if anchor is None else "stated",
            len(schedule.observation_dates),
            ""
            if solved["offset"] == 0
            else " (+{} in the guaranteed lock-up)".format(solved["offset"]),
            num(schedule.ko_levels[0] if schedule.ko_levels else None, 8),
            num(schedule.ki_levels[0] if schedule.ki_levels else None, 8),
        ),
        "dates      : {} -> {} | shift ko={} ki={}".format(
            schedule.start_date.date().isoformat(),
            schedule.expiry_date.date().isoformat(),
            schedule.ko_shift.describe(),
            schedule.ki_shift.describe(),
        ),
        "coupon     : {}".format(solved["description"]),
        "solved     : periods {}-{} (of {}{}) | moves {} -> {} | rebate {:.6%} annual{}".format(
            start_index,
            end_index,
            solved["nominal"],
            ""
            if solved["offset"] == 0
            else ", {} locked up".format(solved["offset"]),
            first_date,
            last_date,
            solved["rebate"],
            _rebate_note(terms, args),
        ),
        "target     : {:.4%} of notional".format(target),
        "npv        : {:,.6f} ({:.4%} of notional) | residual {:+.2e} | {} valuation(s){}"
        " | 1st observation pays {:.4%}".format(
            solved["npv"],
            solved["npv"] / notional,
            solved["npv"] / notional - target,
            solved["evaluations"],
            _engine_note(solved["result"]),
            solved["first_period"],
        ),
    ]
    greeks = _greeks(args, engine, schedule, market)
    if greeks is not None:
        lines.append(
            "greeks     : {}".format(
                " | ".join(
                    "{}={}".format(name, num(getattr(greeks, name, None), 8))
                    for name in parse_greeks(args.greeks)
                )
            )
        )

    target_path = _output_path(args)
    if target_path in {"-", "stdout"} and args.json:
        # both would land on stdout, and a reader could not tell them apart
        print("ERROR: --json and '--out -' both write to stdout; pick one")
        return 2
    if target_path in {"-", "stdout"}:
        print(json.dumps(payload, indent=2, default=str))
    elif target_path is not None:
        try:
            dump_payload(target_path, payload)
        except OSError as error:
            print("ERROR: cannot write {}: {}".format(target_path, error))
            return 2
        lines.append(
            "payload    : written {} (price / slide it like any other payload)".format(
                target_path
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "kind": "autocall_solved_coupon",
                    "run": run.name,
                    "index": run.underlying,
                    "target": target,
                    "rate": solved["rate"],
                    "segment": {"from": start_index, "to": end_index},
                    "coupon_schedule": {
                        str(key): value for key, value in solved["coupon"].items()
                    },
                    "rebate": solved["rebate"],
                    "first_period": solved["first_period"],
                    "npv": solved["npv"],
                    "npv_per_notional": solved["npv"] / notional,
                    "evaluations": solved["evaluations"],
                    "greeks": (
                        {
                            name: getattr(greeks, name, None)
                            for name in parse_greeks(args.greeks)
                        }
                        if greeks is not None
                        else {}
                    ),
                    "terms": payload,
                },
                indent=2,
                default=str,
            )
        )
    else:
        print("\n".join(lines))
    return 0


def _engine_note(result: Any) -> str:
    """The discretisation the solve ran on (worth reading once per quote)."""
    metadata = getattr(result, "metadata", None) or {}
    if metadata.get("method") == "monte_carlo":
        return " | mc paths={} seed={}".format(metadata.get("paths"), metadata.get("seed"))
    if metadata.get("method") == "pde":
        return " | pde nodes={} theta={}".format(metadata.get("nodes"), metadata.get("theta"))
    return ""


def _greeks(args: argparse.Namespace, engine, schedule, market) -> Optional[Any]:
    """The Greeks at the solved coupon, when the run asked for any."""
    selection = parse_greeks(args.greeks)
    if not selection:
        return None
    return engine.greeks_schedule(schedule, market, risk_settings(args, selection))


# ------------------------------------------------------------------------ solve
def _solve(
    args: argparse.Namespace,
    terms: Mapping[str, Any],
    market: Any,
    run: Any,
    engine: Any,
    notional: float,
) -> Dict[str, Any]:
    """Bisect the flat coupon on ``npv / notional``.

    Every trial rebuilds the contract - the shift rule may be sized off the coupon,
    so freezing the first schedule would solve a different trade - and values it on
    the same market and the same local-vol table.  The bracket's two ends are
    evaluated first, so an unreachable target is named (with the NPV it *can* reach)
    instead of being bisected into a wrong rate; each distinct rate is valued once
    (``evaluate`` remembers), so the endpoints the bisection re-probes cost nothing
    and the reported ``evaluations`` is the number of real valuations.
    """
    target = float(args.target)
    low, high = float(args.coupon_min), float(args.coupon_max)
    if low >= high:
        raise ValueError("coupon_min ({}) must be below coupon_max ({})".format(low, high))
    gap = float(args.rebate_gap)
    if gap < 0.0:
        # The rebate is taken *off* the last coupon, so a coupon below ``|gap|``
        # would need a negative rebate - which the contract refuses.  When the tail
        # is the segment being solved that floor is reachable by the low probe
        # (``coupon_min=0`` is the usual setting), so the search starts at the
        # smallest coupon that still pays a non-negative rebate instead of failing.
        floor = -gap
        if max(low, floor) >= high:
            raise ValueError(
                "coupon_min ({}) leaves no room for rebate_gap {}: the rebate would "
                "have to go negative below a coupon of {:.6%} - raise coupon_min or "
                "shrink the gap".format(args.coupon_min, gap, floor)
            )
        low = max(low, floor)
    settings = risk_settings(args, ())
    schedule_terms = _coupon_schedule(terms)
    hole = next(key for key, rate in schedule_terms.items() if rate is None)
    # The grid depends on the dates and the calendar, never on the coupon, so one
    # dummy build gives both halves of the numbering: how many periods the contract
    # observes, and how many the guarantee hides.  The ladder is counted in
    # **nominal** periods (build_json._coupon_rates), so the segment is bounded by
    # their sum - a key past the observed count is still a meaningful number.
    dummy = _autocall_payload({**terms, "coupon": 0.0}, market, run, args)[1]
    periods = len(dummy.observation_dates)
    offset = _guarantee_offset(terms, dummy.start_date, dummy.expiry_date)
    nominal = periods + offset
    end = _segment_end(schedule_terms, hole, nominal)
    if end - offset < 1:
        raise ValueError(
            "the marked segment (periods {}-{}) sits inside the guaranteed lock-up "
            "({} period(s)): the trade observes nothing there, so its coupon cannot "
            "be solved - mark a segment the trade actually observes".format(
                hole, end, offset
            )
        )
    calls = {"count": 0}
    valued: Dict[float, Tuple[float, Dict[str, Any]]] = {}

    def evaluate(rate: float) -> Tuple[float, Dict[str, Any]]:
        """Value the segment at ``rate`` - **once per rate**.

        The bracket ends are checked here and then read again by :func:`brentq`
        (it probes both to confirm the sign change), and bisection can revisit a
        point it has already seen.  The same rate is the same contract and the same
        NPV, so the answer is reused: no second valuation, no repeated ``trial``
        line, and a stochastic engine cannot hand the bisection two different
        numbers for one rate either.
        """
        rate = float(rate)
        done = valued.get(rate)
        if done is not None:
            return done
        trial = dict(terms)
        trial["coupon"] = _filled_schedule(schedule_terms, rate)
        trial["rebate"] = _rebate_rate(terms, schedule_terms, rate, gap)
        _, schedule, _ = _autocall_payload(trial, market, run, args)
        result = engine.price_schedule(schedule, market, settings)
        calls["count"] += 1
        state = {
            "npv": float(result.npv),
            "schedule": schedule,
            "result": result,
            "rate": float(rate),
            "rebate": float(trial["rebate"]),
            "segment": (hole, end),
            "first_period": float(schedule.coupon_rates[0])
            * float(accrual(schedule, schedule.observation_dates[0])),
        }
        if args.progress:
            print(
                "trial {:<3} segment rate={:.6%} npv={:.6%} of notional".format(
                    calls["count"], rate, state["npv"] / notional
                ),
                file=sys.stderr,
                flush=True,
            )
        net = state["npv"] / notional - target
        valued[rate] = (net, state)
        return net, state

    net_low, state_low = evaluate(low)
    if net_low > 0.0:
        raise ValueError(
            "the target {:.4%} is below what this structure is worth with a coupon "
            "of {:.4%} ({:.4%} of notional): the coupon would have to be negative - "
            "relax the terms or raise the target".format(
                target, low, state_low["npv"] / notional
            )
        )
    net_high, state_high = evaluate(high)
    if net_high < 0.0:
        raise ValueError(
            "even a coupon of {:.4%} only reaches {:.4%} of notional (target "
            "{:.4%}): raise coupon_max or relax the terms".format(
                high, state_high["npv"] / notional, target
            )
        )
    coupon = float(
        brentq(
            lambda value: evaluate(value)[0],
            low,
            high,
            xtol=float(args.coupon_tol),
            rtol=1e-12,
            maxiter=200,
        )
    )
    # one valuation **at the reported rate**: the bisection's last trial may sit a
    # tolerance away, and the payload written below has to be this exact contract
    _net, state = evaluate(coupon)
    return {
        "rate": state["rate"],
        "segment": state["segment"],
        "periods": periods,
        "offset": offset,
        "nominal": nominal,
        "rebate": state["rebate"],
        "description": _describe_coupons(schedule_terms, state["rate"], nominal),
        "coupon": _filled_schedule(schedule_terms, state["rate"]),
        "npv": state["npv"],
        "first_period": state["first_period"],
        "evaluations": calls["count"],
        "schedule": state["schedule"],
        "result": state["result"],
    }


# ------------------------------------------------------------------------ terms
def _terms(args: argparse.Namespace) -> Dict[str, Any]:
    """The autocall terms: the block, overridden key by key by ``--terms FILE``.

    An unknown key is refused - the schedule builder reads the keys it knows, so a
    typo would otherwise produce a coupon for a trade nobody asked for.
    """
    block = QUICK_DEFAULTS.get("autocall")
    if not isinstance(block, Mapping):
        raise ValueError("QUICK_DEFAULTS['autocall'] must be a block of terms")
    terms = dict(block)
    if args.terms:
        path = Path(args.terms).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("{} must hold a JSON object of terms".format(path))
        terms.update({str(key): value for key, value in payload.items()})
    unknown = sorted(key for key in terms if key not in TERM_KEYS)
    if unknown:
        raise ValueError(
            "unknown term(s) {}: the terms are build_json's autocall block "
            "({})".format(", ".join(unknown), ", ".join(sorted(TERM_KEYS)))
        )
    return terms


# --------------------------------------------------------------------- coupons
def _rate(value: Any) -> float:
    """A stated coupon rate, refused if it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "a fixed coupon rate must be a number, got {!r} (a rate to be solved "
            "is written null)".format(value)
        )
    return float(value)


def _coupon_schedule(terms: Mapping[str, Any]) -> Dict[int, Optional[float]]:
    """``{nominal period: rate or None}`` - the terms' ``coupon``, normalised.

    Three spellings, the ones ``build_json`` already reads: a scalar (every
    period), a list (one rate per period of the full grid) and a mapping (**step
    schedule**, keyed by nominal period - counted by month from the start date, so
    a ``guaranteed_period`` does not move the numbers).  Exactly one entry may be
    ``None``: that is the segment this app solves, and it runs to the next key - or
    to the last period.  No ``None`` -> nothing to solve; two -> two unknowns, which
    one bisection cannot serve.
    """
    value = terms.get("coupon", None)
    if value is None:
        return {1: None}  # no schedule at all: the whole thing is the unknown
    if isinstance(value, Mapping):
        schedule = {
            int(key): (None if rate is None else _rate(rate))
            for key, rate in value.items()
        }
    elif isinstance(value, (list, tuple)):
        schedule = {
            index + 1: (None if rate is None else _rate(rate))
            for index, rate in enumerate(value)
        }
    else:
        schedule = {1: _rate(value)}
    if not schedule:
        raise ValueError("the coupon schedule is empty")
    if 1 not in schedule:
        raise ValueError(
            "the coupon schedule must start at period 1 (got {})".format(min(schedule))
        )
    holes = sorted(key for key, rate in schedule.items() if rate is None)
    if not holes:
        raise ValueError(
            "the coupon schedule states every rate - there is nothing to solve: "
            "mark the segment with null, e.g. {{1: 0.10, 13: null}}"
        )
    if len(holes) > 1:
        raise ValueError(
            "the coupon schedule marks {} segments to solve ({}): one bisection "
            "solves one rate - state all but one of them".format(
                len(holes), ", ".join(str(key) for key in holes)
            )
        )
    return schedule


def _filled_schedule(
    schedule: Mapping[int, Optional[float]], rate: float
) -> Dict[int, float]:
    """The schedule with the marked segment filled in (what the builder gets)."""
    return {
        key: (float(rate) if value is None else float(value))
        for key, value in schedule.items()
    }


def _segment_end(schedule: Mapping[int, Optional[float]], key: int, periods: int):
    """The last **nominal period** the solved segment covers (inclusive)."""
    later = [candidate for candidate in schedule if candidate > key]
    end = (min(later) - 1) if later else periods
    if key > periods:
        raise ValueError(
            "the coupon schedule marks period {} as the unknown but the contract "
            "only has {} period(s)".format(key, periods)
        )
    if end < key:
        raise ValueError(
            "the coupon schedule marks observation {} as the unknown but the next "
            "entry starts at {}: the segment is empty".format(key, end + 1)
        )
    return end


def _rebate_rate(
    terms: Mapping[str, Any],
    schedule: Mapping[int, Optional[float]],
    rate: float,
    gap: float,
) -> float:
    """The rebate's annual rate: the terms' own, else the **last coupon + gap**.

    The reference is the coupon of the **last period**, whichever segment that is:
    solving the tail moves the rebate with it, while solving the head leaves the
    rebate on the (stated) tail.  ``gap`` is an **additive spread in annual-rate
    units** - points of the rate itself, not a relative change - so ``-0.005`` asks
    for a rebate 50bp under the last coupon (``0.10`` -> ``0.095``) and ``0.0``
    follows it exactly.  A ``rebate`` written in the terms wins over all of it.
    """
    explicit = terms.get("rebate")
    if explicit is not None:
        return _rate(explicit)
    last = max(schedule)
    value = rate if schedule[last] is None else schedule[last]
    return float(value) + gap


def _rebate_note(terms: Mapping[str, Any], args: argparse.Namespace) -> str:
    """The report's parenthesis on where the rebate came from."""
    if terms.get("rebate") is not None:
        return " (stated)"
    gap = float(getattr(args, "rebate_gap", 0.0) or 0.0)
    if not gap:
        return " (follows the last period)"
    return " (the last period's coupon {:+.4%})".format(gap)


def _describe_coupons(
    schedule: Mapping[int, Optional[float]], rate: float, periods: int
) -> str:
    """The whole coupon ladder, the solved segment marked - one report line."""
    keys = sorted(schedule)
    parts = []
    for index, key in enumerate(keys):
        end = keys[index + 1] - 1 if index + 1 < len(keys) else periods
        value = rate if schedule[key] is None else schedule[key]
        parts.append(
            "{:.4%} (obs {}{}{})".format(
                value,
                key,
                "" if end <= key else "-{}".format(end),
                ", solved" if schedule[key] is None else "",
            )
        )
    return " | ".join(parts)


def _output_path(args: argparse.Namespace) -> Optional[str]:
    """Where the solved payload goes (``-``/``stdout`` prints it instead)."""
    value = getattr(args, "out", None)
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    if text in {"-", "stdout"}:
        return text
    return quick_output_path(text)


# -------------------------------------------------------------------------- CLI
def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    """The flags mirror :data:`QUICK_DEFAULTS`; the *terms* stay in the block."""
    parser = argparse.ArgumentParser(
        prog="surface_pricer autocall-pricer",
        description=(
            "Solve the flat coupon that prices an autocallable at a target NPV "
            "(fraction of notional) on a stored fit run.  The contract terms come "
            "from the QUICK_DEFAULTS block or --terms FILE, exactly like build-json."
        ),
    )
    solve = parser.add_argument_group("solve")
    solve.add_argument(
        "--target",
        type=float,
        default=0.95,
        help="target NPV over notional (0.95 = 95%%)",
    )
    solve.add_argument(
        "--coupon-min",
        type=float,
        default=0.0,
        help="lower bracket, annual, of the segment being solved",
    )
    solve.add_argument(
        "--coupon-max",
        type=float,
        default=1.0,
        help="upper bracket, annual, of the segment being solved (1.0 = 100%%)",
    )
    solve.add_argument(
        "--coupon-tol",
        type=float,
        default=1e-6,
        help="convergence tolerance on that rate (annual)",
    )
    solve.add_argument(
        "--rebate-gap",
        type=float,
        default=0.0,
        help=(
            "additive spread on the **last period's** coupon, in annual-rate units "
            "(-0.005 = 50bp under it; 0 = follow it exactly); ignored when the terms "
            "state a rebate"
        ),
    )
    solve.add_argument(
        "--terms",
        default=None,
        help=(
            "JSON of contract terms (same keys as the block); the path is taken "
            "**as given**, relative to the working directory"
        ),
    )

    market = parser.add_argument_group("market")
    market.add_argument("--fit", default="latest", help="fit run: 'latest' / name / directory / surface.json")
    market.add_argument(
        "--output-root",
        default=None,
        help="output root: None -> surface_pricer/output (fit runs in vol_fit/<index>/, curves beside)",
    )
    market.add_argument("--spot", type=float, default=None, help="override the run spot (start_spot follows it)")
    market.add_argument("--rate", type=float, default=None, help="flat rate, used when --ir-curve none")
    market.add_argument("--borrow", type=float, default=None, help="flat borrow, used when --borrow-curve none")
    market.add_argument("--ir-curve", default="latest", help="ir_curve run: 'latest' / a path / 'none'")
    market.add_argument(
        "--borrow-curve",
        default="latest",
        help="borrow_curve run for the index: 'latest' / a path / 'none'",
    )
    market.add_argument("--valuation-date", default=None, help="override the valuation date")
    market.add_argument("--calendar-file", default=None, help="JSON file with calendar holidays")
    market.add_argument("--shift-config", default=None, help="barrier-shift rule JSON (None -> packaged)")

    engine = parser.add_argument_group("engine")
    engine.add_argument("--method", default="pde", help="pde (default, deterministic) / mc")
    engine.add_argument("--pde-nodes", type=int, default=None)
    engine.add_argument("--pde-theta", type=float, default=None)
    engine.add_argument("--paths", type=int, default=None, help="MC paths (method=mc)")
    engine.add_argument("--seed", type=int, default=None)
    engine.add_argument(
        "--no-local-vol-cache",
        dest="local_vol_cache",
        action="store_false",
        default=True,
        help="rebuild the local-vol table instead of using output/local_vol/<index>/",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out",
        default=None,
        help=(
            "write the solved payload here ('-' = stdout, None = do not write); a "
            "relative path is read from the **package** directory (like build-json's "
            "output_dir), because an IDE Run button's cwd is anyone's guess"
        ),
    )
    output.add_argument("--json", action="store_true", default=False, help="print the machine-readable result")
    output.add_argument("--greeks", default="none", help="Greeks at the answer, e.g. delta_cash,gamma_cash")
    output.add_argument("--progress", action="store_true", default=False, help="one stderr line per trial")
    output.add_argument("--env-file", default=None, help="additional .env file")
    return parser.parse_args(list(argv) if argv is not None else None)


def _engine(args: argparse.Namespace, table_cache: Any = None):
    """The engine for the search (the shift is already in the terms)."""
    if str(args.method).strip().lower() in ("pde", "fd", "fdm"):
        return AutocallPDE(local_vol_cache=table_cache)
    return AutocallMonteCarlo(local_vol_cache=table_cache)


__all__ = ["QUICK_DEFAULTS", "TERM_KEYS", "main"]


if __name__ == "__main__":  # pragma: no cover - plain script launch
    raise SystemExit(main())
