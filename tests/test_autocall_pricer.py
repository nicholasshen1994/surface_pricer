"""The coupon solver: a round trip, the brackets, and the terms it refuses."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from surface_pricer.apps._market import risk_settings
from surface_pricer.apps.autocall_pricer import (
    QUICK_DEFAULTS,
    TERM_KEYS,
    _apply_quick_defaults,
    _coupon_schedule,
    _filled_schedule,
    _parse_args,
    _rebate_rate,
    _solve,
    _terms,
)
from surface_pricer.apps.build_json import _autocall_payload
from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.fit_runs import FitRun
from surface_pricer.pricing.exotics.autocall import AutocallSchedule
from surface_pricer.pricing.exotics.autocall.pde import AutocallPDE

VALUATION = datetime(2026, 1, 5, 15, 0)
SPOT = 100.0

#: A short contract: six monthly observations keep a PDE solve in the test cheap,
#: and the knock-out sits **above** the spot (a snowball pays a coupon, so it must
#: not knock out on the first observation).
TERMS = {
    "underlying": "TEST",
    "start": None,
    "tenor": "6M",
    "obs_freq": "M",
    "guaranteed_period": 0,
    "ko": 1.03,
    "ki": 0.65,
    "ki_frequency": "observation_dates",  # three windows instead of daily monitoring
    "notional": 1.0,
    "start_spot": None,
    "no_shift": True,
}


def _market():
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(0.02, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(0.0, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
        surface=EDSSabrSurface(
            init_date=VALUATION,
            init_spot=SPOT,
            expiry_dates=[datetime(2026, 4, 5), datetime(2027, 1, 5)],
            atm_vols=[0.22, 0.24],
            calendar=BusinessCalendar(name="TEST"),
            trading_days_per_year=252.0,
            holiday_weight=0.0,
        ),
    )


def _run():
    return FitRun(
        name="TEST_20260105_150000",
        directory=Path("."),
        surface_path=Path("."),
        manifest={"underlying": "TEST"},
    )


def _args(**overrides):
    args = _parse_args(
        ["--method", "pde", "--pde-nodes", "201", "--coupon-min", "0.0", "--coupon-max", "0.5"]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


#: The shipped terms block, snapshotted before any test patches it.
BLOCK = dict(QUICK_DEFAULTS["autocall"])


def _value_at(rate, terms=TERMS, market=None, args=None):
    """What the app's own evaluation path says one rate for the marked segment is worth.

    Built through the same helpers the solver uses (schedule, filled segment, the
    rebate rule), so a round trip compares one contract with itself.
    """
    market = market or _market()
    args = args or _args()
    schedule_terms = _coupon_schedule(terms)
    trial = {
        **terms,
        "coupon": _filled_schedule(schedule_terms, rate),
        "rebate": _rebate_rate(terms, schedule_terms, rate, float(args.rebate_gap)),
    }
    _, schedule, _ = _autocall_payload(trial, market, _run(), args)
    result = AutocallPDE(local_vol_cache=None).price_schedule(
        schedule, market, risk_settings(args, ())
    )
    return float(result.npv), schedule


def test_the_solved_coupon_reproduces_the_target():
    """Round trip: value a known coupon, then solve for that value."""
    known = 0.15
    target, _ = _value_at(known)

    args = _args()
    args.target = target
    solved = _solve(args, TERMS, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["rate"] == pytest.approx(known, abs=1e-5)
    assert solved["coupon"] == {1: pytest.approx(known, abs=1e-5)}
    assert solved["npv"] == pytest.approx(target, abs=1e-6)
    assert solved["evaluations"] >= 3  # the two bracket ends plus the bisection


def test_no_rate_is_valued_twice():
    """The bisection re-probes the bracket ends: that must not cost a valuation."""
    target, _ = _value_at(0.15)
    args = _args()
    args.target = target
    valued = []

    class _Recording(AutocallPDE):
        def price_schedule(self, schedule, *positional, **keywords):
            valued.append(tuple(schedule.coupon_rates))
            return super().price_schedule(schedule, *positional, **keywords)

    solved = _solve(args, TERMS, _market(), _run(), _Recording(), notional=1.0)

    assert len(valued) == solved["evaluations"]  # one engine call per reported trial
    assert len(valued) == len(set(valued))  # ... and every trial is a new rate


def test_the_solved_payload_is_priceable_again():
    """``--out`` writes a normal ``autocall_schedule``: it must feed back in."""
    args = _args()
    args.target = 1.00  # par, reached with a positive coupon
    solved = _solve(args, TERMS, _market(), _run(), AutocallPDE(), notional=1.0)

    payload = solved["schedule"].to_dict()
    again = AutocallSchedule.from_dict(payload, _market())

    assert payload["kind"] == "autocall_schedule"
    assert again.coupon_rates[0] == pytest.approx(solved["rate"])
    assert set(again.coupon_rates) == {solved["rate"]}
    assert again.observation_dates == solved["schedule"].observation_dates


def test_the_start_spot_defaults_to_the_valuation_spot():
    """The app's own default: the barriers are anchored where the index is today."""
    _, schedule, _ = _autocall_payload(
        {**TERMS, "coupon": 0.1, "rebate": 0.1}, _market(), _run(), _args()
    )

    assert schedule.spot0 == pytest.approx(SPOT)
    assert schedule.ko_levels[0] == pytest.approx(TERMS["ko"] * SPOT)


def test_an_unreachable_target_is_named_not_made_up():
    market, run = _market(), _run()

    args = _args()
    args.target = 1.60  # even a 50% coupon cannot reach it
    with pytest.raises(ValueError, match="only reaches"):
        _solve(args, TERMS, market, run, AutocallPDE(), notional=1.0)

    args = _args()
    args.target = -0.50  # a 0% coupon is already worth more: the coupon would be negative
    with pytest.raises(ValueError, match="negative"):
        _solve(args, TERMS, market, run, AutocallPDE(), notional=1.0)


#: A 2Y-style split on the six-observation test contract: the head (obs 1-2) pays a
#: stated 10%, the tail (obs 3-6) is what gets solved.
SEGMENTED = {**TERMS, "coupon": {1: 0.10, 3: None}}
#: ... and the other way round: solve the head, keep the stated tail.
REVERSED = {**TERMS, "coupon": {1: None, 3: 0.12}}


def test_only_the_marked_segment_is_solved():
    known = 0.18
    target, schedule = _value_at(known, terms=SEGMENTED)

    assert schedule.coupon_rates[0] == pytest.approx(0.10)  # the stated head is untouched
    assert schedule.coupon_rates[2] == pytest.approx(known)

    args = _args()
    args.target = target
    solved = _solve(args, SEGMENTED, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["rate"] == pytest.approx(known, abs=1e-5)
    assert solved["npv"] == pytest.approx(target, abs=1e-6)
    # the marked segment runs to the last observation (periods are calendar-driven)
    assert solved["segment"] == (3, len(solved["schedule"].observation_dates))
    assert solved["coupon"][1] == pytest.approx(0.10)
    assert solved["coupon"][3] == pytest.approx(known, abs=1e-5)
    # the tail is the last observation, so the rebate follows the solved rate
    assert solved["rebate"] == pytest.approx(known, abs=1e-5)
    assert solved["schedule"].to_dict()["rebate"] == pytest.approx(known, abs=1e-5)


def test_solving_the_head_leaves_the_rebate_on_the_stated_tail():
    target, _ = _value_at(0.05, terms=REVERSED)

    args = _args()
    args.target = target
    solved = _solve(args, REVERSED, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["segment"] == (1, 2)
    assert solved["rate"] == pytest.approx(0.05, abs=1e-5)
    # the rebate belongs to the *last* observation, which is stated - not the solved head
    assert solved["rebate"] == pytest.approx(0.12)

    payload = solved["schedule"].to_dict()
    assert payload["rebate"] == pytest.approx(0.12)
    assert payload["observations"][0]["coupon_rate"] == pytest.approx(0.05, abs=1e-5)
    assert payload["observations"][2]["coupon_rate"] == pytest.approx(0.12)


def test_the_rebate_gap_is_an_additive_spread_on_the_last_coupon():
    """``-0.005`` = 50bp under the last period's coupon - whichever segment that is."""
    from surface_pricer.apps.autocall_pricer import _rebate_rate

    # the tail is the unknown being filled: the reference is what it lands on
    assert _rebate_rate({}, {1: None}, 0.13, -0.005) == pytest.approx(0.125)
    assert _rebate_rate({}, {1: 0.10, 5: None}, 0.13, -0.005) == pytest.approx(0.125)
    # the head is what gets solved: the reference is still the *stated* tail
    assert _rebate_rate({}, {1: None, 5: 0.10}, 0.13, -0.005) == pytest.approx(0.095)
    # no gap -> exactly the last coupon; an explicit rebate ignores the gap
    assert _rebate_rate({}, {1: None, 5: 0.10}, 0.13, 0.0) == pytest.approx(0.10)
    assert _rebate_rate({"rebate": 0.09}, {1: None}, 0.13, -0.005) == pytest.approx(0.09)


def test_a_negative_gap_raises_the_search_floor():
    """Solving the tail, a coupon under ``|gap|`` would need a negative rebate."""
    args = _args()  # coupon_min = 0.0: the probe that used to fail
    args.rebate_gap = -0.005
    target, _ = _value_at(0.15, terms=TERMS, args=args)
    args.target = target

    solved = _solve(args, TERMS, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["rate"] == pytest.approx(0.15, abs=1e-5)
    assert solved["rebate"] == pytest.approx(0.145)  # the solved tail minus the gap

    # no room for the floor at all -> named, not a mysterious "negative rebate"
    args = _args()
    args.rebate_gap = -0.005
    args.coupon_max = 0.004  # below |gap|: even the floor is out of reach
    args.target = 1.0
    with pytest.raises(ValueError, match="no room for rebate_gap"):
        _solve(args, TERMS, _market(), _run(), AutocallPDE(), notional=1.0)


def test_a_gap_lowers_the_rebate_against_the_last_coupon():
    """End to end: solving the head, the rebate follows the stated tail minus the gap."""
    args = _args()
    args.rebate_gap = -0.005
    target, _ = _value_at(0.05, terms=REVERSED, args=args)
    args.target = target

    solved = _solve(args, REVERSED, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["rate"] == pytest.approx(0.05, abs=1e-5)
    assert solved["rebate"] == pytest.approx(0.115)  # not 0.05 - 0.005
    assert solved["schedule"].to_dict()["rebate"] == pytest.approx(0.115)


#: 1Y monthly with a 3-month lock-up: the ladder still counts months 1..12, so the
#: segment is written in months, not in "which observation survived the guarantee".
GUARANTEED = {
    **TERMS,
    "tenor": "1Y",
    "guaranteed_period": 3,
    "coupon": {1: 0.10, 5: None},
}


def test_the_ladder_counts_nominal_periods_under_a_guarantee():
    """``{1: 0.10, 5: None}`` means "month 5 on" even with 3 months locked up."""
    target, schedule = _value_at(0.18, terms=GUARANTEED)

    assert len(schedule.observation_dates) == 9  # 12 nominal periods, 3 hidden
    # nominal 5 is the *second* observation, so the first still pays the stated head
    assert schedule.coupon_rates[0] == pytest.approx(0.10)
    assert schedule.coupon_rates[1] == pytest.approx(0.18)

    args = _args()
    args.target = target
    solved = _solve(args, GUARANTEED, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["segment"] == (5, 12)
    assert (solved["offset"], solved["nominal"], solved["periods"]) == (3, 12, 9)
    assert solved["rate"] == pytest.approx(0.18, abs=1e-5)
    # the tail is the last nominal period, so the rebate follows the solved rate
    assert solved["rebate"] == pytest.approx(solved["rate"], abs=1e-5)

    rates = solved["schedule"].coupon_rates
    assert rates[0] == pytest.approx(0.10)
    assert all(rate == pytest.approx(solved["rate"], abs=1e-5) for rate in rates[1:])


def test_a_segment_inside_the_lock_up_is_refused():
    """A rate nobody observes cannot be solved: it would not move the price."""
    terms = {
        **TERMS,
        "tenor": "1Y",
        "guaranteed_period": 3,
        "coupon": {1: None, 4: 0.10},  # nominal 1-3 are inside the lock-up
    }
    args = _args()
    args.target = 1.0

    with pytest.raises(ValueError, match="inside the guaranteed lock-up"):
        _solve(args, terms, _market(), _run(), AutocallPDE(), notional=1.0)


def test_an_explicit_rebate_wins_over_following_the_last_observation():
    terms = {**TERMS, "rebate": 0.09, "coupon": {1: None}}
    target, _ = _value_at(0.15, terms=terms)

    args = _args()
    args.target = target
    solved = _solve(args, terms, _market(), _run(), AutocallPDE(), notional=1.0)

    assert solved["rebate"] == pytest.approx(0.09)
    assert solved["schedule"].to_dict()["rebate"] == pytest.approx(0.09)


def test_the_bracket_must_be_a_bracket():
    args = _args(coupon_min=0.3, coupon_max=0.2)
    with pytest.raises(ValueError, match="coupon_min"):
        _solve(args, TERMS, _market(), _run(), AutocallPDE(), notional=1.0)


def test_the_coupon_schedule_marks_exactly_one_unknown():
    """The ladder is the terms' own shape, with one ``null`` for the solved segment."""
    # no schedule at all -> the whole thing is the unknown
    assert _coupon_schedule({}) == {1: None}
    # build_json's spellings still read - as long as one entry is the unknown
    assert _coupon_schedule({"coupon": [0.10, 0.10, None]}) == {1: 0.10, 2: 0.10, 3: None}
    assert _coupon_schedule({"coupon": {1: 0.10, 13: None}}) == {1: 0.10, 13: None}
    assert _coupon_schedule({"coupon": {1: None, 13: 0.10}}) == {1: None, 13: 0.10}

    with pytest.raises(ValueError, match="nothing to solve"):
        _coupon_schedule({"coupon": {1: 0.10, 13: 0.12}})
    with pytest.raises(ValueError, match="nothing to solve"):
        _coupon_schedule({"coupon": 0.10})  # a stated flat rate is not a solve
    with pytest.raises(ValueError, match="marks 2 segments"):
        _coupon_schedule({"coupon": {1: None, 3: None}})
    with pytest.raises(ValueError, match="must be a number"):
        _coupon_schedule({"coupon": {1: "ten"}})
    with pytest.raises(ValueError, match="start at period 1"):
        _coupon_schedule({"coupon": {3: 0.10, 5: None}})


def test_an_unknown_term_is_refused(tmp_path):
    """A typo in the block or the file would otherwise be silently ignored."""
    payload = tmp_path / "terms.json"
    payload.write_text(json.dumps({"stepdown": 0.01}), encoding="utf-8")  # meant stepdown_size

    with pytest.raises(ValueError, match="unknown term\\(s\\) stepdown"):
        _terms(_parse_args(["--terms", str(payload)]))


def test_the_file_overrides_the_block_key_by_key(tmp_path):
    payload = tmp_path / "terms.json"
    payload.write_text(json.dumps({"tenor": "1Y", "ko": 1.0}), encoding="utf-8")

    terms = _terms(_parse_args(["--terms", str(payload)]))

    assert terms["tenor"] == "1Y" and terms["ko"] == 1.0
    assert terms["ki"] == QUICK_DEFAULTS["autocall"]["ki"]  # untouched keys survive
    assert set(terms) <= TERM_KEYS


def test_the_quick_block_matches_the_flags():
    args = _parse_args([])
    unknown = [key for key in QUICK_DEFAULTS if key != "autocall" and not hasattr(args, key)]
    assert not unknown, "the block names keys the parser does not define: {}".format(unknown)

    applied = _apply_quick_defaults(_parse_args([]))
    assert applied.target == QUICK_DEFAULTS["target"]
    assert applied.coupon_max == QUICK_DEFAULTS["coupon_max"]
    assert applied.out == QUICK_DEFAULTS["out"]

    QUICK_DEFAULTS["coupon-max"] = 1.0  # a dash instead of an underscore
    try:
        with pytest.raises(ValueError, match="coupon-max"):
            _apply_quick_defaults(_parse_args([]))
    finally:
        del QUICK_DEFAULTS["coupon-max"]


def test_the_launcher_hands_the_block_to_the_solver(monkeypatch):
    """No arguments -> the block is on; any argument -> the CLI defaults apply."""
    from surface_pricer.__main__ import main as launcher_main
    from surface_pricer.apps import autocall_pricer

    calls = []

    def _fake(rest, **kwargs):
        calls.append((list(rest), dict(kwargs)))
        return 0

    monkeypatch.setattr(autocall_pricer, "main", _fake)

    assert launcher_main(["autocall-pricer"]) == 0
    assert calls == [([], {"quick": True})]

    calls.clear()
    assert launcher_main(["autocall-pricer", "--target", "0.98"]) == 0
    assert calls == [(["--target", "0.98"], {"quick": False})]
