"""Autocall contract + effective schedule: absolute barrier expansion, shift
priority, history replay and the shared cash-flow rules."""

import dataclasses
import json
from datetime import date, datetime, timedelta

import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.pricing.exotics import available_product_types, get_pricer
from surface_pricer.pricing.exotics.autocall import (
    AutocallContract,
    AutocallSchedule,
    accrual,
    apply_history,
    build_schedule,
    build_time_grid,
    expiry_cash_flow,
    ko_cash_flow,
    pricer_for,
    rebate_cash_flow,
    replay_history,
    resolve_trigger_basis,
)

VALUATION = datetime(2026, 1, 5, 15, 0)
OBSERVATIONS = (date(2026, 4, 5), date(2026, 7, 5), date(2026, 10, 5))
SPOT = 100.0
LATER = datetime(2026, 5, 1, 15, 0)


def _market(spot=SPOT, valuation=VALUATION):
    return MarketState(
        valuation_date=valuation,
        spot=spot,
        rate_curve=ConstantRateCurve(0.02, anchor=valuation),
        borrow_curve=ConstantRateCurve(0.0, anchor=valuation),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


def _contract(**overrides):
    params = dict(
        underlying="MO",
        start_date=date(2026, 1, 5),
        expiry_date=date(2026, 10, 5),
        observation_dates=OBSERVATIONS,
        ko_levels=(1.0,),
        ki_level=0.75,
        annual_coupon=0.20,
        notional=1.0e6,
        start_spot=SPOT,
    )
    params.update(overrides)
    return AutocallContract(**params)


def _business_days(start, end):
    """Business days in ``(start, end]`` - spelled out, not taken from the code
    under test - plus the knock-out observations (always monitored)."""
    calendar = _market().calendar
    days = set()
    day = start.date() + timedelta(days=1)
    while day <= end.date():
        if calendar.is_business_day(day):
            days.add(day)
        day += timedelta(days=1)
    days.update(day for day in OBSERVATIONS if start.date() < day <= end.date())
    return sorted(days)


# --------------------------------------------------------------- the schedule
def test_default_rules_produce_shifted_absolute_levels():
    schedule = build_schedule(_contract(), _market())

    assert schedule.spot0 == pytest.approx(SPOT)
    assert schedule.anchored_on == "start_spot"
    assert schedule.ko_levels_raw == pytest.approx((100.0, 100.0, 100.0))

    # coupon_fraction rule: -0.1 * 0.20 = -2% relative, accrued to the last date
    assert schedule.ko_shift.value == pytest.approx(-0.02)
    assert schedule.ko_shift.stepwise is True
    assert schedule.ko_levels == pytest.approx(
        (100 * (1 - 0.02 / 3), 100 * (1 - 0.04 / 3), 100 * (1 - 0.02))
    )

    # ki: index-like -> relative -1.25%, flat
    assert schedule.ki_shift.value == pytest.approx(-0.0125)
    assert schedule.ki_level_raw == pytest.approx(75.0)
    assert schedule.ki_levels == pytest.approx((75.0 * 0.9875,) * 3)


def test_relative_and_additive_overrides_agree_on_absolute_levels():
    """Shift values live in barrier-ratio space: additive -0.02 on a 100% KO
    equals relative -2%, and on a 75% KI equals relative -2.67%."""
    relative = build_schedule(_contract(), _market(), ko_shift=-0.02, ki_shift=-0.0125)
    additive = build_schedule(
        _contract(),
        _market(),
        ko_shift={"mode": "additive", "value": -0.02},
        ki_shift={"mode": "additive", "value": -0.75 * 0.0125},
    )

    assert relative.ko_levels == pytest.approx(additive.ko_levels)
    assert relative.ki_levels == pytest.approx(additive.ki_levels)


def test_contractual_disables_the_shift_entirely():
    schedule = build_schedule(_contract(), _market(), contractual=True)

    assert schedule.ko_shift.active is False
    assert schedule.ki_shift.active is False
    assert schedule.ko_levels == pytest.approx(schedule.ko_levels_raw)
    assert schedule.ki_levels == pytest.approx((75.0, 75.0, 75.0))


def test_override_priority_contract_then_cli():
    contract = _contract(shift_override={"ko": -0.03})

    from_contract = build_schedule(contract, _market())
    assert from_contract.ko_shift.source == "contract-override"
    assert from_contract.ko_shift.value == pytest.approx(-0.03)
    # ki side untouched by the partial override
    assert from_contract.ki_shift.value == pytest.approx(-0.0125)

    from_cli = build_schedule(contract, _market(), ko_shift=-0.05)
    assert from_cli.ko_shift.source == "cli"
    assert from_cli.ko_shift.value == pytest.approx(-0.05)


def test_schedule_anchors_on_the_start_spot_by_default():
    market = _market(spot=90.0)

    pinned = build_schedule(_contract(start_spot=100.0), market)
    assert pinned.spot0 == pytest.approx(100.0)
    assert pinned.ko_levels_raw == pytest.approx((100.0, 100.0, 100.0))

    floating = build_schedule(_contract(anchor="valuation_spot"), market)
    assert floating.spot0 == pytest.approx(90.0)
    assert floating.anchored_on == "valuation_spot"

    with pytest.raises(ValueError, match="start spot"):
        build_schedule(_contract(start_spot=None), _market(spot=90.0, valuation=LATER))


def test_settlement_days_move_the_payment_dates():
    schedule = build_schedule(_contract(settlement_days=3), _market())

    for day, payment in zip(schedule.observation_dates, schedule.payment_dates):
        assert payment == day + timedelta(days=3)
    assert schedule.expiry_payment_date == datetime(2026, 10, 8)
    assert schedule.discount_factors[0] == pytest.approx(
        _market().discount_factor(datetime(2026, 4, 8))
    )


# ------------------------------------------------------------ history replay
def _elapsed(**overrides):
    """A contract whose first observation is already in the past.

    The sparse fixings used here fit the simplified knock-in convention; the
    daily one is covered by its own test below.
    """
    overrides.setdefault("ki_frequency", "observation_dates")
    return _contract(**overrides)


def test_past_observations_require_history_fixings():
    # the replay is the *caller's* job and lives outside the engines
    with pytest.raises(ValueError, match="history"):
        apply_history(_elapsed(), _market(valuation=LATER))

    market = _market(valuation=LATER)
    contract = apply_history(_elapsed(history=((date(2026, 4, 5), 95.0),)), market)
    schedule = build_schedule(contract, market)

    assert schedule.n_observations == 2
    assert schedule.observation_dates[0] == datetime(2026, 7, 5)
    assert schedule.knocked_out is False
    assert schedule.knocked_in_before is False


def test_replay_turns_the_fixings_into_flags():
    market = _market(valuation=LATER)

    assert replay_history(
        _elapsed(history=((date(2026, 4, 5), 101.0),)), market
    ) == (None, datetime(2026, 4, 5))
    assert replay_history(
        _elapsed(history=((date(2026, 4, 5), 70.0),)), market
    ) == (datetime(2026, 4, 5), None)
    assert replay_history(
        _elapsed(history=((date(2026, 4, 5), 95.0),)), market
    ) == (None, None)


def test_the_boundary_convention_travels_and_decides_a_fixing_on_the_level():
    """Does touching the barrier trigger?  Contractual and per side: the payload
    carries it, the replay reads it, and a fixing exactly on the level is the one
    case where it changes the answer."""
    schedule = build_schedule(_contract(), _market())
    assert schedule.ko_boundary == "inclusive" and schedule.ki_boundary == "inclusive"
    payload = schedule.to_dict()
    assert payload["knock_out"]["boundary"] == "inclusive"
    assert payload["knock_in"]["boundary"] == "inclusive"

    payload["knock_out"]["boundary"] = "exclusive"
    payload["knock_in"]["boundary"] = "exclusive"
    rebuilt = AutocallSchedule.from_dict(payload, _market())
    assert rebuilt.ko_boundary == "exclusive"
    assert rebuilt.ki_boundary == "exclusive"

    # one spelling per convention: the operator forms are refused, not translated
    payload["knock_in"]["boundary"] = "strict"
    with pytest.raises(ValueError, match="no aliases"):
        AutocallSchedule.from_dict(payload, _market())

    market = _market(valuation=LATER)
    at_ko = _elapsed(history=((date(2026, 4, 5), SPOT),))  # KO ratio 1.0 -> 100
    assert replay_history(at_ko, market) == (None, datetime(2026, 4, 5))
    assert replay_history(
        dataclasses.replace(at_ko, ko_boundary="exclusive"), market
    ) == (None, None)

    at_ki = _elapsed(history=((date(2026, 4, 5), 75.0),), ki_level=0.75)
    assert replay_history(at_ki, market) == (datetime(2026, 4, 5), None)
    assert replay_history(
        dataclasses.replace(at_ki, ki_boundary="exclusive"), market
    ) == (None, None)


def test_daily_knock_in_needs_a_fixing_per_business_day():
    """The default convention monitors the knock-in every business day, so the
    sparse fixing that suits the simplified convention is not enough."""
    market = _market(valuation=LATER)

    with pytest.raises(ValueError, match="2026-01-06"):
        replay_history(_contract(history=((date(2026, 4, 5), 95.0),)), market)

    # one fixing per business day replays cleanly
    daily = tuple(
        (day, 99.0)
        for day in _business_days(datetime(2026, 1, 5), LATER)
    )
    assert replay_history(_contract(history=daily), market) == (None, None)


def test_knock_out_after_a_knock_in_still_settles():
    """A knock-in switches the expiry payoff; a later knock-out still pays the
    coupon and ends the trade."""
    market = _market(valuation=datetime(2026, 8, 1, 15, 0))
    contract = _elapsed(
        history=((date(2026, 4, 5), 70.0), (date(2026, 7, 5), 101.0)), ki_level=0.75
    )

    knocked_in_date, knocked_out_at = replay_history(contract, market)

    assert knocked_in_date == datetime(2026, 4, 5)
    assert knocked_out_at == datetime(2026, 7, 5)


def test_past_knock_out_reduces_to_a_single_cash_flow():
    market = _market(valuation=LATER)
    contract = apply_history(_elapsed(history=((date(2026, 4, 5), 101.0),)), market)
    schedule = build_schedule(contract, market)

    assert schedule.knocked_out is True
    assert schedule.knocked_out_index == 0
    assert schedule.is_settled is True
    # principal + 90 days of coupon at 20% (act/365)
    assert schedule.knocked_out_cash == pytest.approx(1.0e6 * (1 + 0.20 * 90 / 365.0))
    assert schedule.knocked_out_payment_date == datetime(2026, 4, 5)
    assert schedule.knocked_out_discount_factor == pytest.approx(
        market.discount_factor(datetime(2026, 4, 5))
    )


def test_past_knock_in_flag_carries_into_the_schedule():
    market = _market(valuation=LATER)
    contract = apply_history(_elapsed(history=((date(2026, 4, 5), 70.0),)), market)
    schedule = build_schedule(contract, market)

    assert schedule.knocked_in_before is True
    assert schedule.knocked_in_date == datetime(2026, 4, 5)
    assert schedule.knocked_out is False
    assert any("knocked in on 2026-04-05" in note for note in schedule.notes)


def test_inconsistent_history_flags_are_rejected():
    market = _market(valuation=LATER)

    with pytest.raises(ValueError, match="past observation"):
        build_schedule(_contract(knocked_out_at=date(2026, 7, 5)), market)
    with pytest.raises(ValueError, match="observation dates"):
        build_schedule(_contract(knocked_out_at=date(2026, 4, 20)), market)
    with pytest.raises(ValueError, match="start_date"):
        build_schedule(_contract(knocked_in_date=date(2026, 1, 5)), _market())
    with pytest.raises(ValueError, match="expiry"):
        build_schedule(_contract(knocked_in_date=date(2026, 10, 6)), _market())


def test_history_uses_raw_levels_not_shifted_ones():
    """Past observations replay against the raw barriers (edslib convention):
    a fixing between the shifted and the raw knock-out level must not knock
    out."""
    market = _market(valuation=LATER)
    contract = apply_history(_elapsed(history=((date(2026, 4, 5), 99.0),)), market)

    # -5% relative, accrued: the first shifted level is 98.33; the fixing (99)
    # exceeds it but still sits below the raw 100 -> no knock-out
    shifted = build_schedule(contract, market, ko_shift=-0.05)
    assert shifted.knocked_out is False

    # for contrast: a fixing above the raw level does knock out
    above = build_schedule(
        apply_history(_elapsed(history=((date(2026, 4, 5), 100.5),)), market), market
    )
    assert above.knocked_out is True


# --------------------------------------------------------------- cash flows
def _resolved(**overrides):
    """(schedule, market) for the cash-flow rules (they read the schedule only)."""
    market = _market()
    return build_schedule(_contract(**overrides), market), market


def test_cash_flow_rules():
    schedule, _ = _resolved()

    assert accrual(schedule, date(2026, 4, 5)) == pytest.approx(90 / 365.0)
    assert ko_cash_flow(schedule, 0) == pytest.approx(1.0e6 * (1 + 0.20 * 90 / 365.0))

    # no knock-in, no knock-out: principal + the rebate (the last coupon here)
    assert expiry_cash_flow(schedule, 120.0, False) == pytest.approx(
        1.0e6 * (1 + 0.20 * 273 / 365.0)
    )
    # knock-in: short put, capped upside
    assert expiry_cash_flow(schedule, 60.0, True) == pytest.approx(0.60e6)
    assert expiry_cash_flow(schedule, 130.0, True) == pytest.approx(1.0e6)


def test_the_coupon_accrues_the_whole_period_when_valued_mid_life():
    """The accrual origin is the inception, never the valuation date.

    A trade valued on 2026-05-01 still pays its first surviving observation the
    full 2026-01-05 -> 2026-07-05 accrual; accruing from the valuation date would
    under-count every coupon (and leak into NPV, theta).
    """
    market = _market(valuation=LATER)
    schedule = build_schedule(_contract(), market)

    assert schedule.observation_dates[0].date() > LATER.date()
    for index, day in enumerate(schedule.observation_dates):
        expected = (day.date() - date(2026, 1, 5)).days / 365.0
        assert accrual(schedule, day) == pytest.approx(expected, rel=1e-12)
        assert ko_cash_flow(schedule, index) == pytest.approx(
            1.0e6 * (1.0 + 0.20 * expected), rel=1e-12
        )


def test_a_valuation_date_bump_moves_no_cash_flow():
    """Theta must be time decay, not an accrual artefact."""
    market = _market(valuation=LATER)
    base = build_schedule(_contract(), market)
    bumped = base.rebased(market.clone(valuation_date=LATER + timedelta(days=1)))

    assert bumped.start_date == base.start_date
    assert bumped.day_count == base.day_count
    assert [
        ko_cash_flow(bumped, index) for index in range(len(bumped.observation_dates))
    ] == [ko_cash_flow(base, index) for index in range(len(base.observation_dates))]
    assert accrual(bumped, bumped.observation_dates[0]) == pytest.approx(
        accrual(base, base.observation_dates[0]), rel=1e-12
    )


def test_the_coupon_day_count_is_configurable():
    """``act/act`` pays the leap day, ``act/365f`` does not (the default)."""
    contract = _contract(
        start_date=date(2027, 11, 30),
        expiry_date=date(2028, 11, 30),
        observation_dates=(date(2028, 11, 30),),
        day_count="act/act",
    )
    market = _market(valuation=datetime(2027, 11, 30, 15, 0))
    schedule = build_schedule(contract, market)

    assert schedule.day_count == "act/act"
    leap = accrual(schedule, schedule.observation_dates[0])
    assert leap == pytest.approx(32 / 365.0 + 334 / 366.0, rel=1e-12)
    # the rebate leg is on the same basis
    assert schedule.rebate_ratio == pytest.approx(1.0 + 0.20 * leap, rel=1e-12)

    fixed = dataclasses.replace(schedule, day_count="act/365f")
    assert accrual(fixed, fixed.observation_dates[0]) == pytest.approx(
        366 / 365.0, rel=1e-12
    )
    assert leap < accrual(fixed, fixed.observation_dates[0])


def test_the_payload_does_not_enumerate_a_rule_based_grid():
    """``daily`` is a *rule*: ~250 dates and levels would dominate the payload."""
    schedule = build_schedule(_contract(ki_frequency="daily"), _market())
    knock_in = schedule.to_dict()["knock_in"]

    assert len(schedule.ki_dates) > 20  # the grid itself is untouched
    assert knock_in["frequency"] == "daily"
    assert "dates" not in knock_in and "levels" not in knock_in
    # one *term-sheet* barrier; the shift turns it into each period's level
    assert knock_in["level"] == pytest.approx(schedule.ki_level_raw)
    assert "observation_levels" not in knock_in

    rebuilt = AutocallSchedule.from_dict(schedule.to_dict(), _market())
    assert rebuilt.ki_frequency == "daily"
    assert rebuilt.ki_dates == schedule.ki_dates
    assert rebuilt.ki_monitor_levels == pytest.approx(schedule.ki_monitor_levels)


def test_a_custom_grid_is_written_out_and_reloads():
    """``custom`` is the one rule that carries its own dates and levels."""
    schedule = build_schedule(_contract(ki_frequency="observation_dates"), _market())
    payload = schedule.to_dict()
    payload["knock_in"] = {
        **payload["knock_in"],
        "frequency": "custom",
        "dates": ["2026-05-06", "2026-08-06"],
        "levels": [70.0, 72.0],
    }

    rebuilt = AutocallSchedule.from_dict(payload, _market())

    assert rebuilt.ki_frequency == "custom"
    assert [day.date().isoformat() for day in rebuilt.ki_dates] == [
        "2026-05-06",
        "2026-08-06",
    ]
    assert rebuilt.ki_monitor_levels == pytest.approx((70.0, 72.0))
    flat = float(payload["knock_in"]["level"])  # the effective (shifted) barrier
    assert rebuilt.ki_levels == pytest.approx((flat, flat, flat))

    # without a flat level the periods fall back to the grid's last level
    without_level = {
        **payload,
        "knock_in": {
            key: value
            for key, value in payload["knock_in"].items()
            if key != "level"
        },
    }
    assert AutocallSchedule.from_dict(
        without_level, _market()
    ).ki_levels == pytest.approx((72.0, 72.0, 72.0))

    # a valuation-date bump keeps both the surviving dates and their own levels
    bumped = rebuilt.rebased(_market(valuation=datetime(2026, 6, 1, 15, 0)))
    assert [day.date().isoformat() for day in bumped.ki_dates] == ["2026-08-06"]
    assert bumped.ki_monitor_levels == pytest.approx((72.0,))
    assert bumped.to_dict()["knock_in"]["dates"] == ["2026-08-06"]

    with pytest.raises(ValueError, match="needs its own dates"):
        AutocallSchedule.from_dict(
            {**payload, "knock_in": {"frequency": "custom", "level": 75.0}}, _market()
        )


def test_the_knock_in_strike_is_absolute_in_the_payload():
    schedule = build_schedule(_contract(ki_strike=0.9), _market())
    payload = schedule.to_dict()

    assert payload["knock_in"]["strike"] == pytest.approx(0.9 * SPOT)  # not the ratio
    assert AutocallSchedule.from_dict(payload, _market()).ki_strike == pytest.approx(0.9)

    # a payload that still spells the strike as a ratio is refused, loudly
    payload["knock_in"]["strike"] = 0.9
    with pytest.raises(ValueError, match="absolute"):
        AutocallSchedule.from_dict(payload, _market())


def test_the_payload_carries_the_term_sheet_barrier_plus_the_shift_rule():
    """The levels are pre-shift and the rule travels with them: editing the rule
    moves the barriers, which is why the payload spells it out."""
    schedule = build_schedule(_contract(), _market())
    payload = schedule.to_dict()
    knock_in = payload["knock_in"]

    assert knock_in["level"] == pytest.approx(schedule.ki_level_raw)
    assert "ki" in payload["shift"] and "ko" in payload["shift"]
    assert payload["shift"]["ko"]["mode"] in ("none", "relative", "additive")

    rebuilt = AutocallSchedule.from_dict(payload, _market())
    assert rebuilt.ko_levels == pytest.approx(schedule.ko_levels)
    assert rebuilt.ki_levels == pytest.approx(schedule.ki_levels)
    assert rebuilt.ki_monitor_levels == pytest.approx(schedule.ki_monitor_levels)

    # a hand edit of the rule (0.5% -> 3%) moves every barrier, no level touched
    payload["shift"]["ko"] = {"mode": "relative", "value": -0.03, "source": "hand"}
    payload["shift"]["ki"] = {"mode": "additive", "value": -0.03, "source": "hand"}
    edited = AutocallSchedule.from_dict(payload, _market())

    assert edited.ko_levels[0] == pytest.approx(schedule.ko_levels_raw[0] * 0.97)
    # an additive value is a *ratio* point: -0.03 is -3% of the anchor, never
    # three cents off the price (``build_schedule`` expands the same rule the same
    # way, and the two entry points must agree)
    assert edited.ki_levels[0] == pytest.approx(
        schedule.ki_level_raw - 0.03 * schedule.spot0
    )
    assert edited.ko_shift.source == "hand"


def test_a_payload_and_its_term_sheet_agree_on_an_additive_shift():
    """Same rule, two entry points, one effective barrier - at a real anchor.

    ``build_schedule`` expands the rule in barrier-ratio space and ``from_dict``
    has to read the prices back into it: a 100% anchor hides the difference
    (ratio = price), an index level does not.  A 75% knock-in shifted by an
    additive -0.03 has to land at 72% of the anchor on both paths.
    """
    market = _market(spot=8_301.8)
    additive = {"mode": "additive", "value": -0.03}

    schedule = build_schedule(
        _contract(start_spot=8_301.8), market, ko_shift=additive, ki_shift=additive
    )
    rebuilt = AutocallSchedule.from_dict(schedule.to_dict(), market)

    assert schedule.ki_levels[0] == pytest.approx(0.72 * 8_301.8)
    assert rebuilt.ki_levels == pytest.approx(schedule.ki_levels)
    assert rebuilt.ko_levels == pytest.approx(schedule.ko_levels)
    assert rebuilt.ki_monitor_levels == pytest.approx(schedule.ki_monitor_levels)


def test_a_stepwise_shift_resumes_where_the_payload_starts():
    """A mid-life payload must reproduce the full timeline's levels, not restart."""
    schedule = build_schedule(_contract(), _market())
    assert schedule.ko_shift.stepwise  # the packaged rule steps over the periods

    later = schedule.rebased(_market(valuation=datetime(2026, 5, 1, 15, 0)))
    payload = later.to_dict()
    rebuilt = AutocallSchedule.from_dict(payload, _market(valuation=datetime(2026, 5, 1, 15, 0)))

    assert payload["shift"]["elapsed"] > 0
    assert [day.date().isoformat() for day in rebuilt.observation_dates] == [
        day.date().isoformat() for day in later.observation_dates
    ]
    # the surviving observations carry the accrual they had on the full timeline
    assert rebuilt.ko_levels == pytest.approx(later.ko_levels)
    assert rebuilt.ki_levels == pytest.approx(later.ki_levels)


def test_the_text_report_shows_the_raw_level_only_when_the_shift_moved_it():
    """The human report prints both views while a shift is in play - and the payload
    fed schedule prints exactly the same thing, since the shift is re-applied."""
    from surface_pricer.pricing.results import PricingResult
    from surface_pricer.reporting.autocall_report import format_autocall

    def report(schedule):
        result = PricingResult(
            npv=1.0e6,
            forward=100.0,
            discount_factor=0.99,
            implied_vol=0.0,
            strike=0.0,
            year_fraction=0.5,
        )
        return format_autocall(schedule, result, _market(), with_greeks=False)

    shifted = build_schedule(_contract(), _market())
    from_payload = AutocallSchedule.from_dict(shifted.to_dict(), _market())
    flat = build_schedule(_contract(), _market(), contractual=True)  # no shift

    assert "(raw" in report(shifted)
    assert report(from_payload) == report(shifted)
    assert "(raw" not in report(flat)
    # the rebate is quoted as an annual rate (the last coupon here), with the total
    # it pays at expiry next to it
    assert "rebate={:.4%} (annual".format(shifted.rebate_rate) in report(shifted)
    assert "{:.4%} of notional at expiry".format(shifted.rebate_ratio) in report(shifted)


def test_a_schedule_without_a_start_date_refuses_to_accrue():
    """Falling back to the valuation date is exactly the under-accrual bug."""
    schedule = dataclasses.replace(
        build_schedule(_contract(), _market()), start_date=None
    )

    with pytest.raises(ValueError, match="no start_date"):
        accrual(schedule, schedule.observation_dates[0])


def test_an_unknown_day_count_is_rejected():
    with pytest.raises(ValueError, match="unsupported day count"):
        _contract(day_count="actual/365l")

    payload = build_schedule(_contract(), _market()).to_dict()
    payload["day_count"] = "30/360"
    with pytest.raises(ValueError, match="unsupported day count"):
        AutocallSchedule.from_dict(payload, _market())


def test_the_payload_carries_the_accrual_origin_and_basis():
    payload = build_schedule(_contract(day_count="act/360"), _market()).to_dict()

    assert payload["day_count"] == "act/360"
    assert payload["start_date"] == "2026-01-05T00:00:00"
    assert AutocallSchedule.from_dict(payload, _market()).day_count == "act/360"

    payload.pop("start_date")
    with pytest.raises(ValueError, match="no start_date"):
        AutocallSchedule.from_dict(payload, _market())


def test_trigger_basis_reads_todays_knock_in_off_the_shifted_line():
    """Intraday reads today through the post-shift barrier, EOD the raw term sheet.

    A spot between the two lines stays *not knocked in* only on the ``effective``
    basis - which is what keeps a desk's greeks on the state the pricing model is
    in until the close (the EOD run, on ``contractual``) makes it legal.
    """
    # 2026-02-02 is a Monday: the daily knock-in grid monitors today
    market = _market(spot=74.5, valuation=datetime(2026, 2, 2, 15, 0))
    shift = {"mode": "relative", "value": -0.01}  # KI 75 -> 74.25, KO 100 -> 99

    eod = build_schedule(_contract(), market, ko_shift=shift, ki_shift=shift)
    assert eod.trigger_basis == "contractual"
    assert eod.knocked_in_before is True  # 74.5 <= the raw 75

    intraday = build_schedule(
        _contract(),
        market,
        ko_shift=shift,
        ki_shift=shift,
        trigger_basis="effective",
    )
    assert intraday.trigger_basis == "effective"
    assert intraday.knocked_in_before is False  # 74.5 > the shifted 74.25
    assert intraday.ki_levels[0] == pytest.approx(74.25)


def test_trigger_basis_reads_a_same_day_knock_out_off_the_shifted_line():
    """The knock-out side follows the same switch: 99.5 clears the shifted 99 but
    not the raw 100, so only the intraday basis settles the fixing on the date."""
    # 2026-04-05 is the first observation: a fixing printed on the valuation date.
    # The packaged knock-out rule steps over the periods, so the first
    # observation's effective line is 100 x (1 - 1% x 1/3) = 99.667.
    market = _market(spot=99.8, valuation=datetime(2026, 4, 5, 15, 0))
    shift = {"mode": "relative", "value": -0.01}

    eod = build_schedule(_contract(), market, ko_shift=shift)
    assert eod.is_settled is False
    assert len(eod.observation_dates) == 2  # observed today, did not trigger, dropped

    intraday = build_schedule(
        _contract(), market, ko_shift=shift, trigger_basis="effective"
    )
    assert intraday.is_settled is True
    # it settles at that observation's own accrued coupon (Jan 5 -> Apr 5, act/365)
    assert intraday.knocked_out_cash == pytest.approx(1.0e6 * (1 + 0.20 * 90 / 365))


def test_trigger_basis_applies_to_a_payload_fed_schedule():
    """The flag rides the JSON entry point too: same payload, same switch."""
    payload = build_schedule(_contract(), _market()).to_dict()
    market = _market(spot=74.5, valuation=datetime(2026, 2, 2, 15, 0))

    assert AutocallSchedule.from_dict(payload, market).knocked_in_before is True
    intraday = AutocallSchedule.from_dict(payload, market, trigger_basis="effective")
    assert intraday.trigger_basis == "effective"
    assert intraday.knocked_in_before is False


def test_an_unknown_trigger_basis_is_refused():
    assert resolve_trigger_basis(None) == "contractual"
    assert resolve_trigger_basis("Effective") == "effective"
    with pytest.raises(ValueError, match="trigger basis"):
        resolve_trigger_basis("intraday")


def test_the_knock_in_state_is_derived_against_the_valuation_date():
    """The ledger stores the date; the state follows from it and the valuation.

    That is what makes back-dating work: no payload edit, just an earlier
    valuation date, and the trade prices pre-knock-in again.
    """
    market = _market(valuation=LATER)
    schedule = build_schedule(
        apply_history(_elapsed(history=((date(2026, 4, 5), 70.0),)), market), market
    )
    payload = schedule.to_dict()

    assert payload["knocked_in_date"] == "2026-04-05"
    assert "knocked_in_before" not in payload

    after = AutocallSchedule.from_dict(payload, market)
    assert after.knocked_in_date == datetime(2026, 4, 5)
    assert after.knocked_in_before is True

    # the same payload valued *before* the knock-in: the date is in the future
    # relative to that valuation, which reads as "not knocked in yet"
    earlier = _market(valuation=datetime(2026, 1, 6, 15, 0))
    before = AutocallSchedule.from_dict(payload, earlier)
    assert before.knocked_in_date == datetime(2026, 4, 5)
    assert before.knocked_in_before is False

    # and the switch survives rebasing (what the bumps and the rungs use)
    assert schedule.rebased(earlier).knocked_in_before is False


def test_the_retired_knocked_in_before_key_is_refused():
    payload = build_schedule(_contract(), _market()).to_dict()
    payload["knocked_in_before"] = True

    with pytest.raises(ValueError, match="knocked_in_date"):
        AutocallSchedule.from_dict(payload, _market())


def test_a_knock_in_date_off_the_monitoring_grid_is_refused():
    market = _market(valuation=LATER)

    # 2026-04-04 is a Saturday: the daily grid never monitors it
    with pytest.raises(ValueError, match="monitoring date"):
        build_schedule(
            _contract(ki_frequency="daily", knocked_in_date=date(2026, 4, 4)), market
        )

    # 2026-04-05 is an observation date: monitored whether or not it is open
    assert (
        build_schedule(
            _contract(
                ki_frequency="observation_dates", knocked_in_date=date(2026, 4, 5)
            ),
            market,
        ).knocked_in_before
        is True
    )


def test_the_knock_in_leg_scales_with_gearing_and_protection():
    for overrides, expected in (
        ({"protected_principal": 0.90}, 0.90e6),
        ({"ki_gearing": 2.0}, 0.20e6),
    ):
        schedule, _ = _resolved(**overrides)
        assert expiry_cash_flow(schedule, 60.0, True) == pytest.approx(expected)


def test_ki_strike_moves_the_loss_leg_off_the_start_spot():
    """OTM structures: the short put is struck below the start spot."""
    # 60% strike: no loss at or above 60%, straight participation below
    for spot, expected in ((80.0, 1.0e6), (60.0, 1.0e6), (30.0, 0.50e6)):
        schedule, _ = _resolved(ki_strike=0.60)
        assert expiry_cash_flow(schedule, spot, True) == pytest.approx(expected)

    # gearing and the protection floor compose with it
    schedule, _ = _resolved(ki_strike=0.60, ki_gearing=2.0)
    assert expiry_cash_flow(schedule, 30.0, True) == pytest.approx(0.0)
    schedule, _ = _resolved(ki_strike=0.60, protected_principal=0.75)
    assert expiry_cash_flow(schedule, 30.0, True) == pytest.approx(0.75e6)

    with pytest.raises(ValueError, match="ki_strike"):
        _contract(ki_strike=0.0)


def test_a_step_up_coupon_prices_every_observation_on_its_own_rate():
    schedule, _ = _resolved(annual_coupon=[0.10, 0.12, 0.14])

    assert schedule.coupon_rates == pytest.approx((0.10, 0.12, 0.14))
    assert ko_cash_flow(schedule, 0) == pytest.approx(1.0e6 * (1 + 0.10 * 90 / 365.0))
    assert ko_cash_flow(schedule, 2) == pytest.approx(1.0e6 * (1 + 0.14 * 273 / 365.0))

    with pytest.raises(ValueError, match="one value or one per observation"):
        _contract(annual_coupon=[0.10, 0.12])


def test_the_rebate_leg_is_independent_of_the_knock_out_coupon():
    schedule, _ = _resolved(annual_coupon=0.10, rebate=0.05)

    # 5% annual over the 273 days to expiry, nowhere near the 10% KO coupon
    assert rebate_cash_flow(schedule) == pytest.approx(1.0e6 * (1 + 0.05 * 273 / 365.0))
    assert expiry_cash_flow(schedule, 120.0, False) == pytest.approx(
        rebate_cash_flow(schedule)
    )

    # default: the last observation's coupon, accrued to expiry
    plain, _ = _resolved(annual_coupon=[0.10, 0.20, 0.30])
    assert plain.rebate_ratio == pytest.approx(1 + 0.30 * 273 / 365.0)

    with pytest.raises(ValueError, match="rebate"):
        _contract(rebate=-0.01)


def test_the_resolved_payload_round_trips_and_stays_editable():
    """The JSON layer: export the resolved contract, edit it, price it again."""
    schedule, market = _resolved(annual_coupon=[0.10, 0.12, 0.14])
    payload = json.loads(json.dumps(schedule.to_dict()))

    rebuilt = AutocallSchedule.from_dict(payload, market)

    assert rebuilt.coupon_rates == pytest.approx(schedule.coupon_rates)
    # the payload carries the term-sheet levels, so the shift is re-applied on read
    assert payload["observations"][0]["ko"] == pytest.approx(schedule.ko_levels_raw[0])
    assert rebuilt.ko_levels_raw == pytest.approx(schedule.ko_levels_raw)
    assert rebuilt.ko_levels == pytest.approx(schedule.ko_levels)
    assert rebuilt.ki_dates == schedule.ki_dates
    assert rebuilt.ki_monitor_levels == pytest.approx(schedule.ki_monitor_levels)
    assert rebuilt.rebate_ratio == pytest.approx(schedule.rebate_ratio)
    assert rebuilt.rebate_rate == pytest.approx(schedule.rebate_rate)
    assert rebuilt.notional == schedule.notional
    assert rebuilt.notes == schedule.notes

    # the rebate is spelled as an annual rate, like ``coupon_rate``
    assert payload["rebate"] == pytest.approx(0.14)  # the last observation's coupon
    assert "ko_raw" not in payload["observations"][0]

    # a hand edit - a step-down knock-out (term-sheet level) and a higher rebate -
    # is priced as written, shift and all
    payload["observations"][1]["ko"] = schedule.ko_levels_raw[1] * 0.97
    payload["rebate"] = 0.02
    edited = AutocallSchedule.from_dict(payload, market)
    assert edited.ko_levels[1] == pytest.approx(schedule.ko_levels[1] * 0.97, rel=1e-3)
    assert edited.rebate_ratio == pytest.approx(1 + 0.02 * 273 / 365.0)

    # an old payload spelling the rebate as a total ratio is refused, loudly
    with pytest.raises(ValueError, match="annual rate"):
        AutocallSchedule.from_dict({**payload, "rebate": 1.02}, market)

    # ... and the quote wrapper of ``--json`` is accepted too
    assert AutocallSchedule.from_dict({"contract": payload}, market).ko_levels == (
        edited.ko_levels
    )


def test_rebased_moves_only_the_market_side():
    schedule, market = _resolved()

    moved = schedule.rebased(market.clone(spot=market.spot * 0.95))

    assert moved.observation_dates == schedule.observation_dates
    assert moved.ko_levels == schedule.ko_levels
    assert moved.coupon_rates == schedule.coupon_rates
    assert moved.spot0 == schedule.spot0  # contractual anchor, not the bumped spot


def test_rebased_rolls_the_future_view_forward():
    """A later valuation date drops the observations and monitoring days behind it."""
    schedule, market = _resolved()
    later = market.clone(valuation_date=datetime(2026, 5, 1, 15, 0))

    rolled = schedule.rebased(later)

    assert rolled.observation_dates == (datetime(2026, 7, 5), datetime(2026, 10, 5))
    assert rolled.coupon_rates == schedule.coupon_rates[1:]
    assert all(day > later.valuation_date for day in rolled.ki_dates)
    assert len(rolled.ki_monitor_levels) == len(rolled.ki_dates)

    # the market side moved with the valuation date
    assert rolled.valuation_date == later.valuation_date
    assert rolled.vol_times == pytest.approx(
        tuple(later.year_fraction(day) for day in rolled.observation_dates)
    )
    assert rolled.discount_factors == pytest.approx(
        tuple(later.discount_factor(day) for day in rolled.payment_dates)
    )


def test_a_settled_payload_round_trips():
    market = _market(valuation=LATER)
    contract = apply_history(
        _contract(
            ki_frequency="observation_dates",
            history=((date(2026, 4, 5), 101.0),),
        ),
        market,
    )
    schedule = build_schedule(contract, market)
    assert schedule.is_settled

    rebuilt = AutocallSchedule.from_dict(schedule.to_dict(), market)

    assert rebuilt.is_settled
    assert rebuilt.knocked_out_cash == pytest.approx(schedule.knocked_out_cash)
    assert rebuilt.knocked_out_payment_date == schedule.knocked_out_payment_date


def test_expiry_knock_in_monitors_maturity_only():
    contract = _contract(ki_frequency="expiry")

    assert contract.ki_monitoring_dates(
        None, date(2026, 1, 5), date(2026, 10, 5)
    ) == (datetime(2026, 10, 5),)
    # one spelling per rule: the old aliases (``at_expiry`` / ``maturity`` / ``obs``)
    # are refused instead of translated
    for alias in ("at_expiry", "maturity", "obs"):
        with pytest.raises(ValueError, match="no aliases"):
            _contract(ki_frequency=alias)

    market = _market()
    schedule = build_schedule(contract, market)
    grid = build_time_grid(schedule, market, steps_per_observation=1)

    assert grid.ki_steps == (len(grid.dates) - 1,)  # the expiry node, nothing else
    assert grid.ki_steps != grid.observation_steps


# ------------------------------------------------------------------ plumbing
def test_contract_validation():
    with pytest.raises(ValueError):
        _contract(observation_dates=())
    with pytest.raises(ValueError):
        _contract(ko_levels=(1.0, 0.9))
    with pytest.raises(ValueError):
        _contract(observation_dates=(date(2025, 12, 5),))
    with pytest.raises(ValueError):
        _contract(ko_levels=(0.0,))
    with pytest.raises(ValueError):
        _contract(ki_level=-0.1)
    with pytest.raises(ValueError):
        _contract(annual_coupon=-0.01)
    with pytest.raises(ValueError):
        _contract(protected_principal=1.5)
    with pytest.raises(ValueError):
        _contract(settlement_days=-1)
    with pytest.raises(ValueError):
        _contract(anchor="mid")
    with pytest.raises(ValueError):
        _contract(expiry_date=date(2025, 12, 31))


def test_schedule_is_frozen_and_comparable():
    first = build_schedule(_contract(), _market())
    again = build_schedule(_contract(), _market())
    shifted = build_schedule(_contract(), _market(), ko_shift=-0.05)

    # identical inputs -> equal schedules (the engines' cross-check invariant)
    assert first == again
    assert first != shifted
    assert first.ko_levels != shifted.ko_levels

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.spot0 = 1.0


def test_pricer_registry_covers_the_autocallable_aliases():
    for alias in ("autocallable", "autocall", "snowball"):
        assert get_pricer(alias) is not None
        assert alias in available_product_types()

    with pytest.raises(ValueError):
        pricer_for("bogus")
