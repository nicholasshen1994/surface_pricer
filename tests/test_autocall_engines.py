"""Cross-engine checks: both engines consume the same effective terms, and MC
and PDE agree within their numerical error on the same contract and market."""

import dataclasses
import math
from datetime import date, datetime, timedelta

import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.exotics.autocall import AutocallContract, build_schedule
from surface_pricer.pricing.exotics.autocall.history import apply_history
from surface_pricer.pricing.exotics.autocall.mc import AutocallMonteCarlo
from surface_pricer.pricing.exotics.autocall.pde import AutocallPDE
from surface_pricer.pricing.results import RiskSettings

VALUATION = datetime(2026, 1, 5, 15, 0)
OBSERVATIONS = (date(2026, 4, 5), date(2026, 7, 5), date(2026, 10, 5))
EXPIRY = datetime(2026, 10, 5)
SPOT = 100.0


def _market(atm_vols=(0.20,), **surface_kwargs):
    pillars = [EXPIRY] if len(atm_vols) == 1 else [date(2026, 7, 5), EXPIRY]
    surface = EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=pillars,
        atm_vols=list(atm_vols),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
        **surface_kwargs
    )
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(0.02, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(0.0, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
        surface=surface,
    )


def _contract(**overrides):
    params = dict(
        underlying="MO",
        start_date=VALUATION,
        expiry_date=EXPIRY,
        observation_dates=OBSERVATIONS,
        ko_levels=(1.0,),
        ki_level=0.75,
        annual_coupon=0.20,
        notional=1.0e6,
        start_spot=SPOT,
    )
    params.update(overrides)
    return AutocallContract(**params)


# ---------------------------------------------------- knock-in observation
def _count_business_days(calendar, start, end):
    total = 0
    day = start
    while day <= end:
        total += 1 if calendar.is_business_day(day) else 0
        day += timedelta(days=1)
    return total


@pytest.mark.parametrize("name", ["mc", "pde"])
def test_a_settled_trade_reports_the_cash_and_no_greeks(name):
    """Knocked out is one discounted cash flow: the NPV is the accrued coupon and
    every Greek is zero - there is nothing left to bump."""
    later = datetime(2026, 5, 1, 15, 0)
    market = dataclasses.replace(_market(), valuation_date=later)
    contract = apply_history(
        _contract(
            ki_frequency="observation_dates",
            history=((date(2026, 4, 5), SPOT + 1.0),),  # above the 100% knock-out
        ),
        market,
    )
    schedule = build_schedule(contract, market)
    assert schedule.is_settled

    engine = {"mc": AutocallMonteCarlo, "pde": AutocallPDE}[name]()
    result = engine.greeks_schedule(
        schedule,
        market,
        RiskSettings(
            greeks=(
                "delta",
                "gamma_cash",
                "vega",
                "volga",
                "vanna",
                "theta",
                "rho",
                "rhoq",
            )
        ),
    )

    assert result.npv == pytest.approx(
        schedule.knocked_out_cash * schedule.knocked_out_discount_factor
    )
    for field in ("delta", "gamma_cash", "vega", "volga", "vanna", "theta", "rho", "rhoq"):
        assert getattr(result, field) == 0.0
    assert result.bucketed_vega == {} and result.bucketed_delta == {}


def test_daily_knock_in_prices_below_the_observation_only_convention():
    """The market snowball observes the knock-in every business day; the engine
    walks that grid, so it must price strictly below the simplified convention
    that only looks at the knock-out observations."""
    market = _market(atm_vols=(0.25,))
    pde = AutocallPDE(nodes=301)

    observed = pde.price(_contract(ki_frequency="observation_dates"), market).npv
    daily = pde.price(_contract(ki_frequency="daily"), market).npv

    assert daily < observed
    assert observed - daily > 0.005 * daily  # a material, measurable gap


def test_broadie_glasserman_shift_moves_the_periodic_knock_in_the_same_way():
    """edslib's ``optimize_ki_observation`` shortcut keeps the periodic schedule
    and shifts the barrier by the Broadie-Glasserman factor.  It is an
    approximation, but it must point the same way as walking the daily grid and
    land closer to it than the unadjusted convention does."""
    market = _market(atm_vols=(0.25,))
    calendar = market.calendar
    pde = AutocallPDE(nodes=301)

    observed = pde.price(_contract(ki_frequency="observation_dates"), market).npv
    daily = pde.price(_contract(ki_frequency="daily"), market).npv

    beta = 0.5826  # -zeta(0.5) / sqrt(2 * pi)
    n_bd = _count_business_days(calendar, date(2026, 1, 1), date(2026, 12, 31))
    n_ki = _count_business_days(calendar, date(2026, 1, 6), date(2026, 10, 5))
    n_periodic = len(OBSERVATIONS)
    adjustment = math.exp(
        beta
        * 0.25
        * (math.sqrt(n_ki / (n_bd * n_periodic)) - 1.0 / math.sqrt(n_bd))
    )
    assert adjustment > 1.0  # a periodic knock-in must be pushed towards the spot

    shifted = pde.price(
        _contract(ki_level=0.75 * adjustment, ki_frequency="observation_dates"),
        market,
    ).npv

    assert shifted < observed
    assert abs(shifted - daily) < abs(observed - daily)


def test_expiry_knock_in_is_the_least_valuable_barrier_to_the_investor():
    """The European knock-in (``ki_frequency="expiry"``) only tests maturity, so
    it must price above every monitored convention - a path that dips and
    recovers keeps its coupon instead of losing it."""
    market = _market(atm_vols=(0.25,))
    pde = AutocallPDE(nodes=301)

    european = pde.price(_contract(ki_level=0.95, ki_frequency="expiry"), market).npv
    observed = pde.price(
        _contract(ki_level=0.95, ki_frequency="observation_dates"), market
    ).npv
    daily = pde.price(_contract(ki_level=0.95, ki_frequency="daily"), market).npv

    assert observed < european
    assert daily < observed


def test_otm_ki_strike_prices_above_the_plain_loss_leg():
    """``ki_strike`` moves the short put off the start spot: an OTM strike can
    only lose less, so the structure is worth more."""
    market = _market(atm_vols=(0.25,))
    pde = AutocallPDE(nodes=301)

    plain = pde.price(_contract(ki_level=0.95, ki_frequency="expiry"), market).npv
    otm = pde.price(
        _contract(ki_level=0.95, ki_strike=0.80, ki_frequency="expiry"), market
    ).npv

    assert otm > plain


# ---------------------------------------------------------- shared schedule
def test_both_engines_price_the_same_schedule():
    market = _market()
    contract = _contract()
    schedule = build_schedule(contract, market)

    mc = AutocallMonteCarlo(paths=2048, seed=3).price_schedule(schedule, market)
    pde = AutocallPDE().price_schedule(schedule, market)

    for key in ("observation_dates", "ko_levels", "ki_levels", "spot0", "anchored_on"):
        assert mc.metadata[key] == pde.metadata[key]
    assert mc.metadata["ko_shift"] == pde.metadata["ko_shift"]
    assert mc.metadata["ki_shift"] == pde.metadata["ki_shift"]
    assert mc.metadata["barrier_smooth"] == pde.metadata["barrier_smooth"]


# ------------------------------------------------------------- agreement
def test_mc_and_pde_agree_on_a_flat_surface():
    market = _market(atm_vols=(0.20,))
    # keep the knock-out away from the spot: sitting exactly on it the
    # knock-out decision is at its most sensitive and the two discrete schemes
    # legitimately differ by more
    contract = _contract(ko_levels=(1.05,))

    mc = AutocallMonteCarlo(paths=65536, seed=7).price(contract, market)
    pde = AutocallPDE().price(contract, market)

    difference = abs(mc.npv - pde.npv)
    error = mc.metadata["std_error"]
    assert difference < max(6.0 * error, 0.015 * abs(pde.npv))
    assert pde.npv > 0.0


def test_mc_and_pde_agree_with_a_smile():
    market = _market(atm_vols=(0.24, 0.20), skews=(0.15, -0.10), convs=(0.25, 0.20))
    contract = _contract(ko_levels=(1.05,))

    mc = AutocallMonteCarlo(paths=65536, seed=11).price(contract, market)
    pde = AutocallPDE().price(contract, market)

    difference = abs(mc.npv - pde.npv)
    error = mc.metadata["std_error"]
    assert difference < max(6.0 * error, 0.02 * abs(pde.npv))


def test_agreement_holds_with_smoothing_off():
    market = _market(atm_vols=(0.20,))
    contract = _contract(ko_levels=(1.05,))

    mc = AutocallMonteCarlo(paths=65536, seed=5, smooth=False).price(contract, market)
    pde = AutocallPDE(smooth=False).price(contract, market)

    difference = abs(mc.npv - pde.npv)
    error = mc.metadata["std_error"]
    assert difference < max(6.0 * error, 0.02 * abs(pde.npv))


# ------------------------------------------------------------ convergence
def test_pde_converges_with_more_nodes():
    market = _market(atm_vols=(0.20,))
    contract = _contract()

    coarse = AutocallPDE(nodes=201).price(contract, market).npv
    fine = AutocallPDE(nodes=1201).price(contract, market).npv

    assert abs(fine - coarse) / fine < 0.02


def test_mc_standard_error_shrinks_with_more_paths():
    market = _market(atm_vols=(0.20,))
    contract = _contract()

    few = AutocallMonteCarlo(paths=4096, seed=9).price(contract, market)
    many = AutocallMonteCarlo(paths=65536, seed=9).price(contract, market)

    assert many.metadata["std_error"] < few.metadata["std_error"]
    assert abs(many.npv - few.npv) < 5.0 * few.metadata["std_error"]


# -------------------------------------------------------- analytic extremes
def test_extreme_knock_out_collapses_to_a_certain_cash_flow():
    market = _market(atm_vols=(0.20,))
    contract = _contract(ko_levels=(0.2,), ki_level=0.01)  # KO 20 -> always

    mc = AutocallMonteCarlo(paths=2048, seed=2).price(contract, market)
    pde = AutocallPDE().price(contract, market)

    expected = 1.0e6 * (1 + 0.20 * 90 / 365.0) * market.discount_factor(
        datetime(2026, 4, 5)
    )
    assert mc.npv == pytest.approx(expected, rel=1e-6)
    assert pde.npv == pytest.approx(expected, rel=1e-3)


def test_higher_coupon_raises_the_value():
    market = _market(atm_vols=(0.20,))

    low = AutocallPDE().price(_contract(annual_coupon=0.05), market).npv
    high = AutocallPDE().price(_contract(annual_coupon=0.30), market).npv

    assert high > low


def test_lower_knock_in_makes_the_put_cheaper():
    market = _market(atm_vols=(0.25,))

    close_ki = AutocallPDE().price(_contract(ki_level=0.95), market).npv
    far_ki = AutocallPDE().price(_contract(ki_level=0.50), market).npv

    # a closer knock-in is more likely to trigger the short put
    assert close_ki < far_ki
