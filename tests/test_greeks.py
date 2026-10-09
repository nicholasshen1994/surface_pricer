"""Greeks: closed-form Black-76 cross-checks and bucket conventions."""

from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import norm

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar, year_fraction
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.vanilla import VanillaContract, calculate_greeks, convert_bucketed_delta
from surface_pricer.pricing.results import RiskSettings
from surface_pricer.pricing.vanilla import VanillaPricer

VALUATION = datetime(2026, 9, 28, 15, 0)
EXPIRY = datetime(2027, 3, 19, 15, 0)
SPOT = 7500.0
STRIKE = 7600.0
RATE = 0.02
BORROW = 0.01
VOL = 0.22


def _flat_surface(pillars=(EXPIRY,), vol=VOL):
    return EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=list(pillars),
        atm_vols=[vol] * len(pillars),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


def _market(pillars=(EXPIRY,), vol=VOL, borrow=BORROW):
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(RATE, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(borrow, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
        surface=_flat_surface(pillars=pillars, vol=vol),
    )


def _call(option_type="call"):
    return VanillaContract(expiry=EXPIRY, strike=STRIKE, option_type=option_type)


def _calendar_year_fraction(value: float) -> float:
    return year_fraction(VALUATION, value, basis="act/365f")


def test_greeks_match_black76_closed_form():
    market = _market()
    contract = _call()
    result = calculate_greeks(contract, market)

    tau = market.year_fraction(EXPIRY)
    forward = market.forward(EXPIRY)
    df = market.discount_factor(EXPIRY)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(forward / STRIKE) + 0.5 * VOL ** 2 * tau) / (VOL * sqrt_tau)
    d2 = d1 - VOL * sqrt_tau
    moneyness_ratio = forward / SPOT

    # price
    expected_price = df * (forward * norm.cdf(d1) - STRIKE * norm.cdf(d2))
    assert result.npv == pytest.approx(expected_price, rel=1e-8)
    assert result.implied_vol == pytest.approx(VOL, rel=1e-9)

    # delta family
    expected_delta = df * norm.cdf(d1) * moneyness_ratio
    assert result.delta == pytest.approx(expected_delta, rel=1e-4)
    # delta_cash / delta_n are exact rescalings of delta ...
    assert result.delta_cash == pytest.approx(result.delta * SPOT, rel=1e-12)
    assert result.delta_n == pytest.approx(result.delta / SPOT, rel=1e-12)
    # ... while delta itself carries the (small) finite-difference error
    assert result.delta_cash == pytest.approx(expected_delta * SPOT, rel=1e-4)

    # gamma: d2NPV/dS2 with F = S * k
    expected_gamma = (
        df * norm.pdf(d1) * moneyness_ratio ** 2 / (forward * VOL * sqrt_tau)
    )
    assert result.gamma == pytest.approx(expected_gamma, rel=1e-3)
    # cash gamma is the same convexity per (1% spot move)^2
    assert result.gamma_cash == pytest.approx(
        result.gamma * SPOT ** 2 / 100.0, rel=1e-12
    )

    # vega / volga (reported per vol point / per vol point squared)
    vega_raw = df * forward * norm.pdf(d1) * sqrt_tau
    assert result.vega == pytest.approx(vega_raw / 100.0, rel=1e-4)
    expected_volga = vega_raw * d1 * d2 / VOL
    assert result.volga == pytest.approx(expected_volga / 10000.0, rel=2e-2)

    # vanna: -df * phi(d1) * d2 / sigma, scaled by F/S
    expected_vanna = df * norm.pdf(d1) * (-d2 / VOL) * moneyness_ratio
    assert result.vanna == pytest.approx(expected_vanna / 100.0, rel=2e-2)

    # rho and rhoQ use the calendar (ACT/365F) year fraction
    tau_cal = _calendar_year_fraction(EXPIRY)
    assert result.rho == pytest.approx(STRIKE * tau_cal * df * norm.cdf(d2) / 100.0, rel=1e-2)
    assert result.rhoq == pytest.approx(
        -tau_cal * df * forward * norm.cdf(d1) / 100.0, rel=1e-2
    )


def test_put_greeks_match_black76_closed_form():
    market = _market()
    contract = _call("put")
    result = calculate_greeks(contract, market)

    tau = market.year_fraction(EXPIRY)
    forward = market.forward(EXPIRY)
    df = market.discount_factor(EXPIRY)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(forward / STRIKE) + 0.5 * VOL ** 2 * tau) / (VOL * sqrt_tau)
    d2 = d1 - VOL * sqrt_tau
    moneyness_ratio = forward / SPOT

    expected_price = df * (STRIKE * norm.cdf(-d2) - forward * norm.cdf(-d1))
    assert result.npv == pytest.approx(expected_price, rel=1e-8)
    assert result.delta == pytest.approx(-df * norm.cdf(-d1) * moneyness_ratio, rel=1e-4)
    tau_cal = _calendar_year_fraction(EXPIRY)
    assert result.rho == pytest.approx(
        -STRIKE * tau_cal * df * norm.cdf(-d2) / 100.0, rel=1e-2
    )


def test_theta_matches_direct_reprice():
    market = _market()
    contract = _call()
    result = calculate_greeks(contract, market)

    theta_market = market.clone(valuation_date=VALUATION + timedelta(days=1))
    expected = VanillaPricer(theta_market).npv(contract) - result.npv
    assert result.theta == pytest.approx(expected, rel=1e-9)
    assert result.theta < 0.0  # long option loses value with time


def test_cash_gamma_scales_with_the_notional():
    market = _market()
    base = calculate_greeks(_call(), market)
    scaled = calculate_greeks(
        VanillaContract(expiry=EXPIRY, strike=STRIKE, option_type="call", notional=25.0),
        market,
    )

    assert scaled.gamma_cash == pytest.approx(base.gamma_cash * 25.0, rel=1e-9)


def test_volume_scaling_is_linear_in_notional():
    market = _market()
    base = calculate_greeks(_call(), market)
    scaled = calculate_greeks(
        VanillaContract(expiry=EXPIRY, strike=STRIKE, option_type="call", notional=25.0),
        market,
    )

    assert scaled.npv == pytest.approx(base.npv * 25.0, rel=1e-12)
    assert scaled.delta == pytest.approx(base.delta * 25.0, rel=1e-9)
    assert scaled.vega == pytest.approx(base.vega * 25.0, rel=1e-9)
    assert scaled.volga == pytest.approx(base.volga * 25.0, rel=1e-6)
    assert scaled.rho == pytest.approx(base.rho * 25.0, rel=1e-6)
    assert scaled.bucketed_delta == pytest.approx(
        {key: value * 25.0 for key, value in base.bucketed_delta.items()}, rel=1e-6
    )


def test_bucketed_vega_sums_to_parallel_vega_for_flat_surface():
    second_pillar = datetime(2027, 6, 18, 15, 0)
    market = _market(pillars=(EXPIRY, second_pillar))
    result = calculate_greeks(_call(), market)

    assert set(result.bucketed_vega) == {
        EXPIRY.date().isoformat(),
        second_pillar.date().isoformat(),
    }
    # the vanilla expires on the first pillar, so only that bucket carries
    # sensitivity on a flat surface
    assert result.bucketed_vega[second_pillar.date().isoformat()] == pytest.approx(
        0.0, abs=1e-10
    )
    assert result.bucketed_vega[EXPIRY.date().isoformat()] == pytest.approx(
        result.vega, rel=1e-6
    )


def test_bucketed_rhoq_and_delta_convention():
    market = _market()
    result = calculate_greeks(_call(), market)

    assert result.bucketed_rhoq, "constant curves must fall back to edslib's tenor grid"
    assert set(result.bucketed_rhoq) == set(result.bucketed_delta)

    # the buckets distribute the spot delta_cash by sensitivity share
    total_rhoq = sum(result.bucketed_rhoq.values())
    assert sum(result.bucketed_delta.values()) == pytest.approx(
        result.delta_cash, rel=1e-12
    )
    for label, rhoq_bucket in result.bucketed_rhoq.items():
        assert result.bucketed_delta[label] == pytest.approx(
            result.delta_cash * rhoq_bucket / total_rhoq, rel=1e-12
        )

    # the flat curve is rebuilt on the bucket grid first (edslib's
    # rebuild_ql_curve_by_tenors), so every bucket is a single-pillar bump, and
    # the trade-aware grid (``bucket_grid``) keeps only what the option can see:
    # four pillars up to and including the one bracketing the expiry
    buckets = list(result.bucketed_rhoq.values())
    assert len(buckets) == 4
    grid = result.metadata["bucket_grid"]
    assert (grid["pillars"], grid["buckets"], grid["dropped"]) == (8, 4, 4)
    assert grid["horizon"] == EXPIRY.date().isoformat()
    # the expiry (2027-03-19) sits between the 3M and 6M pillars, so exactly
    # those two buckets move the forward - single-pillar bumps, not a parallel
    # shift - and the later pillar carries the larger interpolation weight
    assert [index for index, value in enumerate(buckets) if abs(value) > 0.0] == [2, 3]
    assert abs(buckets[3]) > abs(buckets[2]) > 0.0
    assert sum(buckets) == pytest.approx(result.rhoq, rel=1e-3)


def test_convert_bucketed_delta_distributes_delta_cash():
    buckets = {"2027-03-19": -1.5e-4, "2027-09-17": -2.5e-4}
    converted = convert_bucketed_delta(buckets, delta_cash=1000.0)

    assert sum(converted.values()) == pytest.approx(1000.0, rel=1e-12)
    assert converted["2027-03-19"] == pytest.approx(
        1000.0 * (-1.5e-4) / (-4.0e-4), rel=1e-12
    )

    # nothing to distribute when the sensitivity is flat or the delta is zero
    assert convert_bucketed_delta({}, 100.0) == {}
    assert convert_bucketed_delta(buckets, 0.0) == {}


def test_pillar_buckets_follow_explicit_inputs():
    market = _market()
    second_pillar = datetime(2027, 6, 18, 15, 0)
    result = calculate_greeks(
        _call(),
        market,
        bucketed_delta_pillars=[second_pillar],
        bucketed_vega_pillars=[EXPIRY],
    )

    assert list(result.bucketed_rhoq) == [second_pillar.date().isoformat()]
    assert list(result.bucketed_vega) == [EXPIRY.date().isoformat()]


def test_greek_conventions_are_documented():
    result = calculate_greeks(_call(), _market())
    conventions = result.metadata["greek_convention"]
    for name in (
        "delta",
        "gamma",
        "gamma_cash",
        "vega",
        "theta",
        "vanna",
        "volga",
        "rho",
        "rhoq",
        "bucketed_vega",
        "bucketed_delta",
    ):
        assert name in conventions


def test_vol_greeks_require_a_surface():
    market = _market()
    market = market.clone(surface=None)
    with pytest.raises(ValueError):
        calculate_greeks(_call(), market)


def test_bucketed_rhoq_follows_the_borrow_curve_pillars():
    """With a real borrow curve the buckets are its pillars, not a default grid."""
    from surface_pricer.core.curves import PiecewiseRateCurve

    pillar_days = [31, 92, 182, 366]
    curve = PiecewiseRateCurve(
        anchor=VALUATION,
        tenors=[float(days) for days in pillar_days],
        rates=[0.05, 0.06, 0.07, 0.08],
        basis="act/365f",
    )
    market = _market().clone(borrow_curve=curve)

    result = calculate_greeks(_call(), market)

    expected = [
        (VALUATION + timedelta(days=days)).date().isoformat() for days in pillar_days
    ]
    # the +366d pillar is past the expiry - and past the pillar bracketing it - so
    # the trade-aware grid drops it; the three the option can see stay
    assert list(result.bucketed_rhoq) == expected[:3]
    assert result.metadata["bucket_grid"]["dropped"] == 1

    values = list(result.bucketed_rhoq.values())
    # the expiry sits between the 92d and 182d pillars: only those two move the
    # forward, and the later one carries the larger interpolation weight
    assert values[0] == pytest.approx(0.0, abs=1e-15)
    assert abs(values[2]) > abs(values[1]) > 0.0
    assert sum(values) == pytest.approx(result.rhoq, rel=1e-3)


def test_bucketed_delta_distributes_exactly_for_otm_strikes():
    """An off-the-money strike pulls the delta stencil error and the borrow
    sensitivity apart - the share split must still reproduce delta_cash."""
    market = _market()
    contract = VanillaContract(expiry=EXPIRY, strike=STRIKE * 1.08, option_type="call")
    result = calculate_greeks(contract, market)

    assert result.delta_cash > 0.0
    assert sum(result.bucketed_delta.values()) == pytest.approx(
        result.delta_cash, rel=1e-12
    )


def test_bucketed_delta_sums_to_cash_delta_for_a_single_bucket():
    """The listed futures option case: the expiry sits on a borrow pillar.

    There the shared expiry-based ``tau`` equals the bucket's own dcf, so the
    conversion matches edslib's per-bucket variant and the bucket reproduces
    ``delta_cash``.
    """
    from surface_pricer.core.curves import PiecewiseRateCurve

    expiry_days = (EXPIRY.date() - VALUATION.date()).days
    curve = PiecewiseRateCurve(
        anchor=VALUATION.date(),
        tenors=[31.0, float(expiry_days)],
        rates=[BORROW, BORROW],
    )
    market = _market().clone(borrow_curve=curve)
    # a date-only expiry, as the CLI produces: the forward time then matches
    # the bucket dcf to the day
    contract = VanillaContract(
        expiry=EXPIRY.date(), strike=STRIKE, option_type="call"
    )
    result = calculate_greeks(contract, market)

    labels = list(result.bucketed_delta)
    assert labels == ["2026-10-29", EXPIRY.date().isoformat()]
    assert result.bucketed_rhoq[labels[0]] == pytest.approx(0.0, abs=1e-15)
    assert sum(result.bucketed_delta.values()) == pytest.approx(
        result.delta_cash, rel=1e-12
    )


def test_bucketed_delta_adds_up_when_the_expiry_is_off_the_bucket_grid():
    """One shared tau keeps the buckets additive off the grid.

    An expiry between two grid pillars is where a per-bucket dcf breaks the
    sum (it would scale each bucket by ``tau / dcf_bucket``); with the shared
    divisor the buckets add back up to ``delta_cash``.
    """
    market = _market()  # flat borrow -> edslib's 1M..2Y grid
    result = calculate_greeks(_call(), market)

    active = [v for v in result.bucketed_rhoq.values() if abs(v) > 0.0]
    assert len(active) == 2, "the expiry must sit between two grid pillars"
    assert sum(result.bucketed_delta.values()) == pytest.approx(
        result.delta_cash, rel=1e-12
    )


def test_intraday_valuation_keeps_the_near_bucket_at_zero():
    """A curve dated 00:00 with a 15:00 valuation must not stub into pillar 1.

    edslib's dcf runs on dates; an intraday stub would make ``zero_rate(start)``
    pick the left-extrapolated first pillar and leak a fake sensitivity into the
    nearest bucket.
    """
    from surface_pricer.core.curves import PiecewiseRateCurve

    curve = PiecewiseRateCurve(
        anchor=VALUATION.date(),  # anchor at midnight; the market values at 15:00
        tenors=[31.0, 92.0, 182.0, 366.0],
        rates=[0.05, 0.06, 0.07, 0.08],
        basis="act/365f",
    )
    market = _market().clone(borrow_curve=curve)
    assert market.valuation_date.hour == 15

    result = calculate_greeks(_call(), market)

    values = list(result.bucketed_rhoq.values())
    assert values[0] == pytest.approx(0.0, abs=1e-15)
    # the far pillar is past the expiry, so the grid drops it outright - and the
    # one bracketing the expiry still carries the sensitivity
    assert result.metadata["bucket_grid"]["dropped"] == 1
    assert abs(values[2]) > 0.0  # the 182d pillar still brackets the expiry
