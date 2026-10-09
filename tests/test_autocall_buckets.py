"""Bucketed Greeks for the snowball, on both engines.

The buckets follow the vanilla convention (``pricing.risk.buckets``): vega per
surface expiry, rhoQ / rho per curve pillar, and bucketed delta split out of the
spot cash delta.  The assertions here are the structural identities the vanilla
side is tested with - a single-expiry surface makes the vega bucket equal the
parallel vega, and ``sum(bucketed_delta) == delta_cash`` holds by construction -
plus a cross-engine comparison of the vega bucket.
"""

from datetime import date, datetime

import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.exotics.autocall import AutocallContract, apply_history
from surface_pricer.pricing.exotics.autocall.mc import AutocallMonteCarlo
from surface_pricer.pricing.exotics.autocall.pde import AutocallPDE
from surface_pricer.pricing.results import RiskSettings
from surface_pricer.pricing.risk.diff import BUCKET_NAMES, GREEK_NAMES

VALUATION = datetime(2026, 1, 5, 15, 0)
EXPIRY = datetime(2026, 10, 5)
OBSERVATIONS = (date(2026, 4, 5), date(2026, 7, 5), date(2026, 10, 5))
SPOT = 100.0


def _market():
    """One surface expiry and flat curves (the vanilla bucket test setup)."""
    surface = EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=[EXPIRY],
        atm_vols=[0.20],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(0.02, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(0.01, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
        surface=surface,
    )


def _contract():
    return AutocallContract(
        underlying="MO",
        start_date=VALUATION,
        expiry_date=EXPIRY,
        observation_dates=OBSERVATIONS,
        ko_levels=(1.03,),
        ki_level=0.70,
        ki_frequency="observation_dates",  # 3 windows: keeps the test quick
        annual_coupon=0.20,
        notional=1.0e6,
        start_spot=SPOT,
    )


def _engine(kind: str):
    """Both engines on the same 6 sub-steps per observation window."""
    if kind == "pde":
        return AutocallPDE(nodes=201)
    return AutocallMonteCarlo(paths=2048, steps_per_observation=6)


@pytest.mark.parametrize("kind", ["pde", "mc"])
def test_every_bucket_is_filled_and_adds_up(kind):
    market = _market()
    contract = apply_history(_contract(), market)
    settings = RiskSettings(greeks=GREEK_NAMES + BUCKET_NAMES)

    result = _engine(kind).greeks(contract, market, settings)

    assert set(result.bucketed_vega) == {EXPIRY.date().isoformat()}
    assert set(result.bucketed_rhoq) == set(result.bucketed_delta)
    assert result.bucketed_rhoq, "flat curves bucket over the default tenor grid"
    assert set(result.bucketed_rho) == set(result.bucketed_rhoq)
    # one surface expiry: the single-pillar bump IS the parallel bump
    assert sum(result.bucketed_vega.values()) == pytest.approx(result.vega, rel=1e-9)
    # bucketed delta is the spot cash delta distributed over the borrow buckets
    assert sum(result.bucketed_delta.values()) == pytest.approx(
        result.delta_cash, rel=1e-9
    )


@pytest.mark.parametrize("kind", ["pde", "mc"])
def test_buckets_are_opt_in(kind):
    """An ordinary risk run must not pay for (or report) any bucket."""
    market = _market()
    contract = apply_history(_contract(), market)

    result = _engine(kind).greeks(contract, market, RiskSettings(greeks=GREEK_NAMES))

    assert result.delta is not None
    assert result.bucketed_vega == {}
    assert result.bucketed_rhoq == {}
    assert result.bucketed_rho == {}
    assert result.bucketed_delta == {}


def test_mc_and_pde_agree_on_the_vega_bucket():
    """The bucket is the parallel bump here, so this compares the two methods.

    Both sides run 6 sub-steps per observation window; the remaining gap is the
    usual MC/PDE method difference (MC vega carries a few percent of seed noise
    at these path counts).
    """
    market = _market()
    contract = apply_history(_contract(), market)
    settings = RiskSettings(greeks=("vega", "bucketed_vega"))

    mc = AutocallMonteCarlo(paths=8192, steps_per_observation=6).greeks(
        contract, market, settings
    )
    pde = AutocallPDE(nodes=401).greeks(contract, market, settings)

    label = EXPIRY.date().isoformat()
    assert mc.vega != 0.0  # the contract must actually carry vol sensitivity
    assert mc.bucketed_vega[label] == pytest.approx(pde.bucketed_vega[label], rel=0.25)
    assert (mc.bucketed_vega[label] - mc.vega) == pytest.approx(
        pde.bucketed_vega[label] - pde.vega, abs=abs(pde.vega) * 0.25
    )


def test_the_auto_bucket_grid_coarsens_and_stops_at_the_horizon():
    """15 pillars are not 15 bump pairs: the long end merges, the far end goes."""
    from surface_pricer.core.curves import PiecewiseRateCurve
    from surface_pricer.pricing.risk.buckets import bucket_grid, bucket_grid_info

    curve = PiecewiseRateCurve(
        anchor=VALUATION,
        tenors=["1M", "3M", "6M", "1Y", "15M", "18M", "2Y", "3Y", "4Y", "5Y"],
        rates=[0.01] * 10,
    )
    market = _market().clone(borrow_curve=curve)
    horizon = datetime(2029, 6, 1)  # the trade's last payment

    groups = bucket_grid(
        curve.pillar_dates,
        valuation=VALUATION,
        horizon=horizon,
        group_after="1Y",
        calendar=market.calendar,
    )
    labels = [label for label, _ in groups]
    sizes = [len(members) for _, members in groups]

    # up to 1Y one bucket per pillar, then one per year of tenor (15M + 18M + 2Y
    # meet in the 2Y band), and the 5Y pillar is past the horizon
    assert sizes == [1, 1, 1, 1, 3, 1, 1]
    assert labels == [
        "2026-02-04", "2026-04-05", "2026-07-04", "2027-01-05",
        "2028-01-05", "2029-01-05", "2030-01-07",
    ]
    # the pillar bracketing the horizon stays: the segment after it is the first
    # that carries no weight
    assert [member.date().isoformat() for member in groups[-1][1]] == ["2030-01-04"]

    info = bucket_grid_info(market, RiskSettings(), horizon=horizon)
    assert (info["pillars"], info["buckets"], info["dropped"]) == (10, 7, 1)
    assert info["group_after"] == "1Y"
    assert info["horizon"] == horizon.date().isoformat()
    assert info["explicit"] is False

    # a pinned grid is used as given (no merging, no trimming), and switching the
    # policy off reproduces the old one-bucket-per-pillar grid
    assert len(bucket_grid(curve.pillar_dates, valuation=VALUATION, horizon=horizon, group_after=None)) == 10


def test_the_coarse_grid_is_still_a_decomposition():
    """Coarser buckets, same total: they add up to the parallel Greek."""
    from surface_pricer.pricing.exotics.autocall import build_schedule
    from surface_pricer.reporting.autocall_report import format_autocall

    market = _market()
    contract = apply_history(_contract(), market)
    schedule = build_schedule(contract, market)
    settings = RiskSettings(
        greeks=("delta_cash", "rhoq", "bucketed_rhoq", "bucketed_delta")
    )

    result = AutocallPDE(nodes=201).greeks(contract, market, settings)

    # every kept pillar is bumped, in groups, and the dropped ones sit past the
    # trade's last payment where a locally interpolated curve has no weight.  The
    # remaining gap is the known flat-vs-rebuilt curve composition (both sides are
    # the same 8 pillars; the rebuilt curve prices off its anchor timestamp).
    assert sum(result.bucketed_rhoq.values()) == pytest.approx(result.rhoq, rel=1e-4)
    assert sum(result.bucketed_delta.values()) == pytest.approx(
        result.delta_cash, rel=1e-9
    )

    info = result.metadata["bucket_grid"]
    assert info["pillars"] == 8  # edslib's tenor grid: a flat curve has no pillars
    assert info["buckets"] < info["pillars"]
    assert info["dropped"] >= 1
    assert "bucket grid:" in format_autocall(schedule, result, market)


def test_bucketed_delta_is_distributed_by_sensitivity_share():
    from surface_pricer.pricing.risk.buckets import convert_bucketed_delta

    buckets = {"2026-04-05": -2.0, "2026-07-05": -30.0, "2026-10-05": -70.0}

    split = convert_bucketed_delta(buckets, delta_cash=102.0)

    assert sum(split.values()) == pytest.approx(102.0, rel=1e-12)
    assert split["2026-07-05"] / split["2026-04-05"] == pytest.approx(15.0)
    assert convert_bucketed_delta(buckets, 0.0) == {}
    assert convert_bucketed_delta({}, 100.0) == {}
