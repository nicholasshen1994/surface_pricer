"""Monte Carlo autocallable engine: zero-vol analytic limits, reproducible
common random numbers and the payoff plumbing."""

import math
from datetime import date, datetime, timedelta

import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.exotics.autocall import AutocallContract, apply_history
from surface_pricer.pricing.exotics.autocall.grid import build_time_grid
from surface_pricer.pricing.exotics.autocall.mc import AutocallMonteCarlo, _normals

VALUATION = datetime(2026, 1, 5, 15, 0)
OBSERVATIONS = (date(2026, 4, 5), date(2026, 7, 5), date(2026, 10, 5))
EXPIRY = datetime(2026, 10, 5)
SPOT = 100.0


def _market(atm_vols=(0.20,), borrow=0.0, spot=SPOT, valuation=VALUATION):
    surface = EDSSabrSurface(
        init_date=valuation,
        init_spot=spot,
        expiry_dates=[EXPIRY],
        atm_vols=list(atm_vols),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )
    return MarketState(
        valuation_date=valuation,
        spot=spot,
        rate_curve=ConstantRateCurve(0.02, anchor=valuation),
        borrow_curve=ConstantRateCurve(borrow, anchor=valuation),
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
        ko_levels=(2.0,),
        ki_level=0.10,
        annual_coupon=0.20,
        notional=1.0e6,
        start_spot=SPOT,
    )
    params.update(overrides)
    return AutocallContract(**params)


# ------------------------------------------------------------------ plumbing
def test_normals_are_deterministic_and_shape_stable():
    first = _normals(3, 8, 42)
    again = _normals(3, 8, 42)
    other_seed = _normals(3, 8, 43)

    assert first.shape == (8, 3)
    assert (first == again).all()
    assert not (first == other_seed).all()


class _Grid:
    """Stand-in for :class:`TimeGrid` (the normals cache only reads these two)."""

    def __init__(self, dates):
        self.dates = tuple(dates)

    @property
    def n_steps(self):
        return len(self.dates) - 1


def test_normals_cache_realigns_the_intervals_instead_of_redrawing():
    """A shifted grid is a suffix: reuse the (offset) columns, do not redraw.

    Drawing fresh numbers made the theta difference of two valuations that
    carry independent standard errors (~700 on a 1M notional at 65k paths) - it
    measured the noise instead of the day roll.
    """
    from surface_pricer.pricing.exotics.autocall.mc import _suffix_offset

    engine = AutocallMonteCarlo()
    cache: dict = {}
    monday = datetime(2026, 1, 5, 15, 0)
    full = _Grid([monday, datetime(2026, 1, 6), datetime(2026, 1, 7)])
    shifted = _Grid([datetime(2026, 1, 6, 15, 0), datetime(2026, 1, 7)])

    first = engine._normals_for(full, 64, 0, cache)
    second = engine._normals_for(shifted, 64, 0, cache)

    assert first.shape == (64, 2)
    assert second.shape == (64, 1)
    assert (second == first[:, 1:]).all()  # by interval, not from column zero
    assert cache["normals"][1] is first  # the draw was reused, not replaced
    assert _suffix_offset(full.dates, full.dates) == 0
    assert _suffix_offset(full.dates, shifted.dates) == 1
    # a grid that is not a suffix of the cached one is a different grid
    assert _suffix_offset(
        full.dates, (datetime(2026, 1, 5, 15, 0), datetime(2026, 1, 8))
    ) is None
    assert _suffix_offset(shifted.dates, full.dates) is None


def test_local_vol_cache_builds_once_per_surface_and_grid():
    """Spot / rate bumps reuse the coefficient table; vol bumps and theta rebuild."""
    from surface_pricer.pricing.models.localvol import LocalVolCache

    market = _market()
    dates = (market.valuation_date, EXPIRY)
    tables = LocalVolCache()

    first = tables.table(market, dates)
    assert tables.table(market.clone(spot=market.spot * 1.01), dates) is first
    assert tables.builds == 1

    bumped_surface = _market(atm_vols=(0.30,)).surface
    assert tables.table(market.clone(surface=bumped_surface), dates) is not first
    assert tables.builds == 2

    tables.table(market, (market.valuation_date, datetime(2026, 9, 5)))
    assert tables.builds == 3  # the theta grid is a different grid


def test_time_grid_puts_every_observation_on_a_node():
    from surface_pricer.pricing.exotics.autocall import build_schedule

    market = _market()
    contract = _contract(ko_levels=(1.0,))
    schedule = build_schedule(contract, market)

    grid = build_time_grid(schedule, market, steps_per_observation=3)

    assert grid.dates[0] == VALUATION
    assert grid.dates[-1] == EXPIRY
    for index, step in enumerate(grid.observation_steps):
        assert grid.dates[step] == schedule.observation_dates[index]
    assert grid.n_steps >= 3 * len(schedule.observation_dates)


def test_time_grid_is_laid_out_by_vol_time_and_survives_a_one_day_bump():
    from surface_pricer.pricing.exotics.autocall import build_schedule

    market = _market()
    # the sub-step placement only applies to the simplified knock-in convention:
    # with a daily knock-in the monitoring dates are the grid
    contract = _contract(ko_levels=(1.0,), ki_frequency="observation_dates")
    schedule = build_schedule(contract, market)
    rolled = market.clone(valuation_date=VALUATION + timedelta(days=1))
    rolled_schedule = build_schedule(contract, rolled)

    grid = build_time_grid(schedule, market, steps_per_observation=3)
    shifted = build_time_grid(rolled_schedule, rolled, steps_per_observation=3)

    # dates strictly increase and every step advances vol time
    assert list(grid.dates) == sorted(set(grid.dates))
    assert all(left < right for left, right in zip(grid.vol_times, grid.vol_times[1:]))

    # sub-steps are equal in **vol time**, not in calendar days: a holiday week
    # must not collapse into one oversized step.  Snapping to whole dates keeps
    # them within about a business day of each other.
    first_segment = grid.dates[: grid.observation_steps[0] + 1]
    times = [market.year_fraction(day) for day in first_segment]
    sizes = [right - left for left, right in zip(times, times[1:])]
    assert max(sizes) <= min(sizes) * 1.35

    # only the first segment may depend on the valuation date - otherwise a
    # one-day theta bump re-lays the whole grid and measures its own
    # discretisation instead of the market
    later = [day for day in grid.dates if day > schedule.observation_dates[0]]
    later_shifted = [
        day for day in shifted.dates if day > rolled_schedule.observation_dates[0]
    ]
    assert later == later_shifted


def test_time_grid_minimum_step_count_keeps_the_pde_resolved():
    from surface_pricer.pricing.exotics.autocall import build_schedule

    market = _market()
    schedule = build_schedule(_contract(ki_frequency="observation_dates"), market)

    # a quarterly segment is 0.25 vol years, so a 0.2 target step alone would
    # collapse it into a single step; the per-segment floor keeps the PDE
    # comparable with the MC engine
    coarse = build_time_grid(schedule, market, steps_per_observation=1, target_step=0.2)
    floored = build_time_grid(schedule, market, steps_per_observation=6, target_step=0.2)

    assert coarse.n_steps < 6 * len(schedule.observation_dates)
    assert floored.n_steps >= 6 * len(schedule.observation_dates)


# ------------------------------------------------------- zero-vol analytics
def test_zero_vol_no_events_pays_principal_and_coupon():
    market = _market(atm_vols=(0.0,))
    contract = _contract()  # KO 200, KI 10 -> nothing ever triggers
    engine = AutocallMonteCarlo(paths=1024, seed=7)

    result = engine.price(contract, market)

    expected = 1.0e6 * (1 + 0.20 * 273 / 365.0) * market.discount_factor(EXPIRY)
    assert result.npv == pytest.approx(expected, rel=1e-10)
    assert result.metadata["std_error"] == pytest.approx(0.0, abs=1e-9)


def test_zero_vol_knock_out_settles_the_first_observation():
    market = _market(atm_vols=(0.0,))
    contract = _contract(ko_levels=(0.5,))  # KO 50 -> knocked out immediately
    engine = AutocallMonteCarlo(paths=1024, seed=7)

    result = engine.price(contract, market)

    cash = 1.0e6 * (1 + 0.20 * 90 / 365.0)
    expected = cash * market.discount_factor(datetime(2026, 4, 5))
    assert result.npv == pytest.approx(expected, rel=1e-10)


def test_zero_vol_knock_in_pays_the_short_put():
    # a heavy borrow pushes the forward below the start spot -> the short put
    # settles at F(T) / spot0
    market = _market(atm_vols=(0.0,), borrow=0.10)
    contract = _contract(ko_levels=(2.0,), ki_level=2.0)  # always knocked in
    engine = AutocallMonteCarlo(paths=1024, seed=7)

    result = engine.price(contract, market)

    forward = market.forward(EXPIRY)
    performance = min(forward / SPOT, 1.0)
    expected = 1.0e6 * performance * market.discount_factor(EXPIRY)
    assert result.npv == pytest.approx(expected, rel=1e-6)


def test_zero_vol_protected_principal_floors_the_settlement():
    market = _market(atm_vols=(0.0,), borrow=0.10)
    contract = _contract(ko_levels=(2.0,), ki_level=2.0, protected_principal=1.0)
    engine = AutocallMonteCarlo(paths=512, seed=1)

    result = engine.price(contract, market)

    expected = 1.0e6 * market.discount_factor(EXPIRY)
    assert result.npv == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------- common random nums
def test_pricing_is_bit_for_bit_reproducible():
    market = _market(atm_vols=(0.25,))
    contract = _contract(ko_levels=(1.0,), ki_level=0.75)
    engine = AutocallMonteCarlo(paths=4096, seed=11)

    first = engine.price(contract, market)
    again = engine.price(contract, market)

    assert first.npv == again.npv
    assert first.metadata["std_error"] == again.metadata["std_error"]
    assert first.metadata["paths"] == 4096


def test_paths_round_up_to_a_power_of_two():
    market = _market(atm_vols=(0.25,))
    engine = AutocallMonteCarlo(paths=1000, seed=0)

    result = engine.price(_contract(ko_levels=(1.0,), ki_level=0.75), market)

    assert result.metadata["paths"] == 1024


def test_standard_error_is_reported_for_a_stochastic_market():
    market = _market(atm_vols=(0.25,))
    engine = AutocallMonteCarlo(paths=4096, seed=3)

    result = engine.price(_contract(ko_levels=(1.0,), ki_level=0.75), market)

    assert result.metadata["std_error"] > 0.0
    assert result.npv > 0.0
    assert result.metadata["method"] == "monte_carlo"


def test_the_second_order_greeks_are_quoted_by_the_mc_engine():
    """The same harness as PDE: volga / vanna ride the CRN-paired valuations."""
    from surface_pricer.pricing.results import RiskSettings

    market = _market(atm_vols=(0.25,))
    contract = _contract(ko_levels=(1.05,), ki_level=0.75)
    engine = AutocallMonteCarlo(paths=2048, seed=5)

    result = engine.greeks(
        contract, market, RiskSettings(greeks=("vega", "volga", "vanna"))
    )

    assert result.vega is not None and result.vega != 0.0
    assert result.volga is not None and math.isfinite(result.volga)
    assert result.vanna is not None and math.isfinite(result.vanna)
    assert result.volga != 0.0 and result.vanna != 0.0
    assert result.metadata["greek_convention"]["vanna"].startswith("spot x vol")


def test_greeks_are_finite_and_deterministic():
    market = _market(atm_vols=(0.25,))
    contract = _contract(ko_levels=(1.0,), ki_level=0.75)
    engine = AutocallMonteCarlo(paths=2048, seed=5)

    first = engine.greeks(contract, market)
    # a second run must reuse the same normals bit for bit
    again = engine.greeks(contract, market)

    assert first.delta == again.delta
    assert first.vega == again.vega
    for name in ("delta", "delta_cash", "gamma", "vega", "rho", "rhoq", "theta"):
        value = getattr(first, name)
        assert value is not None and math.isfinite(value)
    assert first.metadata["greek_convention"]["delta"].startswith("bump-and-revalue")


def test_settled_contract_returns_the_discounted_knock_out_cash():
    market = _market(atm_vols=(0.25,), valuation=datetime(2026, 5, 1, 15, 0))
    # knocked out on a past observation: that is ledger state, replayed by the
    # caller before the engine sees the contract (sparse fixings, so the knock-in
    # is monitored together with the knock-out here)
    contract = apply_history(
        _contract(
            ko_levels=(1.0,),
            ki_level=0.75,
            ki_frequency="observation_dates",
            history=((date(2026, 4, 5), 101.0),),
        ),
        market,
    )
    engine = AutocallMonteCarlo(paths=256, seed=0)

    result = engine.price(contract, market)

    cash = 1.0e6 * (1 + 0.20 * 90 / 365.0)
    expected = cash * market.discount_factor(datetime(2026, 4, 5))
    assert result.npv == pytest.approx(expected, rel=1e-12)
    assert result.metadata["knocked_out"] is True
