"""PDE engine and finite-difference helpers: grid pinning, solver sanity and
the deterministic payoff limits."""

import math
from datetime import date, datetime

import numpy as np
import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.exotics.autocall import AutocallContract
from surface_pricer.pricing.exotics.autocall import apply_history
from surface_pricer.pricing.exotics.autocall.pde import AutocallPDE
from surface_pricer.pricing.numerics.fdm import (
    bsm_coefficients,
    build_log_grid,
    theta_step,
    thomas_solve,
)

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


# --------------------------------------------------------------- fdm helpers
def test_thomas_solve_inverts_a_known_tridiagonal_system():
    lower = np.array([0.0, 2.0, 3.0, 2.0])
    diag = np.array([4.0, 5.0, 6.0, 7.0])
    upper = np.array([-1.0, -1.0, -1.0, 0.0])
    rhs = np.array([1.0, 2.0, 3.0, 4.0])

    solution = thomas_solve(lower, diag, upper, rhs)

    residual = diag * solution
    residual[1:] += lower[1:] * solution[:-1]
    residual[:-1] += upper[:-1] * solution[1:]
    assert residual == pytest.approx(rhs, rel=1e-12)


def test_build_log_grid_pins_crucial_levels_exactly():
    grid = build_log_grid(50.0, 200.0, 101, crucial_levels=[75.0, 130.0])

    assert np.all(np.diff(grid) > 0.0)
    for level in (75.0, 130.0):
        assert np.min(np.abs(grid - math.log(level))) < 1e-12

    plain = build_log_grid(50.0, 200.0, 101)
    assert np.allclose(np.diff(plain), np.diff(plain)[0])  # uniform without levels


def test_theta_step_keeps_interior_values_for_a_zero_operator():
    zeros = np.zeros(5)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    stepped = theta_step(
        zeros, zeros, zeros, values,
        dt=0.1, theta=0.5, low_boundary=0.0, high_boundary=10.0,
    )

    assert stepped[1:-1] == pytest.approx(values[1:-1], rel=1e-12)
    assert stepped[0] == pytest.approx(0.0)
    assert stepped[-1] == pytest.approx(10.0)


def test_theta_step_matches_the_implicit_discounting():
    zeros = np.zeros(5)
    diag = np.full(5, -0.05)  # A = -r with r = 5%
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    stepped = theta_step(
        zeros, diag, zeros, values,
        dt=1.0, theta=1.0, low_boundary=0.0, high_boundary=0.0,
    )

    # (1 - dt * A) V_new = V  ->  (1 + 0.05) V_new = V
    assert stepped[1:-1] == pytest.approx(values[1:-1] / 1.05, rel=1e-12)


def test_bsm_coefficients_satisfy_the_constant_sum_rule():
    x = build_log_grid(50.0, 200.0, 51)
    lower, diag, upper = bsm_coefficients(x, 0.2, 0.02, 0.01)

    # a constant payoff must be reproduced: A . 1 = -r
    applied = diag.copy()
    applied[1:] += lower[1:]
    applied[:-1] += upper[:-1]
    assert applied[1:-1] == pytest.approx(-0.02, rel=1e-12)


# -------------------------------------------------------------- the engine
def test_no_event_pays_principal_and_coupon():
    market = _market(atm_vols=(0.05,))
    contract = _contract()  # KO 200, KI 10 - nothing triggers
    engine = AutocallPDE()

    result = engine.price(contract, market)

    expected = 1.0e6 * (1 + 0.20 * 273 / 365.0) * market.discount_factor(EXPIRY)
    assert result.npv == pytest.approx(expected, rel=1e-3)
    assert result.metadata["method"] == "pde"
    assert result.metadata["nodes"] >= 601
    assert result.metadata["theta"] == pytest.approx(0.5)


def test_knock_out_settles_the_first_observation():
    market = _market(atm_vols=(0.05,))
    contract = _contract(ko_levels=(0.5,))
    engine = AutocallPDE()

    result = engine.price(contract, market)

    expected = 1.0e6 * (1 + 0.20 * 90 / 365.0) * market.discount_factor(
        datetime(2026, 4, 5)
    )
    assert result.npv == pytest.approx(expected, rel=1e-3)


def test_knock_in_with_full_protection_is_deterministic():
    # the always-knocked-in payoff is flat under full protection, so the PDE
    # must reproduce the discounted notional whatever the volatility
    market = _market(atm_vols=(0.20,), borrow=0.10)
    contract = _contract(ko_levels=(2.0,), ki_level=2.0, protected_principal=1.0)

    result = AutocallPDE().price(contract, market)

    expected = 1.0e6 * market.discount_factor(EXPIRY)
    assert result.npv == pytest.approx(expected, rel=1e-3)


def test_the_second_order_greeks_are_wired_and_cost_only_new_states(monkeypatch):
    """volga / vanna on the exotic side, at the documented cost.

    The stencil arithmetic and the reporting units are pinned on the harness itself
    (``test_risk_selection``); what an engine has to prove is the wiring - the fields
    are filled and finite - and the **cost**: volga rides the vega pair's vol states,
    vanna only adds the four crossed spot states.
    """
    from surface_pricer.pricing.exotics.autocall.schedule import build_schedule
    from surface_pricer.pricing.results import RiskSettings

    market = _market(atm_vols=(0.25,))
    contract = _contract(ko_levels=(1.05,), ki_level=0.75)
    engine = AutocallPDE(nodes=201)
    schedule = build_schedule(contract, market)

    valuations = []
    original = AutocallPDE._value

    def counting(self, *args, **kwargs):
        valuations.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AutocallPDE, "_value", counting)

    def run(selection):
        valuations.clear()
        outcome = engine.greeks_schedule(schedule, market, RiskSettings(greeks=selection))
        return len(valuations), outcome

    plain_count, plain = run(("vega",))
    volga_count, with_volga = run(("vega", "volga"))
    vanna_count, with_vanna = run(("vega", "volga", "vanna"))

    assert plain_count == 3  # the base plus the vol pair
    assert volga_count == plain_count  # volga reuses that pair: no extra valuation
    assert vanna_count == volga_count + 4  # vanna adds the four crossed states

    assert with_volga.vega == pytest.approx(plain.vega)
    assert with_volga.volga is not None and math.isfinite(with_volga.volga)
    assert with_volga.volga != 0.0
    assert with_vanna.volga == pytest.approx(with_volga.volga)  # same states, same number
    assert with_vanna.vanna is not None and math.isfinite(with_vanna.vanna)
    assert with_vanna.vanna != 0.0
    conventions = with_vanna.metadata["greek_convention"]
    assert "volga" in conventions and "vanna" in conventions

    # ... and both renderers carry them (the report used to drop what the engine filled)
    from surface_pricer.reporting.autocall_report import autocall_to_dict, format_autocall

    payload = autocall_to_dict(schedule, with_vanna)
    assert payload["greeks"]["volga"] == pytest.approx(with_volga.volga)
    assert payload["greeks"]["vanna"] == pytest.approx(with_vanna.vanna)
    assert payload["greeks"]["delta"] is None  # only what was asked for is filled
    text = format_autocall(schedule, with_vanna, market)
    assert "volga (per (1 vol pt)^2)" in text
    assert "vanna (per 1 vol pt)" in text


def test_greeks_are_finite_with_the_barrier_at_the_spot():
    # a knock-out pinned on the current spot leaves a kink right where the delta
    # is measured: the value stays finite and the Greeks must too, but the local
    # grid slope is not a meaningful reference there (see the test below)
    market = _market(atm_vols=(0.25,))
    contract = _contract(ko_levels=(1.0,), ki_level=0.75)

    result = AutocallPDE().greeks(contract, market)

    for name in ("delta", "delta_cash", "gamma", "vega", "rho", "rhoq", "theta"):
        value = getattr(result, name)
        assert value is not None and math.isfinite(value)
    assert result.delta != 0.0


def test_greeks_match_the_grid_delta_away_from_the_barrier():
    market = _market(atm_vols=(0.25,))
    contract = _contract(ko_levels=(1.05,), ki_level=0.75)

    result = AutocallPDE().greeks(contract, market)

    # the grid derivative is a local slope while the bump delta averages over
    # +/-1% of spot; they agree once the knock-out is off the measurement point
    assert result.delta == pytest.approx(result.metadata["grid_delta"], rel=0.25)


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

    result = AutocallPDE().price(contract, market)

    cash = 1.0e6 * (1 + 0.20 * 90 / 365.0)
    expected = cash * market.discount_factor(datetime(2026, 4, 5))
    assert result.npv == pytest.approx(expected, rel=1e-12)
    assert result.metadata["knocked_out"] is True
