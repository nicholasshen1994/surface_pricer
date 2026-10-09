"""Dupire local volatility: flat-surface limits, analytic term structure,
arbitrage diagnostics, clipping and caching."""

import math
from datetime import date, datetime

import numpy as np
import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.models.localvol import DupireLocalVol

VALUATION = datetime(2026, 1, 5, 15, 0)
SPOT = 100.0
PILLARS = (date(2026, 4, 5), date(2026, 10, 5))
MID = date(2026, 7, 5)


def _market(atm_vols=(0.20, 0.30), pillars=PILLARS, **surface_kwargs):
    surface = EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=list(pillars),
        atm_vols=list(atm_vols),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
        **surface_kwargs,
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


def test_flat_surface_recovers_the_flat_vol():
    local = DupireLocalVol(_market(atm_vols=(0.20, 0.20)))

    values = local.local_vols(MID, np.asarray([80.0, 100.0, 120.0]))
    assert values == pytest.approx([0.20, 0.20, 0.20], rel=1e-6)

    diagnostics = local.slice_at(MID).diagnostics
    assert diagnostics.calendar_arb_ratio == 0.0
    assert diagnostics.butterfly_arb_ratio == 0.0
    assert diagnostics.negative_ratio == 0.0
    assert diagnostics.clipped_ratio == 0.0


def test_term_structure_without_smile_matches_the_analytic_dupire():
    """Without smile the denominator is 1 and sigma_lv^2 = dw/dT, where the
    surface interpolates total variance linearly between the pillars."""
    market = _market(atm_vols=(0.20, 0.30))
    t1, t2 = (float(value) for value in market.surface.expiry_times)
    w1 = 0.20 ** 2 * t1
    w2 = 0.30 ** 2 * t2
    expected = math.sqrt((w2 - w1) / (t2 - t1))

    local = DupireLocalVol(market)
    value = local.local_vol(MID, SPOT)

    assert value == pytest.approx(expected, rel=1e-4)
    assert t1 < local.slice_at(MID).vol_time < t2


def test_smile_keeps_local_vol_finite_and_bounded():
    market = _market(skews=(0.30, -0.20), convs=(0.40, 0.30))
    local = DupireLocalVol(market)

    spots = np.linspace(70.0, 130.0, 25)
    values = local.local_vols(MID, spots)

    assert np.all(np.isfinite(values))
    assert np.all(values >= local.floor - 1e-15)
    assert np.all(values <= local.cap + 1e-15)
    # the smile must actually move the local vol around
    assert values.max() - values.min() > 1e-3


def test_calendar_arbitrage_is_diagnosed_and_floored():
    # 30% -> 10% ATM vols make total variance fall with maturity
    local = DupireLocalVol(_market(atm_vols=(0.30, 0.10)))
    slice_ = local.slice_at(MID)

    assert slice_.diagnostics.calendar_arb_ratio == 1.0
    assert slice_.diagnostics.negative_ratio == 1.0
    assert np.allclose(slice_.local_vols, local.floor)
    assert slice_.diagnostics.clipped_ratio == 1.0


def test_cap_clips_and_is_diagnosed():
    local = DupireLocalVol(_market(atm_vols=(0.20, 0.20)), cap_multiple=0.5)
    slice_ = local.slice_at(MID)

    assert local.cap == pytest.approx(0.10)
    assert slice_.diagnostics.clipped_ratio == 1.0
    assert np.allclose(slice_.local_vols, 0.10)


def test_moneyness_extrapolation_returns_edge_values():
    local = DupireLocalVol(_market())

    low = local.local_vols(MID, np.asarray([1.0e-6]))[0]
    high = local.local_vols(MID, np.asarray([1.0e9]))[0]
    at_left_edge = local.slice_at(MID).local_vols[0]
    at_right_edge = local.slice_at(MID).local_vols[-1]

    assert low == pytest.approx(at_left_edge)
    assert high == pytest.approx(at_right_edge)
    assert np.isfinite(low) and np.isfinite(high)


def test_degenerate_vol_time_returns_the_atm_vol():
    local = DupireLocalVol(_market())

    value = local.local_vol(VALUATION, SPOT)

    assert value == pytest.approx(0.20)  # the first pillar's ATM vol
    slice_ = local.slice_at(VALUATION)
    assert slice_.vol_time <= 0.0
    assert slice_.diagnostics.clipped_ratio == 0.0


def test_slices_are_cached_per_expiry():
    local = DupireLocalVol(_market())

    first = local.slice_at(MID)
    assert local.slice_at(MID) is first
    assert local.slice_at(PILLARS[0]) is not first


def test_diagnostics_summarise_the_built_slices():
    local = DupireLocalVol(_market())
    assert local.diagnostics() == {}
    assert "no slices" in local.describe()

    local.slice_at(MID)
    stats = local.diagnostics()
    assert stats["slices"] == 1.0
    assert stats["floor"] == pytest.approx(local.floor)
    assert stats["cap"] == pytest.approx(local.cap)
    assert "local vol:" in local.describe()


def test_local_vol_requires_a_fitted_surface():
    market = _market().clone(surface=None)
    with pytest.raises(ValueError, match="surface"):
        DupireLocalVol(market)
