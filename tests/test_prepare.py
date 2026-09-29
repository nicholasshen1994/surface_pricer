from datetime import datetime

import numpy as np

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.math.black import black_price
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.pipeline import build_market_state
from surface_pricer.fitting.prepare import prepare_slices
from surface_pricer.fitting.settings import FitSettings
from surface_pricer.marketdata.data import OptionQuoteRecord, RawSnapshot

VALUATION = datetime(2026, 9, 28, 14, 55)
EXPIRY = datetime(2026, 12, 18, 15, 0)
FORWARD = 7500.0
VOL = 0.22
STRIKES = (6500.0, 7000.0, 7300.0, 7500.0, 7700.0, 8000.0, 8500.0)


def _market(spot=FORWARD):
    calendar = BusinessCalendar(name="TEST")
    rate_curve = ConstantRateCurve(0.0, anchor=VALUATION)
    return MarketState(
        valuation_date=VALUATION,
        spot=spot,
        rate_curve=rate_curve,
        calendar=calendar,
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


def _snapshot(forward=FORWARD, volatility=VOL, strikes=STRIKES, future_price=None, bad_spread_strike=None):
    market = _market()
    tau = market.year_fraction(EXPIRY)
    records = []
    for strike in strikes:
        for option_type in ("call", "put"):
            price = black_price(forward, strike, tau, volatility, 1.0, option_type)
            spread = 0.002
            if bad_spread_strike is not None and abs(strike - bad_spread_strike) < 1.0e-9:
                spread = 0.5
            records.append(
                OptionQuoteRecord(
                    underlying="TEST",
                    expiry=EXPIRY,
                    strike=float(strike),
                    option_type=option_type,
                    bid=float(price * (1.0 - spread)),
                    ask=float(price * (1.0 + spread)),
                )
            )
    future_prices = {}
    if future_price is not None:
        future_prices[EXPIRY.date().isoformat()] = float(future_price)
    return RawSnapshot(
        underlying="TEST",
        valuation_datetime=VALUATION,
        spot=forward,
        option_records=records,
        rate_curve=market.rate_curve,
        calendar=market.calendar,
        trading_days_per_year=243.0,
        holiday_weight=0.05,
        future_price_by_expiry=future_prices,
    )


def test_prepare_uses_parity_forward_and_otm_quotes():
    snapshot = _snapshot()
    settings = FitSettings()
    market = build_market_state(snapshot, settings)
    slices, overrides = prepare_slices(snapshot, settings, market)

    assert len(slices) == 1
    slice_info = slices[0]
    assert abs(slice_info.forward - FORWARD) < 1.0
    assert overrides[EXPIRY.date().isoformat()] == slice_info.forward
    assert np.all(np.diff(slice_info.strikes) > 0.0)
    assert np.all(slice_info.bid_vols <= slice_info.vols)
    assert np.all(slice_info.vols <= slice_info.ask_vols)
    assert abs(float(slice_info.weights.sum()) - 1.0) < 1.0e-12
    # below the forward the puts are used, at/above it the calls
    low = slice_info.strikes < slice_info.forward
    assert all(kind == "put" for kind, flag in zip(slice_info.option_types, low) if flag)
    assert all(kind == "call" for kind, flag in zip(slice_info.option_types, low) if not flag)


def test_future_forward_source_keeps_future_price():
    snapshot = _snapshot(future_price=7300.0)
    settings = FitSettings(forward_source="future")
    market = build_market_state(snapshot, settings)
    slices, _ = prepare_slices(snapshot, settings, market)

    assert len(slices) == 1
    assert slices[0].forward == 7300.0


def test_mad_filter_drops_wide_spread_strike():
    clean = _snapshot()
    dirty = _snapshot(bad_spread_strike=7700.0)
    settings = FitSettings()
    market = build_market_state(clean, settings)

    clean_slices, _ = prepare_slices(clean, settings, market)
    dirty_slices, _ = prepare_slices(dirty, settings, market)

    assert len(clean_slices) == 1
    assert len(dirty_slices) == 1
    assert 7700.0 not in dirty_slices[0].strikes
    assert len(dirty_slices[0].strikes) < len(clean_slices[0].strikes)
