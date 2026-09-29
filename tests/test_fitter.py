from datetime import datetime

import numpy as np

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.math.black import black_price
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.eds_slice import EDSSabrSlice
from surface_pricer.fitting.pipeline import build_market_state, fit_surface
from surface_pricer.fitting.settings import FitSettings
from surface_pricer.marketdata.data import OptionQuoteRecord, RawSnapshot

VALUATION = datetime(2026, 9, 28, 14, 55)
EXPIRY = datetime(2026, 12, 18, 15, 0)
FORWARD = 7500.0
STRIKES = (7000.0, 7300.0, 7500.0, 7700.0, 8000.0)


def _snapshot(volatility=0.22, sabr_params=None):
    calendar = BusinessCalendar(name="TEST")
    rate_curve = ConstantRateCurve(0.0, anchor=VALUATION)
    market = MarketState(
        valuation_date=VALUATION,
        spot=FORWARD,
        rate_curve=rate_curve,
        calendar=calendar,
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )
    tau = market.year_fraction(EXPIRY)

    if sabr_params is None:
        smiles = {strike: volatility for strike in STRIKES}
    else:
        slice_ = EDSSabrSlice(
            ref_strike=FORWARD,
            forward=FORWARD,
            vol_atmf=volatility,
            tau=tau,
            **sabr_params
        )
        smiles = {
            strike: float(slice_.get_implied_vol(np.asarray([strike]))[0])
            for strike in STRIKES
        }

    records = []
    for strike in STRIKES:
        for option_type in ("call", "put"):
            price = black_price(FORWARD, strike, tau, smiles[strike], 1.0, option_type)
            records.append(
                OptionQuoteRecord(
                    underlying="TEST",
                    expiry=EXPIRY,
                    strike=float(strike),
                    option_type=option_type,
                    bid=float(price * 0.998),
                    ask=float(price * 1.002),
                )
            )
    return RawSnapshot(
        underlying="TEST",
        valuation_datetime=VALUATION,
        spot=FORWARD,
        option_records=records,
        rate_curve=rate_curve,
        calendar=calendar,
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


def test_fit_flat_smile_recovers_atm_vol():
    snapshot = _snapshot(volatility=0.22)
    settings = FitSettings(max_iterations=40)
    result = fit_surface(snapshot, settings)

    assert len(result.slices) == 1
    slice_result = result.slices[0]
    assert slice_result.rmse < 5.0e-3
    assert abs(slice_result.atm_vol - 0.22) < 1.0e-3
    assert abs(float(result.surface.atm_vols[0]) - 0.22) < 1.0e-3
    assert abs(float(result.surface.expiry_times[0]) - slice_result.tau) < 1.0e-12


def test_fit_recovers_skewed_smile():
    params = dict(
        skew=0.6,
        conv=0.4,
        left_skew_1=0.5,
        left_skew_2=0.0,
        right_skew_1=0.3,
        right_skew_2=0.0,
    )
    snapshot = _snapshot(volatility=0.22, sabr_params=params)
    settings = FitSettings(max_iterations=60)
    result = fit_surface(snapshot, settings)

    assert len(result.slices) == 1
    slice_result = result.slices[0]
    assert slice_result.rmse < 2.0e-3
    # the fitted surface keeps a positive skew and convexity
    assert result.surface.skews[0] > 0.0
    assert result.surface.convs[0] > 0.0
