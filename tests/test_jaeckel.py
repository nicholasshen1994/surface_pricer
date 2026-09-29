import numpy as np

from surface_pricer.core.math.black import black_price
from surface_pricer.core.math.implied_vol import implied_vol
from surface_pricer.core.math.jaeckel import implied_vol_jaeckel


def test_jaeckel_round_trip_across_moneyness():
    forward, tau, volatility = 7500.0, 0.25, 0.22
    for moneyness in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5):
        strike = forward * moneyness
        for option_type in ("call", "put"):
            price = black_price(forward, strike, tau, volatility, 1.0, option_type)
            implied = implied_vol_jaeckel(price, forward, strike, tau, option_type)
            assert implied is not None
            assert abs(implied - volatility) < 1.0e-8


def test_jaeckel_agrees_with_brent_fallback():
    forward, tau = 7500.0, 0.5
    strikes = np.array([5000.0, 6500.0, 7400.0, 7600.0, 8500.0, 10000.0])
    for strike in strikes:
        option_type = "put" if strike < forward else "call"
        price = black_price(forward, strike, tau, 0.23, 1.0, option_type)
        jaeckel = implied_vol_jaeckel(price, forward, strike, tau, option_type)
        brent = implied_vol(price, forward, strike, tau, 1.0, option_type)
        assert abs(jaeckel - brent) < 1.0e-8


def test_jaeckel_rejects_invalid_prices():
    forward, strike, tau = 7500.0, 8000.0, 0.1
    assert implied_vol_jaeckel(0.0, forward, strike, tau, "call") is None
    assert implied_vol_jaeckel(1.0e9, forward, strike, tau, "call") is None
    assert implied_vol_jaeckel(10.0, forward, strike, 0.0, "call") is None
    assert implied_vol_jaeckel(float("nan"), forward, strike, tau, "call") is None


def test_jaeckel_deep_otm_survives_tiny_time_value():
    forward, tau = 7500.0, 8.0 / 243.0
    strike = forward * 1.10
    price = black_price(forward, strike, tau, 0.35, 1.0, "call")
    implied = implied_vol_jaeckel(price, forward, strike, tau, "call")
    assert implied is not None
    assert abs(implied - 0.35) < 1.0e-6
