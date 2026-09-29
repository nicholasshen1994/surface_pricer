"""Black implied-volatility inversion and quote conversion."""

from typing import Tuple

from scipy.optimize import brentq

from .black import black_price, undiscounted_intrinsic


def put_to_call(put_price: float, forward: float, strike: float, discount_factor: float) -> float:
    return float(put_price + discount_factor * (forward - strike))


def call_to_put(call_price: float, forward: float, strike: float, discount_factor: float) -> float:
    return float(call_price + discount_factor * (strike - forward))


def implied_vol(
    price: float,
    forward: float,
    strike: float,
    year_fraction: float,
    discount_factor: float = 1.0,
    option_type: str = "call",
    max_volatility: float = 10.0,
) -> float:
    if year_fraction <= 0.0:
        return 0.0
    if price != price:
        raise ValueError("option price is NaN")

    intrinsic = discount_factor * undiscounted_intrinsic(forward, strike, option_type)
    upper = discount_factor * (
        forward if option_type.lower() in {"c", "call"} else strike
    )
    # Do not use an absolute price tolerance here.  Deep OTM options can
    # legitimately have prices far below 1e-12 while still carrying a
    # meaningful implied volatility.
    if price <= intrinsic:
        return 1.0e-12
    if price >= upper:
        return max_volatility

    def objective(vol):
        return black_price(
            forward,
            strike,
            year_fraction,
            vol,
            discount_factor,
            option_type,
        ) - price

    return float(brentq(objective, 1.0e-12, max_volatility, xtol=1.0e-12, rtol=1.0e-12))


def implied_vol_bid_ask(
    bid_price: float,
    ask_price: float,
    forward: float,
    strike: float,
    year_fraction: float,
    discount_factor: float,
    option_type: str,
) -> Tuple[float, float, float]:
    bid = implied_vol(
        bid_price,
        forward,
        strike,
        year_fraction,
        discount_factor,
        option_type,
    )
    ask = implied_vol(
        ask_price,
        forward,
        strike,
        year_fraction,
        discount_factor,
        option_type,
    )
    return bid, ask, 0.5 * (bid + ask)
