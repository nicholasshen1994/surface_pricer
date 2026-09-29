"""Black-forward pricing primitives."""

import math
from typing import Union

import numpy as np
from scipy.stats import norm


def _as_array(value):
    return np.asarray(value, dtype=float)


def black_price(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    year_fraction: float,
    volatility: Union[float, np.ndarray],
    discount_factor: float = 1.0,
    option_type: str = "call",
):
    fwd = _as_array(forward)
    strike_arr = _as_array(strike)
    vol = np.maximum(_as_array(volatility), 1.0e-12)
    tau = max(float(year_fraction), 0.0)
    sign = 1.0 if option_type.lower() in {"c", "call"} else -1.0

    if tau <= 0.0:
        value = np.maximum(sign * (fwd - strike_arr), 0.0)
        return discount_factor * value

    sigma_sqrt_t = vol * math.sqrt(tau)
    d1 = (np.log(fwd / strike_arr) + 0.5 * sigma_sqrt_t ** 2) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    if sign > 0:
        value = fwd * norm.cdf(d1) - strike_arr * norm.cdf(d2)
    else:
        value = strike_arr * norm.cdf(-d2) - fwd * norm.cdf(-d1)
    result = discount_factor * value
    return float(result) if result.ndim == 0 else result


def black_vega(
    forward: float,
    strike: float,
    year_fraction: float,
    volatility: float,
    discount_factor: float = 1.0,
) -> float:
    tau = max(float(year_fraction), 0.0)
    if tau <= 0.0:
        return 0.0
    sigma_sqrt_t = max(float(volatility), 1.0e-12) * math.sqrt(tau)
    d1 = (math.log(float(forward) / float(strike)) + 0.5 * sigma_sqrt_t ** 2) / sigma_sqrt_t
    return float(discount_factor * forward * norm.pdf(d1) * math.sqrt(tau))


def undiscounted_intrinsic(forward: float, strike: float, option_type: str) -> float:
    sign = 1.0 if option_type.lower() in {"c", "call"} else -1.0
    return float(max(sign * (forward - strike), 0.0))
