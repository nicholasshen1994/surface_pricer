"""Jaeckel "Let's Be Rational" implied volatility inversion.

A thin wrapper around the MIT licensed ``py_lets_be_rational`` package so the
fit pipeline has one import point and consistent edge-case handling.
"""

from __future__ import annotations

import math
from typing import Optional

from py_lets_be_rational import (
    implied_volatility_from_a_transformed_rational_guess as _implied_volatility,
)


def implied_vol_jaeckel(
    undiscounted_price: float,
    forward: float,
    strike: float,
    tau: float,
    option_type: str,
) -> Optional[float]:
    """Invert a Black-76 price into implied volatility.

    ``undiscounted_price`` is the option price without the discount factor.
    Returns ``None`` when the price is outside the no-arbitrage range, which
    lets the caller drop the quote instead of failing the whole slice.
    """
    try:
        price = float(undiscounted_price)
        fwd = float(forward)
        k = float(strike)
        expiry = float(tau)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(price) and math.isfinite(fwd) and math.isfinite(k)):
        return None
    if expiry <= 0.0 or price <= 0.0 or fwd <= 0.0 or k <= 0.0:
        return None

    sign = 1.0 if str(option_type).lower() in {"c", "call"} else -1.0
    try:
        volatility = _implied_volatility(price, fwd, k, expiry, sign)
    except Exception:
        return None
    volatility = float(volatility)
    if not math.isfinite(volatility) or volatility <= 0.0:
        return None
    return volatility


__all__ = ["implied_vol_jaeckel"]
