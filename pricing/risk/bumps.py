"""Market-state bumps shared by every product's risk run.

Only the bump *objects* live here, not the difference stencils: the vanilla
pricer and the exotic engines must difference exactly the same states (relative
spot shift, parallel vol shift, parallel rate / borrow shift) so that a
cross-check compares products rather than two risk setups.  Stencils and
reporting units stay with the caller (:mod:`surface_pricer.pricing.vanilla.greeks`
and :mod:`surface_pricer.pricing.risk.diff`).
"""

from __future__ import annotations

import numpy as np

from ...core.market import MarketState


def require_surface(market: MarketState):
    """The fitted surface, or a clear error for vol-related Greeks."""
    if market.surface is None:
        raise ValueError("MarketState.surface is required for vol-related Greeks")
    return market.surface


def spot_bump(market: MarketState, amount: float) -> MarketState:
    """Shift the spot by ``amount`` absolute points."""
    return market.clone(spot=market.spot + amount)


def vol_bump(market: MarketState, amount: float) -> MarketState:
    """Shift every vol pillar in parallel by ``amount`` vol points."""
    return market.clone(surface=require_surface(market).bump_parallel(amount))


def curve_bump(market: MarketState, curve_name: str, pillar, amount: float) -> MarketState:
    """Shift one pillar of the ``rate`` / ``borrow`` curve by ``amount``."""
    curve = market.rate_curve if curve_name == "rate" else market.borrow_curve
    if curve is None:
        return market
    bumped = curve.bump_pillar(pillar, amount)
    return market.clone(**{curve_name + "_curve": bumped})


def parallel_bump(market: MarketState, curve_name: str, amount: float) -> MarketState:
    """Shift a whole curve in parallel (falling back to its single pillar)."""
    curve = market.rate_curve if curve_name == "rate" else market.borrow_curve
    if curve is None:
        return market
    rates = getattr(curve, "rates", None)
    if rates is not None and hasattr(curve, "with_rates"):
        bumped = curve.with_rates(np.asarray(rates, dtype=float) + amount)
        return market.clone(**{curve_name + "_curve": bumped})
    return market.clone(**{curve_name + "_curve": curve.bump_pillar(0, amount)})


__all__ = [
    "curve_bump",
    "parallel_bump",
    "require_surface",
    "spot_bump",
    "vol_bump",
]
