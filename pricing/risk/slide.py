"""Spot slide: one NPV / Greek set per bumped spot - the desk's spot ladder.

A slide answers "what would this book look like if the underlying traded there".
Every rung is a **full repricing on the bumped market state**
(``market.clone(spot=level)`` through the ordinary pricer / engine), never a
first-order extrapolation from the base quote: a rung inside the knock-in region
shows the real convexity, and the Greeks are the ones that state would quote -
``delta`` is differenced *around the rung*, not at the base.

The trade itself stays put.  A vanilla strike is already absolute
(:class:`~surface_pricer.pricing.vanilla.spec.VanillaSpec`, resolved once) and a
snowball's barriers are anchored on ``spot0``
(:meth:`~surface_pricer.pricing.exotics.autocall.schedule.AutocallSchedule.rebased`
moves the market side only), so sliding the spot moves the market and never the
contract.

The module is deliberately product-agnostic: it owns the ladder and the row
container while the caller supplies ``price_at(market) -> PricingResult``.  That
keeps :mod:`surface_pricer.pricing.risk` free of imports from the product
packages - they import *it*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..results import PricingResult

#: Ladder defaults: ``+-30%`` around the base spot in ``5%`` steps (13 rungs).
DEFAULT_SPAN = 0.30
DEFAULT_STEP = 0.05


@dataclass(frozen=True)
class SlideRow:
    """One rung of a slide: the market at ``spot``, repriced there."""

    spot: float
    #: Move against the base spot (``spot / base - 1``), so the base row reads 0.
    bump: float
    npv: float
    #: Only the Greeks that were asked for - an unrequested one is absent, not None.
    greeks: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"spot": self.spot, "bump": self.bump, "npv": self.npv}
        row.update(self.greeks)
        return row


def spot_ladder(
    base_spot: float,
    *,
    span: Optional[float] = None,
    step: Optional[float] = None,
    spots: Optional[Iterable[float]] = None,
) -> Tuple[float, ...]:
    """The spot levels of a slide, ascending, always including the base spot.

    ``spots`` are absolute levels and win when given.  Otherwise the ladder is
    ``base_spot x (1 + bump)`` for ``bump`` running from ``-span`` to ``+span`` in
    ``step`` increments (both are fractions: ``0.05`` = 5%), and the base row is
    added when the grid does not land on it - a slide always answers for "no move"
    as well.
    """
    base = float(base_spot)
    if base <= 0.0:
        raise ValueError("the base spot must be positive")

    if spots is not None:
        levels = sorted({float(level) for level in spots})
        if not levels:
            raise ValueError("no spot levels given")
        if levels[0] <= 0.0:
            raise ValueError("spot levels must be positive (got {})".format(levels[0]))
        return tuple(levels)

    span = DEFAULT_SPAN if span is None else abs(float(span))
    step = DEFAULT_STEP if step is None else abs(float(step))
    if step <= 0.0:
        raise ValueError("the slide step must be positive")

    steps = int(round(span / step))
    bumps = sorted({round(index * step, 12) for index in range(-steps, steps + 1)})
    # a move of -100% or more has no spot left to price; the ladder stops above it
    bumps = [bump for bump in bumps if bump > -1.0]
    if not any(bump == 0.0 for bump in bumps):
        bumps.append(0.0)
    levels = tuple(base * (1.0 + bump) for bump in sorted(bumps))
    if levels[0] <= 0.0:  # pragma: no cover - defensive, the filter above holds
        raise ValueError("a {:.2%} move makes the spot non-positive".format(bumps[0]))
    return levels


def run_slide(
    market: Any,
    *,
    price_at: Callable[[Any], PricingResult],
    greeks: Sequence[str] = (),
    ladder: Optional[Iterable[float]] = None,
    span: Optional[float] = None,
    step: Optional[float] = None,
    spots: Optional[Iterable[float]] = None,
    on_rung: Optional[Callable[[int, float], None]] = None,
) -> List[SlideRow]:
    """Price every rung of the ladder on the bumped market.

    ``price_at`` receives the **bumped market state** - spot moved, everything
    else (curves, surface anchor, valuation date) untouched - and must return a
    :class:`PricingResult` carrying the requested Greeks.  The caller rebases the
    trade inside ``price_at``, because only it knows the product.
    """
    levels = (
        tuple(ladder)
        if ladder is not None
        else spot_ladder(market.spot, span=span, step=step, spots=spots)
    )
    base = float(market.spot)
    wanted = tuple(greeks)
    rows: List[SlideRow] = []
    for index, level in enumerate(levels):
        if on_rung is not None:
            on_rung(index, float(level))
        result = price_at(market.clone(spot=float(level)))
        values: Dict[str, float] = {}
        for name in wanted:
            value = getattr(result, name, None)
            if value is not None:
                values[name] = float(value)
        rows.append(
            SlideRow(
                spot=float(level),
                bump=float(level) / base - 1.0,
                npv=float(result.npv),
                greeks=values,
            )
        )
    return rows


__all__ = ["DEFAULT_SPAN", "DEFAULT_STEP", "SlideRow", "run_slide", "spot_ladder"]
