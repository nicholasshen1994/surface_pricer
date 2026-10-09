"""Replay of past observations - ledger work, kept out of the pricing engines.

A trade that has been alive for a while already settled some of its terms: it may
have knocked out on a past observation (then the NPV is a single discounted cash
flow) or it may have knocked in already (then the expiry pays the short put).
Which of the two happened is *not* a pricing question - it is read off the
observed fixings - so it is resolved here, once, by the caller::

    contract = apply_history(contract, market)      # outside the engines
    schedule = build_schedule(contract, market)     # effective terms only
    result = AutocallPDE().price_schedule(contract, schedule, market)

The engines therefore consume nothing but the ledger dates
(:attr:`AutocallContract.knocked_in_date` / :attr:`AutocallContract.knocked_out_at`)
and never touch the fixing table.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Optional, Tuple

from ....core.daycount import to_datetime
from ....core.market import MarketState
from .contract import AutocallContract, triggers
from .schedule import resolve_spot0


def replay_history(
    contract: AutocallContract, market: MarketState
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """``(knocked_in_date, knocked_out_at)`` read off the fixings on record.

    The knock-in is recorded as the **first** monitoring date whose fixing sits at
    or below the barrier (``None`` when it never did) - a date, so that a later
    valuation can be moved back before it and will price the trade pre-knock-in
    without any payload edit.

    Past observations use the **raw** (unshifted) barriers, matching edslib's
    ``on_or_before(ds_date).merge(shifted.after(ds_date))`` convention, and a
    fixing is required for every monitoring date: with the default daily knock-in
    that means one fixing per business day, and a missing one is an error rather
    than a silent assumption.  Knock-out wins on a tied date; a knock-out after a
    knock-in still settles (it pays the coupon).
    """
    valuation = to_datetime(market.valuation_date)
    spot0, _ = resolve_spot0(contract, market)
    ko_levels = tuple(float(ratio) * spot0 for ratio in contract.ko_levels)
    ki_level = float(contract.ki_level) * spot0

    observations = {
        day: index for index, day in enumerate(contract.observation_dates)
    }
    monitoring = contract.ki_monitoring_dates(
        market.calendar, contract.start_date, valuation
    )

    fixings = contract.history_map
    knocked_in: Optional[datetime] = None
    for when in monitoring:
        spot = fixings.get(when.date())
        if spot is None:
            raise ValueError(
                "missing history fixing for {}: the knock-in is monitored {}; add "
                "the fixings to AutocallContract.history or set "
                "ki_frequency='observation_dates'".format(
                    when.date().isoformat(), contract.ki_frequency
                )
            )
        ko_index = observations.get(when)
        if ko_index is not None and triggers(
            spot, ko_levels[ko_index], contract.ko_boundary, above=True
        ):
            return knocked_in, when
        if knocked_in is None and ki_level > 0.0 and triggers(
            spot, ki_level, contract.ki_boundary, above=False
        ):
            knocked_in = when  # the first monitoring date below the barrier
    return knocked_in, None


def apply_history(contract: AutocallContract, market: MarketState) -> AutocallContract:
    """Return ``contract`` with its past observations replayed onto the ledger dates."""
    knocked_in_date, knocked_out_at = replay_history(contract, market)
    return replace(
        contract,
        knocked_in_date=knocked_in_date,
        knocked_out_at=knocked_out_at,
    )


__all__ = ["apply_history", "replay_history"]
