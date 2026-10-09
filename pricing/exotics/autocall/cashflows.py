"""Cash-flow rules of the snowball, shared by both engines (pure functions).

Every rule reads the **resolved** schedule - absolute barriers, one coupon rate
per observation, the rebate, the loss-leg terms - so the engines do no term-sheet
arithmetic at all: they apply the monitoring indicators, multiply by the notional
and discount on the schedule's own dates.  Two pieces of arithmetic are left, both
contractual and both carried by the schedule: the **accrual origin**
(``start_date``, the inception - never the valuation date, so a mid-life valuation
still accrues the whole period) and its **day count** (``day_count``: ``act/365f``
by default, ``act/360`` or ISDA ``act/act`` on request).

The three legs:

* **knock-out** (observation ``index``): principal + the coupon accrued to that
  observation, at that observation's own rate;
* **rebate** (expiry, neither knocked out nor knocked in): principal + the rebate
  accrued to expiry - the same shape, a rate the term sheet is free to set
  differently from the knock-out coupon;
* **knock-in** (expiry): a short put struck at ``ki_strike`` x the anchor, geared
  and floored by the protected principal (no coupon).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core.daycount import DateLike, resolve_basis, to_date, year_fraction

if TYPE_CHECKING:  # pragma: no cover - typing only (schedule imports nothing here)
    from .schedule import AutocallSchedule


def accrual_between(start: DateLike, when: DateLike, basis: str = "act/365f") -> float:
    """Accrual between two dates under ``basis``, ignoring any time of day.

    The schedule builder resolves its cash-flow *ratios* with this too, so the
    accrual a contract was priced with and the one its payload reports agree.
    """
    return float(year_fraction(to_date(start), to_date(when), basis=resolve_basis(basis)))


def accrual(schedule: "AutocallSchedule", when: DateLike) -> float:
    """Accrual from the schedule's contractual start to ``when``, in years.

    The origin is ``start_date`` - the **inception**, never the valuation date.  A
    trade valued mid-life still accrues its coupon over the whole period (the
    coupon of the i-th observation is ``rate_i x act/365(start, obs_i)`` however
    late it is valued), and the valuation-date bump that ``theta`` performs must
    not move a single cash flow.  A schedule without a start date therefore
    *fails* instead of quietly accruing from the valuation date, which is exactly
    the "one day short" error this guards against.
    """
    if schedule.start_date is None:
        raise ValueError(
            "the schedule carries no start_date: the accrual origin is "
            "contractual, and falling back to the valuation date would "
            "under-count the coupon (set start_date on the payload/schedule)"
        )
    return accrual_between(
        schedule.start_date, when, basis=getattr(schedule, "day_count", "act/365f")
    )


def rebate_ratio(
    rate: float, start: DateLike, expiry: DateLike, basis: str = "act/365f"
) -> float:
    """Total payoff ratio of the rebate leg: ``1 + rate x accrual(start, expiry)``.

    One formula, used both where the schedule is built from a term sheet and where
    it is read back from a payload, so the two can never drift apart.
    """
    if not rate:
        return 1.0
    return 1.0 + float(rate) * accrual_between(start, expiry, basis=basis)


def ko_cash_flow(schedule: "AutocallSchedule", index: int) -> float:
    """Cash (notional included) when the observation at ``index`` knocks out."""
    when = schedule.observation_dates[index]
    rate = float(schedule.coupon_rates[index])
    return float(schedule.notional) * (1.0 + rate * accrual(schedule, when))


def rebate_cash_flow(schedule: "AutocallSchedule") -> float:
    """Cash (notional included) at expiry when neither leg triggered."""
    return float(schedule.notional) * float(schedule.rebate_ratio)


def expiry_cash_flow(schedule: "AutocallSchedule", spot, knocked_in: bool):
    """Cash at expiry (notional included), vectorised over ``spot``.

    ``knocked_in`` is a scalar flag; callers holding per-path knock-in
    probabilities combine the two branches themselves
    (``p * ki + (1 - p) * no_ki``), which keeps this a pure function of the
    terminal spot.  Returns a float for scalar input, an array otherwise.
    """
    if not knocked_in:
        return rebate_cash_flow(schedule)
    values = np.asarray(spot, dtype=float)
    # The loss leg is a put struck at ``ki_strike`` x the anchor: 1.0 is the plain
    # snowball (loss measured from the start spot), 0.9 an OTM structure whose
    # loss only starts below 90%.
    strike = float(schedule.spot0) * float(schedule.ki_strike)
    if strike:
        performance = np.minimum(values / strike, 1.0)
    else:  # pragma: no cover - defensive: the anchor is always positive
        performance = np.zeros_like(values)
    settlement = 1.0 - float(schedule.ki_gearing) * (1.0 - performance)
    result = float(schedule.notional) * np.maximum(
        settlement, float(schedule.protected_principal)
    )
    return float(result) if values.ndim == 0 else result


__all__ = [
    "accrual",
    "accrual_between",
    "expiry_cash_flow",
    "ko_cash_flow",
    "rebate_cash_flow",
    "rebate_ratio",
]
