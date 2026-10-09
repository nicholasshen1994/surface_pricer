"""Autocallable (standard snowball): terms, effective schedule, MC + PDE engines.

The product is layered one way only::

    contract.py    AutocallContract   raw terms - barriers as ratios of the
                                      start spot, coupon, observations, plus an
                                      optional shift override
        |  pricing.rules.barrier_shift.resolve_shift  (config / contract / cli)
        v
    schedule.py    build_schedule()   THE ONLY place a barrier shift is applied:
                                      expands relative/absolute specs into
                                      per-observation *absolute* levels (step
                                      accrual, cap/floor, past observations kept,
                                      anchored on the start spot)
        v
    schedule.py    AutocallSchedule   frozen effective terms AND the cash-flow
                                      inputs (notional, per-observation coupon
                                      rates, rebate, loss-leg terms); it is also
                                      the JSON layer - ``to_dict`` exports the
                                      resolved contract, ``from_dict`` reads a
                                      hand-edited one back
        |  grid.py       TimeGrid / build_time_grid / smooth_indicator
        |  cashflows.py  accrual / ko_cash_flow / rebate_cash_flow / expiry_cash_flow
        v
    mc.py | pde.py  the engines - they take nothing but the schedule: no ratios,
                    no shift, no term-sheet rules to interpret

Cash-flow conventions (fixed here, shared by the engines):

* discrete monitoring only - knock-out and knock-in are checked on the
  observation dates; the knock-in is **not** absorbing (a knocked-in path that
  later reaches a knock-out level still knocks out and takes the coupon) and on a
  tied date the knock-out wins - it is tested after the knock-in; the KI grid is
  ``ki_frequency``: ``daily`` (business days, the market standard), ``expiry``
  (European knock-in, edslib's ``at_expiry``) or ``observation_dates``;
* on knock-out: principal plus the coupon accrued from the start date to that
  observation (paid ``settlement_days`` later);
* at expiry without knock-out: principal plus the full coupon to expiry;
* at expiry after a knock-in: ``max(protected_principal, 1 - gearing *
  (1 - min(S_T / (ki_strike * S_0), 1)))`` times notional - the short-put
  settlement, struck at ``ki_strike`` (ratio of the start spot; 1.0 = the plain
  snowball) and floored by the protected principal (no coupon);
* past observations must be supplied through ``AutocallContract.history``
  (spot fixings) and replayed **outside** the engines by
  :func:`autocall.history.apply_history`, which turns them into the
  ``knocked_in_date`` / ``knocked_out_at`` ledger dates; a missing fixing is an
  error, not a silent assumption.  The engines read the *derived*
  ``AutocallSchedule.knocked_in_before`` (date vs valuation date), so moving a
  valuation back before the knock-in re-values the trade pre-knock-in.
"""

from __future__ import annotations

from typing import Any

from .. import ExoticPricer, register_pricer
from .cashflows import accrual, expiry_cash_flow, ko_cash_flow, rebate_cash_flow
from .contract import AutocallContract
from .grid import TimeGrid, build_time_grid, smooth_indicator
from .history import apply_history, replay_history
from .schedule import (
    AutocallSchedule,
    build_schedule,
    resolve_spot0,
    resolve_trigger_basis,
)


def pricer_for(method: str = "mc", **options: Any) -> ExoticPricer:
    """Factory used by the registry: ``"mc"`` (default) or ``"pde"``."""
    name = str(method or "mc").strip().lower()
    if name in ("pde", "fd", "fdm", "finite_difference"):
        from .pde import AutocallPDE

        return AutocallPDE(**options)
    if name in ("mc", "monte_carlo", "montecarlo"):
        from .mc import AutocallMonteCarlo

        return AutocallMonteCarlo(**options)
    raise ValueError("unknown pricing method {!r}; use 'mc' or 'pde'".format(method))


for _alias in ("autocallable", "autocall", "snowball"):
    register_pricer(_alias, pricer_for)


__all__ = [
    "AutocallContract",
    "AutocallSchedule",
    "TimeGrid",
    "accrual",
    "apply_history",
    "build_schedule",
    "build_time_grid",
    "expiry_cash_flow",
    "ko_cash_flow",
    "rebate_cash_flow",
    "pricer_for",
    "replay_history",
    "resolve_spot0",
    "resolve_trigger_basis",
    "smooth_indicator",
]
