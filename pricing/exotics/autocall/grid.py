"""Step dates and barrier smoothing shared by both engines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np

from ....core.daycount import to_datetime
from ....core.market import MarketState
from .schedule import AutocallSchedule


@dataclass(frozen=True)
class TimeGrid:
    """Dates the engines step through (valuation -> ... -> expiry)."""

    dates: Tuple[datetime, ...]
    vol_times: Tuple[float, ...]
    forwards: Tuple[float, ...]
    observation_steps: Tuple[int, ...]
    #: Node index of every knock-in monitoring date (same order as
    #: ``AutocallSchedule.ki_dates``); defaults to the knock-out observations for
    #: grids built outside :func:`build_time_grid`.
    ki_steps: Tuple[int, ...] = ()

    @property
    def n_steps(self) -> int:
        return len(self.dates) - 1


def _vol_time_date(
    market: MarketState, after: datetime, before: datetime, target: float
) -> Optional[datetime]:
    """The midnight date in ``(after, before)`` closest to ``target`` in vol time."""
    best: Optional[datetime] = None
    best_gap: Optional[float] = None
    day = after.date() + timedelta(days=1)
    last = before.date()
    while day < last:
        candidate = to_datetime(day)
        gap = abs(float(market.year_fraction(candidate)) - float(target))
        if best_gap is None or gap < best_gap:
            best, best_gap = candidate, gap
        day += timedelta(days=1)
    return best


def build_time_grid(
    schedule: AutocallSchedule,
    market: MarketState,
    *,
    steps_per_observation: int = 1,
    target_step: Optional[float] = None,
) -> TimeGrid:
    """Step dates shared by both engines.

    Every observation date is a node (the barriers are only tested there).
    ``steps_per_observation`` is the *minimum* number of sub-steps of a segment;
    ``target_step`` (vol-time years) raises that count until no step is longer
    than it - the PDE default (0.2, matching edslib).  Both criteria matter: a
    monthly snowball has ~0.083 vol-year segments, so a 0.2 target alone would
    collapse the PDE to a single step per month.

    The interior nodes are placed at **equal vol-time steps**, not at equal
    calendar offsets, and the layout is anchored on the segment (the valuation
    date only enters through the first segment).  Vol time and calendar time are
    far from proportional - the EDS convention counts business days, so a whole
    holiday week contributes nothing - and rounding calendar offsets made a
    one-day theta bump redistribute the variance of the first segment by ~30%
    (its last step went from three business days to two), which showed up as
    +/-2k/day of pure layout noise in both engines.  Equal vol-time steps keep
    the discretisation - and therefore the theta it is supposed to measure -
    stable: every segment but the first is re-used verbatim after the bump.
    """
    monitoring = tuple(schedule.ki_dates)
    if len(monitoring) > len(schedule.observation_dates):
        # Daily knock-in observation: the monitoring dates ARE the time grid, so
        # every node can carry the barrier constraint - edslib does the same by
        # appending the contractual dates to its stepper dates.  Sub-stepping
        # adds nothing here: a business-day step is already finer than any target
        # step, and the knock-in is tested exactly where it is observed.
        points = [to_datetime(market.valuation_date)]
        for day in monitoring:
            if day > points[-1]:
                points.append(day)
        if schedule.expiry_date > points[-1]:
            points.append(schedule.expiry_date)
        return _finalise(
            market,
            points,
            [_node_index(points, day) for day in schedule.observation_dates],
            [_node_index(points, day) for day in monitoring],
        )

    points = [to_datetime(market.valuation_date)]
    observation_steps = []
    previous = points[0]
    for day in schedule.observation_dates:
        start_time = float(market.year_fraction(previous))
        span_time = float(market.year_fraction(day)) - start_time
        if target_step is not None and target_step > 0.0 and span_time > 0.0:
            steps = max(1, int(math.ceil(span_time / target_step)))
        else:
            steps = 1
        steps = max(steps, int(steps_per_observation), 1)
        for index in range(1, steps):
            target = start_time + span_time * index / steps
            candidate = _vol_time_date(market, previous, day, target)
            if candidate is not None and candidate > points[-1]:
                points.append(candidate)
        if day > points[-1]:
            points.append(day)
        observation_steps.append(len(points) - 1)
        previous = day
    if schedule.expiry_date > points[-1]:
        points.append(schedule.expiry_date)
    # The knock-in nodes follow the *contractual* monitoring dates: the
    # simplified convention tests it on every knock-out date, the European one
    # (``ki_frequency="expiry"``) only at maturity - and both are nodes above.
    ki_steps = tuple(_node_index(points, day) for day in schedule.ki_dates)
    return _finalise(market, points, observation_steps, ki_steps or observation_steps)


def _node_index(points, day) -> int:
    for index, item in enumerate(points):
        if item == day:
            return index
    raise ValueError("{} is not a node of the time grid".format(day.date().isoformat()))


def _finalise(
    market: MarketState,
    points,
    observation_steps,
    ki_steps,
) -> TimeGrid:
    """Assemble the grid (forwards and vol times for every node)."""
    # F(0) is the spot by no-arbitrage, while ``forward(valuation_date)`` can
    # pick up a half-day intraday stub from the curve anchor (it is dated at
    # midnight).  Pinning the first node keeps the opening ratio exact and the
    # day-0 / day+1 grids comparable.
    forwards = [float(market.spot)]
    forwards.extend(market.forward(item) for item in points[1:])
    return TimeGrid(
        dates=tuple(points),
        vol_times=tuple(market.year_fraction(item) for item in points),
        forwards=tuple(forwards),
        observation_steps=tuple(observation_steps),
        ki_steps=tuple(ki_steps),
    )


def smooth_indicator(
    distance,
    level: float,
    *,
    enabled: bool = True,
    width: float = 0.01,
    floor: float = 0.0,
) -> np.ndarray:
    """Smoothed step indicator of a barrier event (0 below, 1 above).

    ``distance`` is positive inside the event (``S - KO`` for knock-out,
    ``KI - S`` for knock-in).  With smoothing on, the transition spans
    ``+/- band`` around the level where ``band = max(width * |level|, floor)``.
    Both engines share this helper so the cross-check compares like with like.
    """
    values = np.asarray(distance, dtype=float)
    if not enabled:
        return (values >= 0.0).astype(float)
    band = max(max(float(width), 0.0) * abs(float(level)), max(float(floor), 0.0))
    if band <= 0.0:
        return (values >= 0.0).astype(float)
    u = np.clip((values / band + 1.0) * 0.5, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


__all__ = ["TimeGrid", "build_time_grid", "smooth_indicator"]
