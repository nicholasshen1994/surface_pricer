"""Lifecycle handling of real trades: status, observation dates, KI/KO state.

The schedule answers three questions for a term sheet at the valuation date:

* is the trade ``not_started`` / ``active`` / ``expired``;
* which observation dates have already passed, and when is the next one;
* has the barrier been breached (knock-in / knock-out), and does that agree
  with the ``ki_flag`` / ``ko_flag`` columns of the sheet.

When both a barrier level and an observation history are given, the flags are
derived from the history and cross-checked against the sheet (a mismatch is an
error: silent disagreement is worse than a failed run).  When only flags are
given they are taken as authoritative; when only barriers are given nothing is
concluded (there is no history to judge from).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

from ..core.daycount import BusinessCalendar, DateLike, to_datetime
from .terms import TradeTerms


class TradeStatus:
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    EXPIRED = "expired"

    ALL = (NOT_STARTED, ACTIVE, EXPIRED)


@dataclass
class TradeSchedule:
    status: str
    valuation_date: date
    start_date: Optional[date]
    expiry_date: date
    observation_dates: Tuple[date, ...] = ()
    past_observations: Tuple[date, ...] = ()
    next_observation: Optional[date] = None
    ki_triggered: bool = False
    ko_triggered: bool = False
    ki_source: str = "sheet"  # "history" | "sheet" | "none"
    ko_source: str = "sheet"
    notes: List[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == TradeStatus.ACTIVE

    @property
    def is_expired(self) -> bool:
        return self.status == TradeStatus.EXPIRED

    @property
    def is_not_started(self) -> bool:
        return self.status == TradeStatus.NOT_STARTED


def build_schedule(
    terms: TradeTerms,
    valuation_date: DateLike,
    calendar: Optional[BusinessCalendar] = None,
) -> TradeSchedule:
    """Resolve the lifecycle state of ``terms`` at ``valuation_date``."""
    valuation = to_datetime(valuation_date).date()
    expiry = to_datetime(terms.expiry_date).date()
    start = to_datetime(terms.start_date).date() if terms.start_date is not None else None

    if start is not None and valuation < start:
        status = TradeStatus.NOT_STARTED
    elif valuation >= expiry:
        status = TradeStatus.EXPIRED
    else:
        status = TradeStatus.ACTIVE

    observation_dates = tuple(record.date.date() for record in terms.observations)
    past = tuple(value for value in observation_dates if value <= valuation)
    future = tuple(value for value in observation_dates if value > valuation)

    ki_triggered, ki_source = _resolve_trigger(
        terms=terms,
        valuations=(record.spot for record in terms.observations if record.date.date() <= valuation),
        barrier=terms.ki_barrier,
        sheet_flag=terms.ki_flag,
        direction="ki",
        calendar=calendar,
    )
    ko_triggered, ko_source = _resolve_trigger(
        terms=terms,
        valuations=(record.spot for record in terms.observations if record.date.date() <= valuation),
        barrier=terms.ko_barrier,
        sheet_flag=terms.ko_flag,
        direction="ko",
        calendar=calendar,
    )

    notes: List[str] = []
    if status == TradeStatus.NOT_STARTED and start is not None:
        notes.append("trade starts on {}".format(start.isoformat()))
    if status == TradeStatus.EXPIRED:
        notes.append("expired on {}".format(expiry.isoformat()))
    if ki_triggered:
        notes.append("knock-in already triggered")
    if ko_triggered:
        notes.append("knock-out already triggered")

    return TradeSchedule(
        status=status,
        valuation_date=valuation,
        start_date=start,
        expiry_date=expiry,
        observation_dates=observation_dates,
        past_observations=past,
        next_observation=future[0] if future else None,
        ki_triggered=ki_triggered,
        ko_triggered=ko_triggered,
        ki_source=ki_source,
        ko_source=ko_source,
        notes=notes,
    )


def _resolve_trigger(
    *,
    terms: TradeTerms,
    valuations,
    barrier: Optional[float],
    sheet_flag: bool,
    direction: str,
    calendar: Optional[BusinessCalendar],
) -> Tuple[bool, str]:
    """Derive one barrier flag and cross-check it against the term sheet."""
    label = "ki" if direction == "ki" else "ko"
    spots = [float(value) for value in valuations]
    if barrier is None or not spots:
        # nothing to derive from: trust the sheet
        return bool(sheet_flag), "sheet" if sheet_flag else "none"

    if direction == "ki":
        triggered = any(spot <= float(barrier) for spot in spots)
        detail = "spot <= ki_barrier ({})".format(barrier)
    else:
        triggered = any(spot >= float(barrier) for spot in spots)
        detail = "spot >= ko_barrier ({})".format(barrier)

    if bool(sheet_flag) != triggered:
        raise ValueError(
            "{} flag mismatch for {}: term sheet says {}, observation history says {} "
            "({})".format(
                label.upper(),
                terms.trade_id,
                bool(sheet_flag),
                triggered,
                detail,
            )
        )
    return triggered, "history"


__all__ = ["TradeSchedule", "TradeStatus", "build_schedule"]
