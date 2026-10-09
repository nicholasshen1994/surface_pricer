"""Raw (pre-shift) terms of an autocallable - see the package docstring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ....core.daycount import DateLike, resolve_basis, to_date, to_datetime

#: Ways the barrier anchor can be set (``start_spot`` = the contractual start).
_ANCHORS = ("start_spot", "valuation_spot")

#: Monitoring rules a contract can name (``daily`` = the knock-in is observed every
#: business day - the market standard, and what edslib's
#: ``ki_dates = SCHEDULE(..., 'DAILY', ...)`` does; ``expiry`` = the European
#: knock-in, edslib's ``DateFrequency.AT_EXPIRY``, only the final observation can
#: knock in - the classic ACN structure; ``observation_dates`` = the simplified
#: convention that only tests the knock-in together with the knock-out).
#:
#: A *payload* can additionally say ``custom``, which means "the monitoring dates
#: and levels are written out" - see :mod:`...schedule`.
KI_FREQUENCIES = ("daily", "expiry", "observation_dates")

#: The payload spellings are exactly :data:`KI_FREQUENCIES` (2026-10): the
#: synonyms this used to translate (``at_expiry``, ``obs``, ``maturity``,
#: ``expiration``, ...) are refused, one word per rule.

#: Barrier-touch conventions, set **per side**: does landing exactly on the level
#: trigger?  A knock-out needs ``S >= KO`` when inclusive and ``S > KO`` when
#: exclusive; a knock-in ``S <= KI`` / ``S < KI`` the same way.
BOUNDARY_INCLUSIVE = "inclusive"
BOUNDARY_EXCLUSIVE = "exclusive"
BOUNDARIES = (BOUNDARY_INCLUSIVE, BOUNDARY_EXCLUSIVE)

#: The operator-style spellings (``>=``, ``gt``, ``touch``, ``strict``, ...) this
#: used to translate are gone (2026-10): a payload says ``inclusive`` or
#: ``exclusive``, and an operator is an error rather than a second name for one of
#: them.  A **missing** key is not an alias - it is the default (``inclusive``).
def resolve_ki_frequency(value: Optional[str]) -> str:
    """Normalise a knock-in monitoring rule name; raise on anything else.

    Exactly the three names in :data:`KI_FREQUENCIES`, one spelling each.
    """
    text = str(value or "daily").strip().lower()
    if text not in KI_FREQUENCIES:
        raise ValueError(
            "ki_frequency must be one of {} (no aliases), got {!r}".format(
                ", ".join(KI_FREQUENCIES), value
            )
        )
    return text


def resolve_boundary(value: Optional[str]) -> str:
    """Normalise a barrier-touch convention (``inclusive`` / ``exclusive``).

    ``inclusive`` is the default and the market convention: a knock-out triggers
    the moment the spot *is* at its level, a knock-in likewise.  ``exclusive``
    needs the spot strictly through it.
    """
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return BOUNDARY_INCLUSIVE
    if text not in BOUNDARIES:
        raise ValueError(
            "boundary must be {} (no aliases such as '>=' / 'gt' / 'touch'), "
            "got {!r}".format(" or ".join(BOUNDARIES), value)
        )
    return text


def triggers(spot: float, level: float, boundary: str, *, above: bool) -> bool:
    """Whether ``spot`` triggers ``level`` under ``boundary``.

    ``above=True`` is the knock-out side (``spot >= level`` when inclusive),
    ``above=False`` the knock-in side (``spot <= level``).  Shared by the ledger
    replay and the same-day determination, so every **deterministic** decision -
    where landing exactly on the level is a real, measurable case - agrees.
    """
    if above:
        return spot >= level if boundary == BOUNDARY_INCLUSIVE else spot > level
    return spot <= level if boundary == BOUNDARY_INCLUSIVE else spot < level


def monitoring_grid(
    frequency: str,
    valuation: DateLike,
    expiry: DateLike,
    calendar=None,
    observation_dates: Sequence[DateLike] = (),
) -> Tuple[datetime, ...]:
    """Monitoring dates a **rule** implies, in ``(valuation, expiry]``, ascending.

    ``daily`` walks the calendar's business days and always adds the observation
    dates (an observation date is monitored whether or not it is a business day);
    ``expiry`` returns the maturity only; ``observation_dates`` returns the knock-out
    observations.  This is what lets a payload carry ``frequency`` alone instead of
    enumerating ~250 dates, and it is the same rule
    :meth:`AutocallContract.ki_monitoring_dates` applies when it replays history.
    """
    first = to_datetime(valuation)
    last = to_datetime(expiry)
    observations = tuple(
        day for day in (to_datetime(value) for value in observation_dates) if first < day <= last
    )
    if frequency == "expiry":
        return (last,) if first < last else ()
    if frequency == "observation_dates":
        return observations

    dates = set(observations)
    day = to_date(first) + timedelta(days=1)
    while day <= to_date(last):
        if calendar is None or calendar.is_business_day(day):
            dates.add(to_datetime(day))
        day += timedelta(days=1)
    return tuple(sorted(dates))


@dataclass
class AutocallContract:
    """Raw (pre-shift) terms of a standard single-underlying autocallable.

    Barriers are expressed as **ratios of the start spot** (``0.75`` = 75%),
    the same ratio space edslib uses; ``build_schedule`` turns them into
    absolute levels.  ``history`` carries the spot fixings of observations that
    already happened (required as soon as the valuation date is past one).
    """

    underlying: str
    start_date: DateLike
    expiry_date: DateLike
    observation_dates: Tuple[DateLike, ...]
    ko_levels: Tuple[float, ...] = (1.0,)
    ki_level: float = 0.0
    # The knock-in is a daily close observation in the market (and in edslib);
    # the barriers are still anchors on ratios of the start spot either way.
    ki_frequency: str = "daily"
    #: Barrier-touch convention **per side** - does landing exactly on the level
    #: trigger?  Contractual, not numerical: it decides the fixings and the
    #: same-day determination, and both sides travel in the payload
    #: (``knock_out.boundary`` / ``knock_in.boundary``).
    ko_boundary: str = "inclusive"
    ki_boundary: str = "inclusive"
    #: Annualised knock-out coupon: one value for every observation, or one per
    #: observation (a step-up / step-down schedule, ``(0.13, 0.14, 0.15)``).  It is
    #: paid on knock-out together with the principal, accrued from the start date
    #: under ``day_count`` - see ``rebate`` for the leg that pays when nothing
    #: triggers.
    annual_coupon: Any = 0.0
    notional: float = 1.0
    #: Day count of the coupon / rebate accrual (``act/365f`` default, or
    #: ``act/360`` / ISDA ``act/act``).  It applies from ``start_date`` to each
    #: observation, i.e. over the whole period even when the trade is valued
    #: mid-life - see :func:`..cashflows.accrual`.
    day_count: str = "act/365f"
    protected_principal: float = 0.0
    ki_gearing: float = 1.0
    #: Strike of the knock-in loss leg, as a ratio of the start spot: ``1.0``
    #: measures the loss from the start spot (the plain snowball), ``0.9`` is an
    #: OTM structure whose loss only starts below 90% -
    #: ``notional * max(protected, 1 - gearing * (1 - min(S_T / (ki_strike *
    #: spot0), 1)))``.  The barrier itself stays ``ki_level``; this only moves the
    #: settlement leg.
    ki_strike: float = 1.0
    #: Annualised rate of the "neither knocked out nor knocked in" leg: the payoff
    #: at expiry is ``notional * (1 + rebate * accrual(start, expiry))`` - the same
    #: accrual shape (and basis) as the knock-out coupon, which is why the two are
    #: allowed to differ.  ``None`` means "the last observation's coupon rate".
    rebate: Optional[float] = None
    settlement_days: int = 0
    start_spot: Optional[float] = None
    history: Tuple[Tuple[DateLike, float], ...] = ()
    product_type: str = "autocallable"
    shift_override: Optional[Mapping[str, Any]] = None
    anchor: str = "start_spot"
    # Observation history is **ledger** data, not pricing state: it is replayed
    # by ``autocall.history.apply_history`` *outside* the engines, which then see
    # nothing but the knock-in **date** and the knock-out date.  The engines never
    # read fixings and never have to decide whether a past observation settled.
    #: The date the knock-in happened (``None`` = not knocked in).  A date, not a
    #: flag: "was it knocked in when we value it?" is a question about the
    #: valuation date, so moving a valuation back before this date re-values the
    #: trade pre-knock-in without touching the payload.
    knocked_in_date: Optional[DateLike] = None
    knocked_out_at: Optional[DateLike] = None

    def __post_init__(self):
        self.underlying = str(self.underlying or "").strip().upper()
        if not self.underlying:
            raise ValueError("underlying must not be empty")
        self.product_type = str(self.product_type or "autocallable").strip().lower()

        self.start_date = to_datetime(self.start_date)
        self.expiry_date = to_datetime(self.expiry_date)
        if self.expiry_date <= self.start_date:
            raise ValueError(
                "expiry_date {} must be after start_date {}".format(
                    self.expiry_date.date(), self.start_date.date()
                )
            )

        observations = tuple(to_datetime(value) for value in (self.observation_dates or ()))
        if not observations:
            raise ValueError("observation_dates must not be empty")
        if any(value <= self.start_date for value in observations):
            raise ValueError("every observation date must be after start_date")
        if observations[-1] > self.expiry_date:
            raise ValueError("observation_dates must not go past expiry_date")
        if any(right <= left for left, right in zip(observations, observations[1:])):
            raise ValueError("observation_dates must be strictly increasing")
        self.observation_dates = observations

        levels = tuple(float(value) for value in (self.ko_levels or ()))
        if not levels or any(value <= 0.0 for value in levels):
            raise ValueError("ko_levels must be positive ratios of the start spot")
        if len(levels) not in (1, len(observations)):
            raise ValueError(
                "ko_levels must hold one value or one per observation "
                "({} given, {} observations)".format(len(levels), len(observations))
            )
        if len(levels) == 1:
            levels = levels * len(observations)
        self.ko_levels = levels

        self.ki_level = float(self.ki_level)
        if self.ki_level < 0.0:
            raise ValueError("ki_level must be a non-negative ratio of the start spot")
        rates = self.annual_coupon
        if isinstance(rates, (int, float)):
            rates = (float(rates),)
        else:
            rates = tuple(float(value) for value in rates)
        if not rates:
            rates = (0.0,)
        if any(value < 0.0 for value in rates):
            raise ValueError("annual_coupon must not be negative")
        if len(rates) == 1:
            rates = rates * len(observations)
        elif len(rates) != len(observations):
            raise ValueError(
                "annual_coupon must hold one value or one per observation "
                "({} given, {} observations)".format(len(rates), len(observations))
            )
        self.annual_coupon = rates
        if self.rebate is not None:
            self.rebate = float(self.rebate)
            if self.rebate < 0.0:
                raise ValueError("rebate must not be negative")
        self.day_count = resolve_basis(self.day_count)
        self.notional = float(self.notional)
        if self.notional <= 0.0:
            raise ValueError("notional must be positive")
        self.protected_principal = float(self.protected_principal)
        if not 0.0 <= self.protected_principal <= 1.0:
            raise ValueError("protected_principal must be a ratio in [0, 1]")
        self.ki_gearing = float(self.ki_gearing)
        if self.ki_gearing < 0.0:
            raise ValueError("ki_gearing must not be negative")
        self.ki_strike = float(self.ki_strike)
        if self.ki_strike <= 0.0:
            raise ValueError("ki_strike must be a positive ratio of the start spot")
        self.settlement_days = int(self.settlement_days)
        if self.settlement_days < 0:
            raise ValueError("settlement_days must not be negative")
        if self.start_spot is not None:
            self.start_spot = float(self.start_spot)
            if self.start_spot <= 0.0:
                raise ValueError("start_spot must be positive")

        history = tuple(
            (to_datetime(day), float(spot)) for day, spot in (self.history or ())
        )
        self.history = tuple(sorted(history, key=lambda item: item[0]))

        if self.knocked_in_date is not None:
            self.knocked_in_date = to_datetime(self.knocked_in_date)
            if self.knocked_in_date <= self.start_date:
                raise ValueError(
                    "knocked_in_date {} must be after start_date {}".format(
                        self.knocked_in_date.date(), self.start_date.date()
                    )
                )
            if self.knocked_in_date > self.expiry_date:
                raise ValueError(
                    "knocked_in_date {} must not be past expiry_date {}".format(
                        self.knocked_in_date.date(), self.expiry_date.date()
                    )
                )
        if self.knocked_out_at is not None:
            self.knocked_out_at = to_datetime(self.knocked_out_at)
            if (
                self.knocked_in_date is not None
                and self.knocked_out_at < self.knocked_in_date
            ):
                raise ValueError(
                    "knocked_out_at {} must not be before knocked_in_date {}: "
                    "a settled trade cannot knock in afterwards".format(
                        self.knocked_out_at.date(), self.knocked_in_date.date()
                    )
                )

        self.ko_boundary = resolve_boundary(self.ko_boundary)
        self.ki_boundary = resolve_boundary(self.ki_boundary)
        self.ki_frequency = resolve_ki_frequency(self.ki_frequency)

        anchor = str(self.anchor or "start_spot").strip().lower()
        if anchor not in _ANCHORS:
            raise ValueError(
                "anchor must be one of {}, got {!r}".format(", ".join(_ANCHORS), self.anchor)
            )
        self.anchor = anchor

    @property
    def n_observations(self) -> int:
        return len(self.observation_dates)

    @property
    def coupon_rate(self) -> float:
        """The base (first) coupon rate.

        The barrier-shift rule sizes its default step off "the coupon", and a
        step-up schedule has no single one: the initial coupon is what the first
        observations are shifted by, and it is the one a term sheet quotes.
        """
        rates = self.annual_coupon
        if isinstance(rates, (int, float)):  # duck-typed callers (tests, term sheets)
            return float(rates)
        return float(rates[0]) if rates else 0.0

    @property
    def history_map(self) -> Dict[Any, float]:
        """Fixings keyed by calendar date (duplicates: the last one wins)."""
        return {day.date(): spot for day, spot in self.history}

    def ki_monitoring_dates(
        self, calendar, start: DateLike, end: DateLike
    ) -> Tuple[Any, ...]:
        """Knock-in observation dates in ``(start, end]``, ascending.

        The rule itself lives in :func:`monitoring_grid` (the payload reader applies
        the same one when it has to rebuild a grid), so history replay and a JSON-fed
        contract can never disagree about which days were monitored.
        """
        return monitoring_grid(
            self.ki_frequency, start, end, calendar, self.observation_dates
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot of the raw terms (for reports / JSON)."""
        return {
            "underlying": self.underlying,
            "product_type": self.product_type,
            "start_date": self.start_date.date().isoformat(),
            "expiry_date": self.expiry_date.date().isoformat(),
            "observation_dates": [day.date().isoformat() for day in self.observation_dates],
            "ko_levels": [float(value) for value in self.ko_levels],
            "ki_level": float(self.ki_level),
            "ki_frequency": self.ki_frequency,
            "annual_coupon": [float(value) for value in self.annual_coupon],
            "rebate": None if self.rebate is None else float(self.rebate),
            "day_count": self.day_count,
            "notional": float(self.notional),
            "protected_principal": float(self.protected_principal),
            "ki_gearing": float(self.ki_gearing),
            "ki_strike": float(self.ki_strike),
            "settlement_days": int(self.settlement_days),
            "start_spot": None if self.start_spot is None else float(self.start_spot),
            "anchor": self.anchor,
            "knocked_in_date": (
                None
                if self.knocked_in_date is None
                else self.knocked_in_date.date().isoformat()
            ),
            "knocked_out_at": (
                None if self.knocked_out_at is None else self.knocked_out_at.date().isoformat()
            ),
            "shift_override": None if self.shift_override is None else dict(self.shift_override),
        }


__all__ = [
    "AutocallContract",
    "KI_FREQUENCIES",
    "monitoring_grid",
    "resolve_ki_frequency",
]
