"""Date and business-calendar utilities used by the standalone pricer."""

from datetime import date, datetime, time, timedelta
import json
import re
from typing import Iterable, Optional, Tuple, Union

import numpy as np


DateLike = Union[str, date, datetime]


def to_datetime(value: DateLike) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.strptime(text[:10], "%Y-%m-%d")


def to_date(value: DateLike) -> date:
    return to_datetime(value).date()


class BusinessCalendar:
    """A small explicit calendar.

    The standalone package does not import the repository calendar engine.
    Callers can pass holiday dates directly, or load a compatible JSON file
    containing ``{"CALENDAR": {"holidays": [...]}}`` entries.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        holidays: Optional[Iterable[DateLike]] = None,
        weekmask: str = "1111100",
    ):
        self.name = name
        self.weekmask = weekmask
        self.holidays = tuple(sorted({to_date(value) for value in (holidays or [])}))

    @classmethod
    def from_json(cls, path: str, name: str) -> "BusinessCalendar":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        item = payload[name]
        return cls(name=name, holidays=item.get("holidays", []))

    def is_business_day(self, value: DateLike) -> bool:
        current = to_date(value)
        return bool(
            np.is_busday(
                np.datetime64(current),
                weekmask=self.weekmask,
                holidays=np.array(self.holidays, dtype="datetime64[D]"),
            )
        )

    def business_days(self, start: DateLike, end: DateLike) -> float:
        start_date = to_date(start)
        end_date = to_date(end)
        return float(
            np.busday_count(
                start_date,
                end_date,
                weekmask=self.weekmask,
                holidays=np.array(self.holidays, dtype="datetime64[D]"),
            )
        )

    def next_business_day(self, value: DateLike) -> date:
        """``value`` itself when it is a business day, else the following one.

        Only forward: a date that falls on a holiday moves to the next open day,
        which is what a term sheet's roll convention says.  Rolling backwards
        would settle *before* the scheduled date, so it is not offered here.
        """
        current = to_date(value)
        while not self.is_business_day(current):
            current += timedelta(days=1)
        return current


class DateHelperBusinessCalendar(BusinessCalendar):
    """China business calendar using the bundled ``DateHelper`` convention.

    Volatility time follows the EDS convention implemented in
    :func:`year_fraction`: business days plus a fraction of the non-business
    days, annualized by the supplied trading-day count.
    """

    def __init__(self, market: str = "SHX"):
        from .calendars import DateHelper

        self.date_helper = DateHelper(market)
        super().__init__(
            name=market,
            holidays=self.date_helper.holiday,
            weekmask="1111100",
        )


def _time_of_day_fraction(value: datetime) -> float:
    start_of_day = datetime.combine(value.date(), time.min)
    return (value - start_of_day).total_seconds() / 86400.0


def year_fraction(
    start: DateLike,
    end: DateLike,
    calendar: Optional[BusinessCalendar] = None,
    trading_days_per_year: Optional[float] = None,
    holiday_weight: float = 0.0,
    basis: str = "act/365f",
) -> float:
    """Return the model year fraction.

    With a calendar and ``trading_days_per_year`` this follows the EDS vol-time
    convention, matching edslib ``DateUtil.dtcf`` for calendar based periods:

    ``(business_days + (calendar_days - business_days) * holiday_weight) / trading_days_per_year``

    where ``business_days`` counts the start date and excludes the end date,
    and the intraday time of day on both ends is added (scaled by
    ``holiday_weight`` when the endpoint is not a business day).

    Otherwise the function falls back to a calendar-day convention selected by
    ``basis``: ``act/365f`` / ``act/360`` (natural days over a fixed denominator)
    or ``act/act`` (ISDA, see :func:`act_act`).  Coupon accrual goes through here
    without a calendar, so the **contractual** basis wins over the vol-time
    convention exactly where it should.
    """
    start_dt = to_datetime(start)
    end_dt = to_datetime(end)
    if end_dt <= start_dt:
        return 0.0

    if calendar is not None and trading_days_per_year:
        start_date = start_dt.date()
        end_date = end_dt.date()
        business_days = calendar.business_days(start_date, end_date)
        if holiday_weight:
            calendar_days = float((end_date - start_date).days)
            business_days += (calendar_days - business_days) * holiday_weight

        start_fraction = _time_of_day_fraction(start_dt)
        end_fraction = _time_of_day_fraction(end_dt)
        if not calendar.is_business_day(start_dt):
            start_fraction *= holiday_weight
        if not calendar.is_business_day(end_dt):
            end_fraction *= holiday_weight
        business_days += end_fraction - start_fraction
        return max(0.0, business_days / float(trading_days_per_year))

    basis_lower = basis.lower().replace(" ", "").replace("/", "").replace(".", "")
    if basis_lower in {"actact", "actualactual", "actactisda", "actualactualisda"}:
        return max(0.0, act_act(start_dt, end_dt))
    denominator = 365.0 if "365" in basis_lower else 360.0
    return max(0.0, (end_dt - start_dt).total_seconds() / 86400.0 / denominator)


#: Day counts accepted for coupon / rebate accrual (see :func:`resolve_basis`).
ACCRUAL_BASES = ("act/365f", "act/360", "act/act")

#: The accepted names, keyed by their punctuation-free form: ``Act/365F`` and
#: ``ACT-ACT`` still read, while a *different* name - ``act365``, ``actualactual``,
#: ``actactisda``, the synonyms this used to translate (2026-10) - is refused.  One
#: name per basis, so a payload cannot say "365 fixed" two ways.
_BASIS_BY_KEY = {
    basis.replace("/", "").replace("-", "").replace(".", ""): basis
    for basis in ACCRUAL_BASES
}


def resolve_basis(value: Optional[str]) -> str:
    """Normalise an accrual day count (``None`` -> ``act/365f``); raise if unknown.

    Validating here rather than in :func:`year_fraction` keeps that function's
    historical leniency (anything without a "365" counts as 360) while the
    *contractual* accrual basis is checked once, where a typo would otherwise
    silently price a different coupon.  Case and separators are tolerated; the
    alias names are not.
    """
    key = (
        str(value if value is not None else "act/365f")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("/", "")
        .replace("-", "")
        .replace(".", "")
    )
    resolved = _BASIS_BY_KEY.get(key)
    if resolved is None:
        raise ValueError(
            "unsupported day count {!r}: choose from {}".format(
                value, ", ".join(ACCRUAL_BASES)
            )
        )
    return resolved


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def act_act(start: DateLike, end: DateLike) -> float:
    """ISDA act/act year fraction: each calendar year over its own length.

    A coupon period spanning a year end accrues ``d1/365 + d2/366`` (or the other
    way round), which is what an act/act term sheet pays - a single ``days/365``
    would be short by the leap-day adjustment.
    """
    start_dt = to_datetime(start)
    end_dt = to_datetime(end)
    if end_dt <= start_dt:
        return 0.0
    total = 0.0
    for year in range(start_dt.year, end_dt.year + 1):
        begin = max(start_dt, datetime(year, 1, 1))
        finish = min(end_dt, datetime(year + 1, 1, 1))
        if finish > begin:
            days = (finish - begin).total_seconds() / 86400.0
            total += days / (366.0 if _is_leap_year(year) else 365.0)
    return total


def shift_days(value: DateLike, days: int) -> datetime:
    return to_datetime(value) + timedelta(days=int(days))


# ------------------------------------------------------------------ tenors
_TENOR_PATTERN = re.compile(r"^\s*(\d+)\s*([DWMY])\s*$", re.IGNORECASE)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _add_months(anchor: date, months: int) -> date:
    month_index = anchor.month - 1 + int(months)
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = min(anchor.day, _days_in_month(year, month))
    return date(year, month, day)


def add_tenor(anchor: DateLike, tenor: str) -> date:
    """Add a ``"3Y"`` / ``"18M"`` / ``"90D"`` / ``"2W"`` tenor to a date."""
    match = _TENOR_PATTERN.match(str(tenor))
    if match is None:
        raise ValueError(
            "invalid tenor {!r}; expected a number followed by D, W, M or Y".format(tenor)
        )
    count = int(match.group(1))
    unit = match.group(2).upper()
    base = to_date(anchor)
    if unit == "D":
        return base + timedelta(days=count)
    if unit == "W":
        return base + timedelta(weeks=count)
    if unit == "M":
        return _add_months(base, count)
    return _add_months(base, 12 * count)


def shift_tenor(
    anchor: DateLike,
    tenor: str,
    calendar: Optional[BusinessCalendar] = None,
) -> date:
    """``add_tenor`` followed by a roll forward to the next business day."""
    target = add_tenor(anchor, tenor)
    if calendar is None:
        return target
    while not calendar.is_business_day(target):
        target += timedelta(days=1)
    return target


def month_grid(
    end: DateLike,
    months: int,
    after: DateLike,
    calendar: Optional[BusinessCalendar] = None,
) -> Tuple[date, ...]:
    """Dates stepping back from ``end`` by ``months``, ascending.

    ``end`` itself plus every ``months`` months earlier, keeping only what is
    **strictly after** ``after`` - the observation grid of a periodic note
    (``months=3`` = quarterly, counted back from the expiry).  An ``end`` that is
    not after ``after`` yields nothing, which is how a caller notices that a term
    sheet has no observation left.

    With a ``calendar`` every date is rolled **forward to the next business day**:
    an observation cannot fall on a holiday, and a weekend or a Golden-week date
    moves to the next open day (the expiry included, where that is a no-op
    because it comes off ``shift_tenor``).  Dates that meet after rolling collapse
    into one, so the result stays strictly increasing - it may just be shorter
    than the raw grid.
    """
    step = int(months)
    if step <= 0:
        raise ValueError("months must be positive, got {}".format(months))
    last = to_date(end)
    boundary = to_date(after)
    if last <= boundary:
        return ()
    dates: list = [last]
    cursor = last
    while True:
        cursor = _add_months(cursor, -step)
        if cursor <= boundary:
            break
        dates.append(cursor)
    if calendar is None:
        return tuple(sorted(dates))
    return tuple(sorted({calendar.next_business_day(day) for day in dates}))
