"""Date and business-calendar utilities used by the standalone pricer."""

from datetime import date, datetime, time, timedelta
import json
import re
from typing import Iterable, Optional, Union

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

    Otherwise the function falls back to a simple calendar-day convention.
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

    basis_lower = basis.lower().replace(" ", "")
    denominator = 365.0 if "365" in basis_lower else 360.0
    return max(0.0, (end_dt - start_dt).total_seconds() / 86400.0 / denominator)


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
