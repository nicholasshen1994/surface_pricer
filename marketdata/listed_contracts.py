"""Listed option contract parsing for the standalone snapshot pipeline.

Two families are supported:

* CFFEX index options, e.g. ``MO2610-C-7600.CFE`` (third-Friday expiry);
* SSE/SZSE ETF options, e.g. ``510500C2610M06000`` (fourth-Wednesday expiry).

``parse_listed_option`` dispatches on the code shape and returns the unified
``ParsedListedOption`` container used by the market-data providers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

from ..core.daycount import BusinessCalendar


_CFFEX_OPTION_RE = re.compile(
    r"^(?P<underlying>[A-Z]+)(?P<year>\d{2})(?P<month>\d{2})"
    r"-(?P<option_type>[CP])-(?P<strike>\d+(?:\.\d+)?)"
    r"(?:\.[A-Z0-9]+)?$",
    re.IGNORECASE,
)

# SSE/SZSE ETF option trade code: 6-digit underlying + C/P + YYMM + M/A + 5-digit strike.
_ETF_OPTION_RE = re.compile(
    r"^(?P<underlying>\d{6})(?P<option_type>[CP])(?P<year>\d{2})(?P<month>\d{2})"
    r"(?P<adjust>[MA])(?P<strike>\d{5})$",
    re.IGNORECASE,
)

_OPTION_TO_FUTURE_PREFIX = {
    "MO": "IM",
    "IO": "IF",
    "HO": "IH",
}

_ETF_STRIKE_SCALE = 1000.0

EXCHANGE_CFFEX = "CFE"
EXCHANGE_SSE = "SSE"
EXCHANGE_SZSE = "SZSE"


@dataclass(frozen=True)
class ParsedIndexOption:
    raw_code: str
    ticker: str
    underlying: str
    contract_month: str
    expiry: datetime
    option_type: str
    strike: float

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_put(self) -> bool:
        return self.option_type == "put"


@dataclass(frozen=True)
class ParsedListedOption:
    raw_code: str
    underlying: str
    exchange: str
    expiry: datetime
    option_type: str
    strike: float

    @property
    def contract_month(self) -> str:
        return "{:04d}{:02d}".format(self.expiry.year, self.expiry.month)

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_put(self) -> bool:
        return self.option_type == "put"


def parse_index_option_ticker(
    code: str,
    *,
    exchange_suffix: str = "CFE",
    calendar: Optional[BusinessCalendar] = None,
) -> Optional[ParsedIndexOption]:
    text = str(code or "").strip().upper()
    if not text:
        return None
    match = _CFFEX_OPTION_RE.match(text)
    if match is None:
        return None

    year = 2000 + int(match.group("year"))
    month = int(match.group("month"))
    option_flag = match.group("option_type").upper()
    underlying = match.group("underlying").upper()
    contract_month = "{:04d}{:02d}".format(year, month)
    bare_code = (
        "{}{:02d}{:02d}"
        "-{}-{}".format(
            underlying,
            year % 100,
            month,
            option_flag,
            match.group("strike"),
        )
    )
    expiry = cffex_option_expiry_datetime(year, month, calendar=calendar)
    return ParsedIndexOption(
        raw_code=text,
        ticker="{}.{}".format(bare_code, exchange_suffix.upper()),
        underlying=underlying,
        contract_month=contract_month,
        expiry=expiry,
        option_type="call" if option_flag == "C" else "put",
        strike=float(match.group("strike")),
    )


def parse_etf_option_code(
    code: str,
    *,
    calendar: Optional[BusinessCalendar] = None,
    exchange: str = EXCHANGE_SSE,
) -> Optional[ParsedListedOption]:
    """Parse an SSE/SZSE ETF option trade code such as ``510500C2610M06000``."""
    text = str(code or "").strip().upper()
    if not text:
        return None
    match = _ETF_OPTION_RE.match(text)
    if match is None:
        return None

    year = 2000 + int(match.group("year"))
    month = int(match.group("month"))
    option_flag = match.group("option_type").upper()
    strike = float(int(match.group("strike"))) / _ETF_STRIKE_SCALE
    expiry = etf_option_expiry_datetime(year, month, calendar=calendar)
    return ParsedListedOption(
        raw_code=text,
        underlying=match.group("underlying"),
        exchange=exchange,
        expiry=expiry,
        option_type="call" if option_flag == "C" else "put",
        strike=strike,
    )


def parse_listed_option(
    code: str,
    *,
    calendar: Optional[BusinessCalendar] = None,
    etf_exchange: str = EXCHANGE_SSE,
) -> Optional[ParsedListedOption]:
    """Parse any supported listed option code into the unified container."""
    parsed_index = parse_index_option_ticker(code, calendar=calendar)
    if parsed_index is not None:
        return ParsedListedOption(
            raw_code=parsed_index.raw_code,
            underlying=parsed_index.underlying,
            exchange=EXCHANGE_CFFEX,
            expiry=parsed_index.expiry,
            option_type=parsed_index.option_type,
            strike=parsed_index.strike,
        )
    return parse_etf_option_code(code, calendar=calendar, exchange=etf_exchange)


def cffex_option_expiry_date(
    year: int,
    month: int,
    *,
    calendar: Optional[BusinessCalendar] = None,
) -> date:
    """Return CFFEX index option expiry: third Friday, following if holiday."""
    first_day = date(int(year), int(month), 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    expiry = first_friday + timedelta(days=14)
    if calendar is None:
        return expiry
    while not calendar.is_business_day(expiry):
        expiry += timedelta(days=1)
    return expiry


def cffex_option_expiry_datetime(
    year: int,
    month: int,
    *,
    calendar: Optional[BusinessCalendar] = None,
    expiry_time: time = time(15, 0),
) -> datetime:
    expiry = cffex_option_expiry_date(year, month, calendar=calendar)
    return datetime.combine(expiry, expiry_time)


def etf_option_expiry_date(
    year: int,
    month: int,
    *,
    calendar: Optional[BusinessCalendar] = None,
) -> date:
    """Return SSE/SZSE ETF option expiry: fourth Wednesday, following if holiday."""
    first_day = date(int(year), int(month), 1)
    first_wednesday = first_day + timedelta(days=(2 - first_day.weekday()) % 7)
    expiry = first_wednesday + timedelta(days=21)
    if calendar is None:
        return expiry
    while not calendar.is_business_day(expiry):
        expiry += timedelta(days=1)
    return expiry


def etf_option_expiry_datetime(
    year: int,
    month: int,
    *,
    calendar: Optional[BusinessCalendar] = None,
    expiry_time: time = time(15, 0),
) -> datetime:
    expiry = etf_option_expiry_date(year, month, calendar=calendar)
    return datetime.combine(expiry, expiry_time)


def underlying_future_ticker(
    underlying: str,
    contract_month: str,
    *,
    exchange_suffix: str = "CFE",
) -> str:
    """Return the CFFEX future corresponding to an index-option month."""
    option_prefix = str(underlying or "").strip().upper()
    future_prefix = _OPTION_TO_FUTURE_PREFIX.get(option_prefix)
    if future_prefix is None:
        raise ValueError(
            "Unsupported CFFEX index-option underlying: {!r}".format(underlying)
        )
    month = str(contract_month or "").strip()
    if len(month) != 6 or not month.isdigit():
        raise ValueError("contract_month must be YYYYMM, got {!r}".format(contract_month))
    return "{}{}.{}".format(future_prefix, month[2:], str(exchange_suffix).strip().upper())


__all__ = [
    "EXCHANGE_CFFEX",
    "EXCHANGE_SSE",
    "EXCHANGE_SZSE",
    "ParsedIndexOption",
    "ParsedListedOption",
    "cffex_option_expiry_date",
    "cffex_option_expiry_datetime",
    "etf_option_expiry_date",
    "etf_option_expiry_datetime",
    "parse_etf_option_code",
    "parse_index_option_ticker",
    "parse_listed_option",
    "underlying_future_ticker",
]
