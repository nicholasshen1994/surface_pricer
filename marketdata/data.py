"""Plain market-data containers shared by providers and the fit pipeline.

The fit layer only ever consumes :class:`RawSnapshot`; keeping the containers
free of provider logic lets a new data source plug in behind
:class:`marketdata.providers.MarketDataProvider` without touching the fit code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.daycount import BusinessCalendar


@dataclass(frozen=True)
class OptionQuoteRecord:
    """One listed vanilla quote of a single strike/expiry."""

    underlying: str
    expiry: datetime
    strike: float
    option_type: str
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0
    raw_code: str = ""
    exchange: str = ""


@dataclass(frozen=True)
class SpotRecord:
    """A reference price: index level, ETF price or future price."""

    ticker: str
    price: float
    kind: str = "spot"
    expiry: Optional[datetime] = None
    timestamp: Optional[datetime] = None


@dataclass
class RawSnapshot:
    """Everything the fit pipeline needs from one market snapshot."""

    underlying: str
    valuation_datetime: datetime
    spot: float
    option_records: List[OptionQuoteRecord] = field(default_factory=list)
    spot_records: List[SpotRecord] = field(default_factory=list)
    future_price_by_expiry: Dict[str, float] = field(default_factory=dict)
    rate_curve: Any = None
    borrow_curve: Any = None
    calendar: Optional[BusinessCalendar] = None
    trading_days_per_year: float = 243.0
    holiday_weight: float = 0.05
    diagnostics: Dict[str, Any] = field(default_factory=dict)


__all__ = ["OptionQuoteRecord", "RawSnapshot", "SpotRecord"]
