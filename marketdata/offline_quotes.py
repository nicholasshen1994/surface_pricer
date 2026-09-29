"""Listed quote containers for offline / JSON driven workflows.

The live fit pipeline consumes raw records from :mod:`providers` and cleans
them in :mod:`fit_prepare`; these containers remain for serialization and for
callers that already hold a quote table.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from ..core.daycount import DateLike, to_datetime


@dataclass
class OptionQuote:
    strike: float
    call_bid: float = 0.0
    call_ask: float = 0.0
    put_bid: float = 0.0
    put_ask: float = 0.0
    call_last: Optional[float] = None
    put_last: Optional[float] = None
    volume: Optional[float] = None

    @classmethod
    def from_dict(cls, value: dict) -> "OptionQuote":
        return cls(
            strike=float(value["strike"]),
            call_bid=float(value.get("call_bid", 0.0) or 0.0),
            call_ask=float(value.get("call_ask", 0.0) or 0.0),
            put_bid=float(value.get("put_bid", 0.0) or 0.0),
            put_ask=float(value.get("put_ask", 0.0) or 0.0),
            call_last=value.get("call_last"),
            put_last=value.get("put_last"),
            volume=value.get("volume"),
        )

    @property
    def call_mid(self) -> float:
        return 0.5 * (self.call_bid + self.call_ask)

    @property
    def put_mid(self) -> float:
        return 0.5 * (self.put_bid + self.put_ask)

    def is_valid(self) -> bool:
        return (
            self.strike > 0.0
            and self.call_bid >= 0.0
            and self.call_ask >= self.call_bid
            and self.put_bid >= 0.0
            and self.put_ask >= self.put_bid
        )


@dataclass
class QuoteSlice:
    expiry: DateLike
    spot: float
    quotes: Sequence[OptionQuote]
    start_date: Optional[DateLike] = None
    calendar: Optional[str] = None
    holiday_weight: float = 0.0
    trading_days_per_year: Optional[float] = None

    def __post_init__(self):
        self.expiry = to_datetime(self.expiry)
        self.start_date = to_datetime(self.start_date) if self.start_date is not None else None
        self.spot = float(self.spot)
        self.quotes = tuple(sorted(self.quotes, key=lambda quote: quote.strike))

    @classmethod
    def from_dict(cls, value: dict, default_spot: Optional[float] = None) -> "QuoteSlice":
        raw_quotes = value.get("price_info", value.get("quotes", []))
        spot = value.get("spot", default_spot)
        if spot is None:
            raise ValueError("quote slice requires spot")
        return cls(
            expiry=value.get("expiry_date", value.get("expiry")),
            spot=spot,
            quotes=[
                item if isinstance(item, OptionQuote) else OptionQuote.from_dict(item)
                for item in raw_quotes
            ],
            start_date=value.get("start_date"),
            calendar=value.get("calendar"),
            holiday_weight=float(value.get("holiday_weight", value.get("holiday_weight_factor", 0.0)) or 0.0),
            trading_days_per_year=value.get("trading_days_per_year"),
        )

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date.isoformat(sep=" ") if self.start_date else None,
            "expiry_date": self.expiry.isoformat(sep=" "),
            "spot": self.spot,
            "calendar": self.calendar,
            "holiday_weight": self.holiday_weight,
            "trading_days_per_year": self.trading_days_per_year,
            "price_info": [
                {
                    "strike": quote.strike,
                    "call_bid": quote.call_bid,
                    "call_ask": quote.call_ask,
                    "put_bid": quote.put_bid,
                    "put_ask": quote.put_ask,
                }
                for quote in self.quotes
            ],
        }


__all__ = ["OptionQuote", "QuoteSlice"]
