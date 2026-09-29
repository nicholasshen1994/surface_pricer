"""Market state shared by the fit and pricing layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .curves import forward
from .daycount import BusinessCalendar, DateLike, to_datetime, year_fraction


@dataclass
class MarketState:
    valuation_date: DateLike
    spot: float
    rate_curve: Any
    borrow_curve: Any = None
    calendar: Optional[BusinessCalendar] = None
    trading_days_per_year: Optional[float] = None
    holiday_weight: float = 0.0
    surface: Any = None
    forward_overrides: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.valuation_date = to_datetime(self.valuation_date)
        self.spot = float(self.spot)

    def clone(self, **changes) -> "MarketState":
        values = {
            "valuation_date": self.valuation_date,
            "spot": self.spot,
            "rate_curve": self.rate_curve,
            "borrow_curve": self.borrow_curve,
            "calendar": self.calendar,
            "trading_days_per_year": self.trading_days_per_year,
            "holiday_weight": self.holiday_weight,
            "surface": self.surface,
            "forward_overrides": dict(self.forward_overrides),
        }
        values.update(changes)
        return MarketState(**values)

    def year_fraction(self, expiry: DateLike) -> float:
        return year_fraction(
            self.valuation_date,
            expiry,
            calendar=self.calendar,
            trading_days_per_year=self.trading_days_per_year,
            holiday_weight=self.holiday_weight,
        )

    def discount_factor(self, expiry: DateLike) -> float:
        if self.rate_curve is None:
            return 1.0
        return float(self.rate_curve.discount_factor(self.valuation_date, expiry))

    def forward(self, expiry: DateLike, spot: Optional[float] = None) -> float:
        key = to_datetime(expiry).date().isoformat()
        if spot is None and key in self.forward_overrides:
            return float(self.forward_overrides[key])
        return forward(
            self.spot if spot is None else float(spot),
            self.valuation_date,
            expiry,
            self.rate_curve,
            self.borrow_curve,
        )


__all__ = ["MarketState"]
