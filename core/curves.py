"""Simple continuous zero-rate curves for a single-currency market."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from scipy.interpolate import CubicSpline

from .daycount import BusinessCalendar, DateLike, to_date, to_datetime, year_fraction


def _tenor_to_days(value: Union[str, int, float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().upper()
    if text.endswith("D"):
        return float(text[:-1])
    if text.endswith("W"):
        return float(text[:-1]) * 7.0
    if text.endswith("M"):
        return float(text[:-1]) * 30.0
    if text.endswith("Y"):
        return float(text[:-1]) * 365.0
    return float(text)


@dataclass
class ConstantRateCurve:
    rate: float = 0.0
    anchor: Optional[DateLike] = None
    basis: str = "act/365f"

    def __post_init__(self):
        self.anchor = to_datetime(self.anchor) if self.anchor is not None else None

    def zero_rate(self, when: DateLike) -> float:
        return float(self.rate)

    def discount_factor(self, start: DateLike, end: DateLike) -> float:
        tau = year_fraction(start, end, basis=self.basis)
        return float(np.exp(-self.rate * tau))

    def bump_pillar(self, _pillar: DateLike, amount: float) -> "ConstantRateCurve":
        bumped = deepcopy(self)
        bumped.rate += amount
        return bumped

    def rebuild(
        self,
        pillar_dates: Iterable[DateLike],
        anchor: Optional[DateLike] = None,
    ) -> "PiecewiseRateCurve":
        """Spread the constant rate over a pillar grid.

        Mirrors edslib's ``rebuild_ql_curve_by_tenors``: a flat curve has no
        bumpable pillar, so the bucketed rate / borrow greeks first expand it
        onto the bucket grid (same rate at every pillar) and then bump one
        pillar at a time - without this step ``bump_pillar`` would shift the
        whole curve and every bucket would carry the same parallel sensitivity.
        """
        base = to_datetime(anchor) if anchor is not None else self.anchor
        if base is None:
            raise ValueError("a constant curve needs an anchor to be rebuilt")
        days = sorted({(to_date(value) - base.date()).days for value in pillar_dates})
        days = [value for value in days if value > 0]
        if not days:
            raise ValueError("no pillar date after {}".format(base.date()))
        return PiecewiseRateCurve(
            anchor=base,
            tenors=[float(value) for value in days],
            rates=[self.rate] * len(days),
            basis=self.basis,
        )

    @property
    def pillar_dates(self):
        return []


class PiecewiseRateCurve:
    """Piecewise zero-rate curve with optional cubic-zero interpolation."""

    def __init__(
        self,
        anchor: DateLike,
        tenors: Sequence[Union[str, int, float]],
        rates: Sequence[float],
        calendar: Optional[BusinessCalendar] = None,
        trading_days_per_year: Optional[float] = None,
        interpolation: str = "linear_zero",
        basis: str = "act/365f",
    ):
        if len(tenors) != len(rates) or not tenors:
            raise ValueError("tenors and rates must be non-empty and have equal length")
        self.anchor = to_datetime(anchor)
        self.tenors = tuple(tenors)
        self.rates = np.asarray(rates, dtype=float).copy()
        self.calendar = calendar
        self.trading_days_per_year = trading_days_per_year
        self.interpolation = interpolation.lower()
        self.basis = basis
        self._tenor_days = np.asarray([_tenor_to_days(x) for x in tenors], dtype=float)
        if np.any(np.diff(self._tenor_days) <= 0):
            raise ValueError("curve tenors must be strictly increasing")

    @property
    def pillar_dates(self):
        return [self.anchor + timedelta(days=int(days)) for days in self._tenor_days]

    def _time(self, when: DateLike) -> float:
        return year_fraction(
            self.anchor,
            when,
            calendar=self.calendar,
            trading_days_per_year=self.trading_days_per_year,
            basis=self.basis,
        )

    def _pillar_times(self) -> np.ndarray:
        return np.asarray(
            [
                year_fraction(
                    self.anchor,
                    self.anchor + timedelta(days=int(days)),
                    calendar=self.calendar,
                    trading_days_per_year=self.trading_days_per_year,
                    basis=self.basis,
                )
                for days in self._tenor_days
            ],
            dtype=float,
        )

    def zero_rate(self, when: DateLike) -> float:
        target = self._time(when)
        pillar_times = self._pillar_times()
        if len(pillar_times) == 1:
            return float(self.rates[0])
        if self.interpolation in {"cubic_zero", "cubic"} and len(pillar_times) >= 3:
            spline = CubicSpline(pillar_times, self.rates, extrapolate=True)
            return float(spline(target))
        return float(np.interp(target, pillar_times, self.rates))

    def discount_factor(self, start: DateLike, end: DateLike) -> float:
        # Discount off the anchor whenever ``start`` falls on the anchor *date*:
        # valuation is timestamped mid-day while curves are dated, and edslib's
        # dcf runs on dates.  Keeping an intra-day stub would add a spurious
        # sensitivity to the first pillar (it shows up as a non-zero near bucket
        # in the bucketed greeks).
        if to_date(start) == to_date(self.anchor):
            tau = year_fraction(
                self.anchor,
                end,
                calendar=self.calendar,
                trading_days_per_year=self.trading_days_per_year,
                basis=self.basis,
            )
            if tau <= 0:
                return 1.0
            return float(np.exp(-self.zero_rate(end) * tau))

        tau = year_fraction(start, end, calendar=self.calendar, trading_days_per_year=self.trading_days_per_year, basis=self.basis)
        if tau <= 0:
            return 1.0
        start_rate = self.zero_rate(start)
        end_rate = self.zero_rate(end)
        return float(np.exp(-end_rate * tau + start_rate * max(0.0, self._time(start))))

    def bump_pillar(self, pillar: Union[int, DateLike], amount: float) -> "PiecewiseRateCurve":
        bumped = deepcopy(self)
        if isinstance(pillar, int):
            index = pillar
        else:
            target = to_datetime(pillar)
            index = int(np.argmin([abs(to_datetime(x) - target) for x in self.pillar_dates]))
        bumped.rates[index] += amount
        return bumped

    def with_rates(self, rates: Iterable[float]) -> "PiecewiseRateCurve":
        bumped = deepcopy(self)
        bumped.rates = np.asarray(list(rates), dtype=float)
        return bumped


def forward(
    spot: float,
    valuation_date: DateLike,
    expiry: DateLike,
    rate_curve,
    borrow_curve=None,
) -> float:
    rate_df = rate_curve.discount_factor(valuation_date, expiry) if rate_curve is not None else 1.0
    borrow_df = borrow_curve.discount_factor(valuation_date, expiry) if borrow_curve is not None else 1.0
    return float(spot * borrow_df / rate_df)
