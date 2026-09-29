"""Borrow curve implied from futures and listed-option put/call parity.

Follows edslib's ``CNBorrowRateFitter`` / ``BorrowFittingUtils``:

* per expiry the forward comes from the CFFEX future; where a future is missing
  the put/call parity of the listed options is used
  (``F = (C - P) / DF + K`` at the strike closest to the money);
* the implied borrow is ``q = f(0, T) - ln(F / S) / dcf(0, T)`` where ``f`` is
  the continuously compounded funding rate of the discount curve
  (``DF(T) = exp(-f * dcf)``, the same relation used when pricing forwards);
* the tail beyond the last observable expiry is extended with the OU forward
  model of ``apps/borrow_fitter/borrow_rate_forward_curve.py``: quarterly
  (third-Friday) pillars up to ``extension_years`` (3Y for this desk).

Everything here is plain numpy: the discount curve enters as a
:class:`surface_pricer.core.curves.PiecewiseRateCurve` (or any object with
``discount_factor``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .curves import PiecewiseRateCurve
from .daycount import BusinessCalendar, DateLike, add_tenor, to_date, to_datetime, year_fraction

# edslib defaults (``apps/borrow_fitter/borrow_rate_forward_curve.py``).
DEFAULT_MIN_DAYS_TO_EXPIRY = 3
DEFAULT_EXTENSION_YEARS = 3
DEFAULT_OU_KAPPA_PRIOR = 1.0
DEFAULT_F0_BAND = 0.02
DEFAULT_MIN_F0_DT_DAYS = 20
DEFAULT_BASIS = "act/365f"

_QUARTER_MONTHS = (3, 6, 9, 12)


@dataclass
class BorrowCurvePillars:
    """Implied borrow pillars (decimals), observed part plus OU tail.

    When ``cum_div_factors`` are supplied the rates are **pure borrow**: the
    dividend part of the futures basis has been removed
    (``q_pure = q_total + ln(D) / dcf``, edslib's ``cum_yield_ratio``
    convention).  Pricing must then multiply the dividend factor back on the
    forward side (``F = S * exp((r - q_pure) * t) * D``); that wiring is not
    in place yet, so the default (no factors) keeps ``q`` inclusive of
    dividends and the forward side untouched.
    """

    valuation_date: date
    pillar_dates: List[date]
    rates: List[float]
    day_counter: str = DEFAULT_BASIS
    spot: float = 0.0
    forwards: Dict[str, float] = field(default_factory=dict)
    forward_source: Dict[str, str] = field(default_factory=dict)
    cum_div_factors: Dict[str, float] = field(default_factory=dict)
    observed: int = 0
    extended: int = 0
    # mixed values on purpose: kappa/mu/f0/calibrated are floats, anchor_date a
    # string - see :func:`extend_borrow_tail`.
    ou: Dict[str, Any] = field(default_factory=dict)
    calendar_name: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def observed_pillars(self) -> List[Tuple[date, float]]:
        return list(zip(self.pillar_dates[: self.observed], self.rates[: self.observed]))

    def to_piecewise_curve(self) -> PiecewiseRateCurve:
        days = [(value - self.valuation_date).days for value in self.pillar_dates]
        return PiecewiseRateCurve(
            anchor=self.valuation_date,
            tenors=[float(day) for day in days],
            rates=self.rates,
            interpolation="linear_zero",
            basis=self.day_counter,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "valuation_date": self.valuation_date.isoformat(),
            "day_counter": self.day_counter,
            "spot": self.spot,
            "pillar_dates": [value.isoformat() for value in self.pillar_dates],
            "pillar_days": [(value - self.valuation_date).days for value in self.pillar_dates],
            "rates": list(self.rates),
            "forwards": dict(self.forwards),
            "forward_source": dict(self.forward_source),
            "cum_div_factors": dict(self.cum_div_factors),
            "observed": self.observed,
            "extended": self.extended,
            "ou": dict(self.ou),
            "calendar": self.calendar_name,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BorrowCurvePillars":
        return cls(
            valuation_date=to_date(str(value["valuation_date"])),
            pillar_dates=[to_date(str(item)) for item in list(value["pillar_dates"])],
            rates=[float(item) for item in list(value["rates"])],
            day_counter=str(value.get("day_counter", DEFAULT_BASIS)),
            spot=float(value.get("spot", 0.0)),
            forwards={str(k): float(v) for k, v in dict(value.get("forwards") or {}).items()},
            forward_source={
                str(k): str(v) for k, v in dict(value.get("forward_source") or {}).items()
            },
            cum_div_factors={
                str(k): float(v) for k, v in dict(value.get("cum_div_factors") or {}).items()
            },
            observed=int(value.get("observed", 0)),
            extended=int(value.get("extended", 0)),
            ou={str(k): v for k, v in dict(value.get("ou") or {}).items()},
            calendar_name=str(value.get("calendar", "")),
            notes=[str(item) for item in list(value.get("notes") or [])],
        )

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BorrowCurvePillars":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ observed
def build_borrow_curve(
    valuation_date: DateLike,
    spot: float,
    forwards: Mapping[DateLike, float],
    rate_curve,
    *,
    basis: str = DEFAULT_BASIS,
    min_days_to_expiry: int = DEFAULT_MIN_DAYS_TO_EXPIRY,
    forward_source: Optional[Mapping[DateLike, str]] = None,
    cum_div_factors: Optional[Mapping[DateLike, float]] = None,
    calendar_name: str = "",
) -> BorrowCurvePillars:
    """Implied borrow pillars from forward prices and a discount curve.

    ``cum_div_factors`` optionally maps an expiry to the cumulative dividend
    factor ``D`` up to that date (edslib's ``cum_yield_ratio``).  With a factor
    the result is **pure borrow**::

        q_pure = r - ln(F / (S * D)) / dcf = q_total + ln(D) / dcf

    and pricing must multiply ``D`` back on the forward side
    (``F = S * exp((r - q_pure) * t) * D``).  Without any factor - the default,
    since no dividend source is wired yet - ``q`` stays inclusive of dividends
    and the forward side is untouched, which keeps the implied forward equal to
    the market future either way.
    """
    if spot is None or float(spot) <= 0.0:
        raise ValueError("spot must be positive to imply borrow rates")
    val_date = to_date(valuation_date)
    spot = float(spot)

    pillar_dates: List[date] = []
    rates: List[float] = []
    kept_forwards: Dict[str, float] = {}
    kept_source: Dict[str, str] = {}
    kept_div: Dict[str, float] = {}
    notes: List[str] = []
    source_map = {to_date(key): str(value) for key, value in dict(forward_source or {}).items()}
    divisor_map = {
        to_date(key): float(value) for key, value in dict(cum_div_factors or {}).items()
    }
    if divisor_map:
        notes.append(
            "pure borrow: {} cumulative dividend factor(s) applied".format(len(divisor_map))
        )

    for key, forward in sorted(forwards.items(), key=lambda item: to_date(item[0])):
        expiry = to_date(key)
        days = (expiry - val_date).days
        if days < int(min_days_to_expiry):
            notes.append("{} skipped: {} day(s) to expiry".format(expiry.isoformat(), days))
            continue
        if forward is None or float(forward) <= 0.0:
            notes.append("{} skipped: non-positive forward".format(expiry.isoformat()))
            continue
        dcf = year_fraction(val_date, expiry, basis=basis)
        if dcf <= 0.0:
            notes.append("{} skipped: non-positive dcf".format(expiry.isoformat()))
            continue
        dividend = divisor_map.get(expiry, 1.0)
        if dividend <= 0.0:
            notes.append(
                "{} skipped: non-positive cum dividend factor".format(expiry.isoformat())
            )
            continue
        discount = float(rate_curve.discount_factor(val_date, expiry))
        funding = -math.log(discount) / dcf
        carry = math.log(float(forward) / dividend / spot) / dcf
        pillar_dates.append(expiry)
        rates.append(funding - carry)
        kept_forwards[expiry.isoformat()] = float(forward)
        kept_source[expiry.isoformat()] = source_map.get(expiry, "forward")
        if abs(dividend - 1.0) > 1.0e-12:
            kept_div[expiry.isoformat()] = dividend

    if not pillar_dates:
        raise ValueError("no usable forward expiries to imply borrow rates")

    return BorrowCurvePillars(
        valuation_date=val_date,
        pillar_dates=pillar_dates,
        rates=rates,
        day_counter=basis,
        spot=spot,
        forwards=kept_forwards,
        forward_source=kept_source,
        cum_div_factors=kept_div,
        observed=len(pillar_dates),
        calendar_name=calendar_name,
        notes=notes,
    )


# ------------------------------------------------------------------ OU tail
def integrate_ou_forward(f0: float, kappa: float, mu: float, horizon: float) -> float:
    """``Integral[0, u] (mu + (f0 - mu) * exp(-kappa * s)) ds`` (edslib)."""
    if not all(np.isfinite(value) for value in (f0, kappa, mu, horizon)) or kappa <= 0 or horizon < 0:
        return float("nan")
    return float(mu * horizon + (f0 - mu) * (-np.expm1(-kappa * horizon)) / kappa)


def instantaneous_f0(T_anchor: float, q_anchor: float, T_prev: float, q_prev: float) -> float:
    """Instantaneous forward at the anchor implied by the last two pillars."""
    if not all(np.isfinite(value) for value in (T_anchor, q_anchor, T_prev, q_prev)):
        return float("nan")
    if T_anchor <= T_prev:
        return float(q_anchor)
    return float((q_anchor * T_anchor - q_prev * T_prev) / (T_anchor - T_prev))


def zero_rate_at(
    q_anchor: float,
    T_anchor: float,
    f0: float,
    kappa: float,
    mu: float,
    T_target: float,
) -> float:
    """``z(T) = (q_anchor * T_anchor + integral(T_anchor -> T)) / T`` (edslib)."""
    if T_target <= T_anchor:
        return float(q_anchor)
    integral = integrate_ou_forward(f0, kappa, mu, T_target - T_anchor)
    return float((q_anchor * T_anchor + integral) / T_target)


def next_quarter_expiry(after: DateLike, calendar: Optional[BusinessCalendar] = None) -> date:
    """Next CFFEX expiry (third Friday, rolled forward off holidays) after ``after``."""
    current = to_date(after)
    year, month = current.year, current.month
    month = next((item for item in _QUARTER_MONTHS if item >= month), 3)
    if month < current.month:
        year += 1
    expiry = _third_friday(year, month, calendar)
    if expiry <= current:
        index = (_QUARTER_MONTHS.index(month) + 1) % len(_QUARTER_MONTHS)
        month = _QUARTER_MONTHS[index]
        if month == 3:
            year += 1
        expiry = _third_friday(year, month, calendar)
    return expiry


def _third_friday(year: int, month: int, calendar: Optional[BusinessCalendar]) -> date:
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    if calendar is not None:
        while not calendar.is_business_day(third_friday):
            third_friday += timedelta(days=1)
    return third_friday


def extend_borrow_tail(
    pillars: BorrowCurvePillars,
    *,
    extension_years: int = DEFAULT_EXTENSION_YEARS,
    kappa: Optional[float] = None,
    mu: Optional[float] = None,
    f0: Optional[float] = None,
    calendar: Optional[BusinessCalendar] = None,
    f0_band: float = DEFAULT_F0_BAND,
    min_f0_dt_days: int = DEFAULT_MIN_F0_DT_DAYS,
) -> BorrowCurvePillars:
    """Append OU-extrapolated quarterly pillars up to ``valuation + N years``.

    Mirrors ``OUBorrowCalibrator.extend_curve``.  Without a calibrated
    ``kappa`` (no history available) the edslib log-kappa prior center
    (``kappa = 1.0``) is used and the fact is recorded in ``notes``.
    """
    observed = pillars.observed_pillars
    if len(observed) < 1:
        return pillars

    val_date = pillars.valuation_date
    basis = pillars.day_counter
    last_date, q_anchor = observed[-1]
    T_anchor = year_fraction(val_date, last_date, basis=basis)

    calibrated = kappa is not None
    kappa_value = float(kappa) if calibrated else DEFAULT_OU_KAPPA_PRIOR

    if f0 is None:
        if len(observed) >= 2:
            prev_date, q_prev = observed[-2]
            T_prev = year_fraction(val_date, prev_date, basis=basis)
            if (last_date - prev_date).days < int(min_f0_dt_days):
                f0 = float(q_anchor)
            else:
                f0 = instantaneous_f0(T_anchor, q_anchor, T_prev, q_prev)
        else:
            f0 = float(q_anchor)
    f0 = float(np.clip(f0, q_anchor - f0_band, q_anchor + f0_band))

    mu_value = float(np.mean([rate for _, rate in observed])) if mu is None else float(mu)

    end_date = add_tenor(val_date, "{}Y".format(int(extension_years)))
    expiry = next_quarter_expiry(last_date + timedelta(days=1), calendar)

    rates = list(pillars.rates)
    dates = list(pillars.pillar_dates)
    extended = 0
    while expiry <= end_date:
        T_target = year_fraction(val_date, expiry, basis=basis)
        rate = zero_rate_at(q_anchor, T_anchor, f0, kappa_value, mu_value, T_target)
        if math.isfinite(rate):
            dates.append(expiry)
            rates.append(rate)
            extended += 1
        expiry = next_quarter_expiry(expiry + timedelta(days=1), calendar)

    notes = list(pillars.notes)
    if extended == 0:
        notes.append("tail extension produced no pillar up to {}Y".format(extension_years))
    elif not calibrated:
        notes.append(
            "OU tail uses the edslib log-kappa prior {:.2f} (no historical calibration input)"
            .format(kappa_value)
        )

    return BorrowCurvePillars(
        valuation_date=pillars.valuation_date,
        pillar_dates=dates,
        rates=rates,
        day_counter=pillars.day_counter,
        spot=pillars.spot,
        forwards=dict(pillars.forwards),
        forward_source=dict(pillars.forward_source),
        cum_div_factors=dict(pillars.cum_div_factors),
        observed=pillars.observed,
        extended=extended,
        ou={
            "kappa": kappa_value,
            "mu": mu_value,
            "f0": f0,
            "anchor_date": last_date.isoformat(),
            "anchor_T": T_anchor,
            "calibrated": 1.0 if calibrated else 0.0,
        },
        calendar_name=getattr(calendar, "name", "") or pillars.calendar_name,
        notes=notes,
    )


__all__ = [
    "BorrowCurvePillars",
    "DEFAULT_EXTENSION_YEARS",
    "build_borrow_curve",
    "extend_borrow_tail",
    "instantaneous_f0",
    "integrate_ou_forward",
    "next_quarter_expiry",
    "zero_rate_at",
]
