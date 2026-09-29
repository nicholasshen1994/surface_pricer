"""Forward extraction from one market snapshot.

Mirrors edslib's ``CNBorrowRateFitter`` order of preference: a listed future
wins, and put/call parity fills the expiries where no future quote is
available::

    F = (C - P) / DF(T) + K

evaluated at the strike closest to the money (``|C - P|`` minimal), i.e. the
``min_margin_strike`` of ``apps/borrow_fitter/borrow_fitter.py``.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.daycount import DateLike, to_date
from .data import OptionQuoteRecord, RawSnapshot


def _mid_price(record: OptionQuoteRecord) -> Optional[float]:
    bid = float(getattr(record, "bid", 0.0) or 0.0)
    ask = float(getattr(record, "ask", 0.0) or 0.0)
    if bid > 0.0 and ask >= bid:
        return 0.5 * (bid + ask)
    last = float(getattr(record, "last", 0.0) or 0.0)
    if last > 0.0:
        return last
    return None


def parity_forward_by_expiry(
    option_records: Sequence[OptionQuoteRecord],
    rate_curve,
    valuation_date: DateLike,
) -> Dict[date, float]:
    """Synthetic forwards per expiry from the closest-to-money call/put pair."""
    val_date = to_date(valuation_date)
    pairs: Dict[date, Dict[float, Dict[str, float]]] = {}
    for record in option_records:
        price = _mid_price(record)
        if price is None:
            continue
        expiry = to_date(record.expiry)
        if expiry <= val_date:
            continue
        option_type = str(record.option_type or "").strip().lower()
        side = "call" if option_type in {"c", "call"} else "put" if option_type in {"p", "put"} else None
        if side is None:
            continue
        pairs.setdefault(expiry, {}).setdefault(float(record.strike), {})[side] = price

    result: Dict[date, float] = {}
    for expiry, strikes in pairs.items():
        best: Optional[Tuple[float, float, float, float]] = None
        for strike, sides in strikes.items():
            if "call" not in sides or "put" not in sides:
                continue
            call, put = sides["call"], sides["put"]
            gap = abs(call - put)
            if best is None or gap < best[0]:
                best = (gap, strike, call, put)
        if best is None:
            continue
        _, strike, call, put = best
        discount = float(rate_curve.discount_factor(val_date, expiry)) if rate_curve is not None else 1.0
        if discount <= 0.0:
            continue
        result[expiry] = float((call - put) / discount + strike)
    return result


def forwards_from_snapshot(
    snapshot: RawSnapshot,
    rate_curve,
) -> Tuple[Dict[date, float], Dict[date, str]]:
    """``({expiry: forward}, {expiry: 'future'|'parity'})`` for one snapshot."""
    forwards: Dict[date, float] = {}
    sources: Dict[date, str] = {}

    for key, price in dict(snapshot.future_price_by_expiry or {}).items():
        value = float(price)
        if value <= 0.0:
            continue
        expiry = to_date(key)
        forwards[expiry] = value
        sources[expiry] = "future"

    parity = parity_forward_by_expiry(
        snapshot.option_records,
        rate_curve,
        snapshot.valuation_datetime.date(),
    )
    for expiry, forward in parity.items():
        if expiry in forwards:
            continue
        forwards[expiry] = float(forward)
        sources[expiry] = "parity"
    return forwards, sources


__all__ = ["forwards_from_snapshot", "parity_forward_by_expiry"]
