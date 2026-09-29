"""Build option-expiry to corresponding index-future snapshot mappings."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .gateway import SnapshotRecord
from .listed_contracts import (
    parse_index_option_ticker,
    underlying_future_ticker,
)


def future_tickers_for_option_records(
    records: Iterable[SnapshotRecord],
    underlying_prefix: str,
) -> List[str]:
    """Return unique futures needed by an option snapshot."""
    result = []
    seen = set()
    prefix = str(underlying_prefix or "").strip().upper()
    for record in records:
        parsed = parse_index_option_ticker(record.resp_stk_code or record.ticker)
        if parsed is None or parsed.underlying != prefix:
            continue
        ticker = underlying_future_ticker(prefix, parsed.contract_month)
        if ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
    return sorted(result)


def future_spots_by_expiry(
    option_records: Iterable[SnapshotRecord],
    future_records: Iterable[SnapshotRecord],
    underlying_prefix: str,
) -> Dict[str, float]:
    """Map each option expiry date to its matching future last price."""
    future_prices = {}
    for record in future_records:
        code = str(record.resp_stk_code or "").strip().upper()
        price = _future_mark(record)
        if price is not None:
            future_prices[code] = price

    result = {}
    prefix = str(underlying_prefix or "").strip().upper()
    for record in option_records:
        parsed = parse_index_option_ticker(record.resp_stk_code or record.ticker)
        if parsed is None or parsed.underlying != prefix:
            continue
        future_ticker = underlying_future_ticker(prefix, parsed.contract_month)
        future_code = future_ticker.rsplit(".", 1)[0]
        price = future_prices.get(future_code)
        if price is not None:
            result[parsed.expiry.date().isoformat()] = price
    return result


def _future_mark(record: SnapshotRecord):
    if record.last > 0.0:
        return float(record.last)
    bid = record.best_bid
    ask = record.best_ask
    if bid is not None and ask is not None and ask >= bid:
        return 0.5 * (bid + ask)
    if record.pre_close > 0.0:
        return float(record.pre_close)
    return None


__all__ = [
    "future_spots_by_expiry",
    "future_tickers_for_option_records",
]
