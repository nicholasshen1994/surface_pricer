"""Raw market-data providers.

Everything provider specific - contract parsing, exchange routing, wildcard
requests - stays in this module so that a new data source can be plugged in
behind :class:`MarketDataProvider` without touching the fit logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from ..core.curves import ConstantRateCurve
from ..core.daycount import BusinessCalendar, DateHelperBusinessCalendar, to_datetime
from .data import OptionQuoteRecord, RawSnapshot, SpotRecord
from .gateway import QuoteGatewaySnapshotClient, SnapshotRecord
from .listed_contracts import (
    EXCHANGE_CFFEX,
    parse_etf_option_code,
    parse_index_option_ticker,
    underlying_future_ticker,
)
from .option_contracts import FILE_NAME as CONTRACT_FILE_NAME
from .option_contracts import spec_for_record
from .registry import UnderlyingSpec, get_underlying_spec

DEFAULT_INDEX_RATE = 0.015


class MarketDataProvider(ABC):
    """Loads one raw market snapshot for an underlying."""

    @abstractmethod
    def load(self, underlying: str, as_of: Optional[datetime] = None) -> RawSnapshot:
        raise NotImplementedError


class QuoteApiDataProvider(MarketDataProvider):
    """Provider backed by the bundled Zhixing/CICC QuoteApi gateway.

    Index underlyings (MO/IO/HO) request the whole CFFEX option chain plus the
    matching futures; ETF underlyings request the exchange option category and
    filter by the ETF code prefix.  Futures are optional for ETFs because the
    forward is implied from put-call parity.
    """

    def __init__(
        self,
        client: QuoteGatewaySnapshotClient,
        *,
        rate: float = DEFAULT_INDEX_RATE,
        calendar: Optional[BusinessCalendar] = None,
        trading_days_per_year: float = 243.0,
        holiday_weight: float = 0.05,
        spec_overrides: Optional[Dict[str, UnderlyingSpec]] = None,
        rate_curve: Any = None,
        borrow_curve: Any = None,
    ):
        self._client = client
        self._rate = float(rate)
        self._rate_curve = rate_curve
        self._borrow_curve = borrow_curve
        self._calendar = calendar if calendar is not None else DateHelperBusinessCalendar("SHX")
        self._trading_days_per_year = float(trading_days_per_year)
        self._holiday_weight = float(holiday_weight)
        self._spec_overrides = dict(spec_overrides or {})

    def spec_for(self, underlying: str) -> UnderlyingSpec:
        key = str(underlying or "").strip().upper()
        if key in self._spec_overrides:
            return self._spec_overrides[key]
        return get_underlying_spec(underlying)

    def load(self, underlying: str, as_of: Optional[datetime] = None) -> RawSnapshot:
        spec = self.spec_for(underlying)
        if spec.kind == "index":
            snapshot = self._load_index_underlying(spec)
        else:
            snapshot = self._load_etf_underlying(spec)
        if as_of is not None:
            snapshot.valuation_datetime = to_datetime(as_of)
        snapshot.rate_curve = (
            self._rate_curve
            if self._rate_curve is not None
            else ConstantRateCurve(self._rate, anchor=snapshot.valuation_datetime)
        )
        snapshot.borrow_curve = self._borrow_curve
        snapshot.calendar = self._calendar
        snapshot.trading_days_per_year = self._trading_days_per_year
        snapshot.holiday_weight = self._holiday_weight
        return snapshot

    # ------------------------------------------------------------------ index
    def _load_index_underlying(self, spec: UnderlyingSpec) -> RawSnapshot:
        raw_options = self._client.get_option_snapshots(
            underlying_prefix=spec.underlying,
            exch_id=spec.exchange_id,
        )
        option_records, dropped_options = _option_records_from_cffex(raw_options, self._calendar)

        future_tickers = []
        seen = set()
        for record in option_records:
            ticker = underlying_future_ticker(
                spec.underlying,
                "{:04d}{:02d}".format(record.expiry.year, record.expiry.month),
            )
            if ticker not in seen:
                seen.add(ticker)
                future_tickers.append(ticker)
        future_records = [
            item for item in self._client.get_snapshots(future_tickers) if item is not None
        ]
        future_price_by_expiry = _future_price_by_expiry(
            option_records, future_records, spec.underlying
        )
        spot_records = [
            SpotRecord(
                ticker=item.ticker or item.resp_stk_code,
                price=_record_price(item),
                kind="future",
            )
            for item in future_records
        ]

        index_level = None
        if spec.index_ticker:
            index_record = self._client.get_index_snapshot(spec.index_ticker)
            if index_record is not None:
                index_level = _record_price(index_record)
                spot_records.append(
                    SpotRecord(
                        ticker=spec.index_ticker,
                        price=index_level,
                        kind="index",
                    )
                )

        spot = index_level
        if spot is None and future_price_by_expiry:
            spot = next(iter(sorted(future_price_by_expiry.items())))[1]
        if spot is None:
            raise RuntimeError(
                "No spot reference for {}: index and futures snapshots are both empty".format(
                    spec.underlying
                )
            )

        valuation = _valuation_datetime(raw_options) or _valuation_datetime(future_records)
        return RawSnapshot(
            underlying=spec.underlying,
            valuation_datetime=valuation if valuation is not None else datetime.today(),
            spot=float(spot),
            option_records=option_records,
            spot_records=spot_records,
            future_price_by_expiry=future_price_by_expiry,
            diagnostics={
                "option_records_raw": len(raw_options),
                "option_records_parsed": len(option_records),
                "option_records_dropped": dropped_options,
                "future_records": len(future_records),
                "index_level": index_level,
            },
        )

    # -------------------------------------------------------------------- etf
    def _load_etf_underlying(self, spec: UnderlyingSpec) -> RawSnapshot:
        raw_options = self._client.get_category_snapshots("O", exch_id=spec.exchange_id)
        prefix = str(spec.option_prefix or spec.underlying)
        option_records, dropped_options = _option_records_from_etf(
            raw_options, prefix, spec.etf_exchange, self._calendar
        )

        spot = None
        spot_records: List[SpotRecord] = []
        if spec.spot_ticker:
            spot_record = self._client.get_snapshots([spec.spot_ticker])
            spot_record = spot_record[0] if spot_record else None
            if spot_record is not None:
                spot = _record_price(spot_record)
                spot_records.append(
                    SpotRecord(ticker=spec.spot_ticker, price=spot, kind="spot")
                )
        if spot is None:
            raise RuntimeError("No ETF spot snapshot for {}".format(spec.underlying))
        if not option_records and raw_options:
            # The ETF option feed is keyed by the exchange's **numeric contract id**
            # (``10012493.SH`` / ``90008063.SZ``), not by the human-readable code
            # (``510500C2610M00600``) the parser needs to read strike / expiry /
            # option type out of - so a whole chain comes back as "no quotes".
            # Say that instead of letting the fit fail later with "no forwards".
            raise RuntimeError(
                "{}: {} option snapshot(s) came back, none readable - the gateway "
                "keys ETF options by the exchange's numeric contract id and "
                "data/{} has no entry for this chain.  Run 'python -m "
                "surface_pricer fetch-contracts --underlying {}' first.".format(
                    spec.underlying,
                    len(raw_options),
                    CONTRACT_FILE_NAME,
                    spec.underlying,
                )
            )

        valuation = _valuation_datetime(raw_options)
        return RawSnapshot(
            underlying=spec.underlying,
            valuation_datetime=valuation if valuation is not None else datetime.today(),
            spot=float(spot),
            option_records=option_records,
            spot_records=spot_records,
            future_price_by_expiry={},
            diagnostics={
                "option_records_raw": len(raw_options),
                "option_records_parsed": len(option_records),
                "option_records_dropped": dropped_options,
            },
        )


# --------------------------------------------------------------- conversions
def _option_records_from_cffex(
    raw_options: Sequence[SnapshotRecord],
    calendar: Optional[BusinessCalendar],
):
    records: List[OptionQuoteRecord] = []
    dropped = 0
    for item in raw_options:
        # One mechanism for both families (2026-10): the contract file knows the
        # strike / expiry / kind of every contract, CFFEX included.  A miss falls
        # back to parsing the code, which is what this used to do for everything.
        parsed = spec_for_record(item)
        if parsed is None:
            parsed = parse_index_option_ticker(
                item.resp_stk_code or item.ticker, calendar=calendar
            )
        if parsed is None:
            dropped += 1
            continue
        bid = item.best_bid
        ask = item.best_ask
        if bid is None or ask is None:
            dropped += 1
            continue
        records.append(
            OptionQuoteRecord(
                underlying=parsed.underlying,
                expiry=parsed.expiry,
                strike=float(parsed.strike),
                option_type=parsed.option_type,
                bid=float(bid),
                ask=float(ask),
                last=float(item.last or 0.0),
                volume=float(item.volume or 0.0),
                open_interest=float(item.open_interest or 0.0),
                raw_code=parsed.raw_code,
                exchange=EXCHANGE_CFFEX,
            )
        )
    return records, dropped


def _option_records_from_etf(
    raw_options: Sequence[SnapshotRecord],
    prefix: str,
    exchange: str,
    calendar: Optional[BusinessCalendar],
    ):
    records: List[OptionQuoteRecord] = []
    dropped = 0
    for item in raw_options:
        # The contract file first: the feed keys an ETF chain by the exchange's
        # numeric contract id (``10012493``), which no code parser can read - the
        # file is what turns it into a strike / expiry / kind.  Without an entry,
        # the code is parsed as before (a feed that does carry the code still works).
        parsed = spec_for_record(item)
        if parsed is None:
            code = str(item.resp_stk_code or "").strip().upper()
            if not code.startswith(prefix):
                continue
            parsed = parse_etf_option_code(code, calendar=calendar, exchange=exchange)
        if parsed is None:
            dropped += 1
            continue
        bid = item.best_bid
        ask = item.best_ask
        if bid is None or ask is None:
            dropped += 1
            continue
        records.append(
            OptionQuoteRecord(
                underlying=parsed.underlying,
                expiry=parsed.expiry,
                strike=float(parsed.strike),
                option_type=parsed.option_type,
                bid=float(bid),
                ask=float(ask),
                last=float(item.last or 0.0),
                volume=float(item.volume or 0.0),
                open_interest=float(item.open_interest or 0.0),
                raw_code=parsed.raw_code,
                exchange=exchange,
            )
        )
    return records, dropped


def _record_price(record: SnapshotRecord) -> float:
    if record.last > 0.0:
        return float(record.last)
    bid = record.best_bid
    ask = record.best_ask
    if bid is not None and ask is not None and ask >= bid:
        return float(0.5 * (bid + ask))
    if record.pre_close > 0.0:
        return float(record.pre_close)
    return 0.0


def _future_price_by_expiry(
    option_records: Sequence[OptionQuoteRecord],
    future_records: Sequence[SnapshotRecord],
    underlying: str,
) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for record in future_records:
        price = _record_price(record)
        if price > 0.0:
            prices[str(record.resp_stk_code or "").strip().upper()] = price
    result: Dict[str, float] = {}
    for record in option_records:
        ticker = underlying_future_ticker(
            underlying, "{:04d}{:02d}".format(record.expiry.year, record.expiry.month)
        )
        code = ticker.rsplit(".", 1)[0].upper()
        price = prices.get(code) or prices.get(ticker.upper())
        if price is not None and price > 0.0:
            result[record.expiry.date().isoformat()] = float(price)
    return result


def _valuation_datetime(records: Sequence[SnapshotRecord]) -> Optional[datetime]:
    trading_day = max((int(item.trading_day or 0) for item in records), default=0)
    if trading_day <= 0:
        return None
    text = str(trading_day)
    latest = max((int(item.time or 0) for item in records), default=0)
    hour = latest // 10_000_000
    minute = (latest // 100_000) % 100
    second = (latest // 1_000) % 100
    if 0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60:
        return datetime(
            int(text[:4]), int(text[4:6]), int(text[6:8]), hour, minute, second
        )
    return datetime(int(text[:4]), int(text[4:6]), int(text[6:8]))


__all__ = [
    "DEFAULT_INDEX_RATE",
    "MarketDataProvider",
    "QuoteApiDataProvider",
]
