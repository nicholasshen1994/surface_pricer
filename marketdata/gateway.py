"""Snapshot access for the bundled Zhixing/CICC QuoteApi gateway.

The vendor SDK returns protobuf frames from ``getSnapshot`` while its simpler
``QuerySnapshot`` API returns a Python quote object. This module keeps the
gateway-specific details here and exposes plain ``SnapshotRecord`` instances
to the standalone surface-pricer pipeline.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# The bundled QuoteApi SDK requires the pure-Python protobuf runtime path.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_VENDOR_API: Optional[Any] = None
_VENDOR_PROTO: Optional[Any] = None
_VENDOR_IMPORT_ERROR: Optional[BaseException] = None
_VENDOR_THREAD_HOOK_INSTALLED = False


def _vendor_candidates() -> List[Path]:
    """Directories that may hold the Zhixing SDK (the ``QuoteApiLib`` package).

    The SDK ships in ``<repo root>/zhixing``; ``surface_pricer/zhixing`` stays a
    candidate so a self-contained copy of the package keeps working.  The repo
    root is found by walking up from this file (``.../surface_pricer/marketdata``).
    """
    here = Path(__file__).resolve()
    return [here.parents[1] / "zhixing", here.parents[2] / "zhixing"]


def _add_vendor_sys_path() -> Optional[Path]:
    """Put the first real Zhixing SDK directory on ``sys.path``.

    Only a directory that actually contains ``QuoteApiLib`` is added, so a
    missing folder cannot shadow an SDK that is already importable.
    """
    for candidate in _vendor_candidates():
        if (candidate / "QuoteApiLib").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


def _load_vendor_bindings() -> Tuple[Any, Any]:
    global _VENDOR_API, _VENDOR_PROTO, _VENDOR_IMPORT_ERROR
    if _VENDOR_API is not None and _VENDOR_PROTO is not None:
        return _VENDOR_API, _VENDOR_PROTO
    if _VENDOR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "QuoteApiLib bindings are unavailable. Check the bundled "
            "zhixing/QuoteApiLib directory and the Python runtime."
        ) from _VENDOR_IMPORT_ERROR

    _add_vendor_sys_path()

    try:
        import zmq

        # The vendor binary historically refers to this misspelled alias.
        if hasattr(zmq, "Again") and not hasattr(zmq, "Aagin"):
            zmq.Aagin = zmq.Again
        _install_vendor_thread_exception_filter()
        import QuoteApiLib.quoteApi as api
        from QuoteApiLib.proto import BeanProtoQuotation_pb2 as proto
    except Exception as exc:
        _VENDOR_IMPORT_ERROR = exc
        searched = ", ".join(str(path) for path in _vendor_candidates())
        raise RuntimeError(
            "Failed to import QuoteApiLib bindings. The Zhixing SDK must live "
            "under {}; its compiled extension is built for CPython 3.8 x64, so "
            "run the project on a 3.8 interpreter and check the protobuf "
            "installation.".format(searched)
        ) from exc

    _VENDOR_API = api
    _VENDOR_PROTO = proto
    return api, proto


def _install_vendor_thread_exception_filter() -> None:
    """Hide the vendor's expected ZeroMQ shutdown exception.

    The compiled SDK terminates its socket context from ``Stop/UnInit`` before
    its monitor thread exits. On Python 3.8+ that produces an uncaught
    ``zmq.ContextTerminated`` report even though shutdown succeeded.
    """
    global _VENDOR_THREAD_HOOK_INSTALLED
    if _VENDOR_THREAD_HOOK_INSTALLED:
        return
    hook = getattr(threading, "excepthook", None)
    if hook is None:
        return

    def filtered_hook(args: Any) -> None:
        exc = getattr(args, "exc_value", None)
        module = getattr(exc.__class__, "__module__", "") if exc else ""
        name = getattr(exc.__class__, "__name__", "") if exc else ""
        if name == "ContextTerminated" and module.startswith("zmq."):
            return
        hook(args)

    threading.excepthook = filtered_hook
    _VENDOR_THREAD_HOOK_INSTALLED = True


from .mapping import (
    REQUEST_USAGE_SNAPSHOT,
    normalize_quote_response_stk_code,
    normalize_quote_ticker,
    resolve_index_request_fields,
    resolve_quote_request_fields,
)


@dataclass(frozen=True)
class SnapshotRequestKey:
    ticker: str
    category: str
    exch_id: str
    stk_code: str


@dataclass
class SnapshotRecord:
    ticker: str
    category: str
    request_exch_id: str
    request_stk_code: str
    resp_exch_id: str
    resp_stk_code: str
    trading_day: int
    time: int
    status: int
    last: float
    pre_close: float
    td_close: float
    open: float
    high: float
    low: float
    volume: int
    turnover: float
    num_trades: int
    local_time: int = 0
    ask_prices: Tuple[float, ...] = ()
    ask_volumes: Tuple[int, ...] = ()
    bid_prices: Tuple[float, ...] = ()
    bid_volumes: Tuple[int, ...] = ()
    open_interest: int = 0
    pre_open_interest: int = 0
    pre_settle_price: float = 0.0
    settle_price: float = 0.0
    lot_size: int = 0
    trade_status: str = ""
    close_flag: str = ""

    @property
    def best_ask(self) -> Optional[float]:
        return _first_positive(self.ask_prices)

    @property
    def best_bid(self) -> Optional[float]:
        return _first_positive(self.bid_prices)

    @property
    def has_two_way_market(self) -> bool:
        bid = self.best_bid
        ask = self.best_ask
        return bid is not None and ask is not None and ask >= bid


class QuoteGatewaySnapshotClient:
    """Bottom-level QuoteApi ``getSnapshot`` wrapper.

    ``get_snapshots`` resolves explicit tickers. ``get_category_snapshots``
    supports a gateway wildcard request, which is how the complete CFFEX
    option chain is obtained:

    ``get_category_snapshots("O", exch_id="F")``
    """

    def __init__(self, host: str, port: int, user: str, password: str):
        self._host = str(host)
        self._port = int(port)
        self._user = str(user)
        self._password = str(password)
        self._api: Optional[Any] = None
        self._api_lock = threading.RLock()
        self._request_key_cache: Dict[str, Optional[SnapshotRequestKey]] = {}
        self._subscribed_tickers: Tuple[str, ...] = ()
        self._subscribed_keys: Tuple[Optional[SnapshotRequestKey], ...] = ()

    def start(self) -> None:
        with self._api_lock:
            if self._api is not None:
                return
            api, _ = _load_vendor_bindings()
            client = api.QuoteApi()
            init_result = client.Init(
                api.QuotationInit(self._host, self._port, self._user, self._password)
            )
            if not init_result:
                err = getattr(client, "lastError", None)
                raise RuntimeError(f"QuoteApi.Init failed: {err}")
            started = client.Start()
            if not started:
                err = getattr(client, "lastError", None)
                raise RuntimeError(f"QuoteApi.Start failed: {err}")
            self._api = client

    def close(self) -> None:
        with self._api_lock:
            if self._api is None:
                return
            try:
                self._api.Stop()
            except Exception:
                pass
            try:
                self._api.UnInit()
            except Exception:
                pass
            self._api = None

    def __enter__(self) -> "QuoteGatewaySnapshotClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def last_error(self) -> Optional[str]:
        with self._api_lock:
            if self._api is None:
                return None
            return getattr(self._api, "lastError", None)

    @property
    def subscribed_tickers(self) -> Tuple[str, ...]:
        with self._api_lock:
            return self._subscribed_tickers

    def get_snapshots(self, tickers: Sequence[str]) -> List[Optional[SnapshotRecord]]:
        """Fetch snapshots for explicit normalized market tickers."""
        self.start()
        normalized = tuple(normalize_quote_ticker(ticker) for ticker in tickers)
        resolved_keys = tuple(
            self._resolve_runtime_request_key(ticker) for ticker in normalized
        )
        return self._get_snapshots_from_keys(normalized, resolved_keys)

    def get_index_snapshots(
        self, tickers: Sequence[str]
    ) -> List[Optional[SnapshotRecord]]:
        """Fetch explicit index tickers using the gateway ``INDEX`` category."""
        self.start()
        normalized = tuple(normalize_quote_ticker(ticker) for ticker in tickers)
        keys = []
        for ticker in normalized:
            fields = resolve_index_request_fields(
                ticker, request_usage=REQUEST_USAGE_SNAPSHOT
            )
            if fields is None:
                keys.append(None)
                continue
            category, exch_id, stk_code = fields
            keys.append(
                SnapshotRequestKey(
                    ticker=ticker,
                    category=category,
                    exch_id=exch_id,
                    stk_code=stk_code,
                )
            )
        return self._get_snapshots_from_keys(normalized, tuple(keys))

    def get_index_snapshot(self, ticker: str) -> Optional[SnapshotRecord]:
        records = self.get_index_snapshots([ticker])
        return records[0] if records else None

    def get_category_snapshots(
        self,
        category: str,
        *,
        exch_id: str = "",
        stk_code: str = "",
        subcategory: str = "",
    ) -> List[SnapshotRecord]:
        """Fetch a category wildcard or condition-based snapshot list."""
        self.start()
        _, proto = _load_vendor_bindings()
        request = proto.StkSnapshotRequest()
        request.strCategory = str(category).strip().upper()
        if subcategory:
            request.strSubCategory = str(subcategory).strip().upper()
        if exch_id or stk_code:
            condition = request.condition.add()
            condition.exchId = str(exch_id).strip().upper()
            condition.stkCode = str(stk_code).strip().upper()
        return self._fetch_records(request, ())

    def get_option_snapshots(
        self,
        *,
        underlying_prefix: Optional[str] = None,
        exch_id: str = "F",
    ) -> List[SnapshotRecord]:
        """Fetch all CFFEX option snapshots and optionally keep one prefix."""
        records = self.get_category_snapshots("O", exch_id=exch_id)
        if not underlying_prefix:
            return records
        prefix = str(underlying_prefix).strip().upper()
        return [
            record
            for record in records
            if record.resp_stk_code.upper().startswith(prefix)
        ]

    def subscribe(self, tickers: Sequence[str]) -> None:
        self.start()
        normalized = tuple(normalize_quote_ticker(ticker) for ticker in tickers)
        resolved_keys = tuple(
            self._resolve_runtime_request_key(ticker) for ticker in normalized
        )
        with self._api_lock:
            self._subscribed_tickers = normalized
            self._subscribed_keys = resolved_keys

    def clear_subscription(self) -> None:
        with self._api_lock:
            self._subscribed_tickers = ()
            self._subscribed_keys = ()

    def get_subscribed_snapshots(self) -> List[Optional[SnapshotRecord]]:
        self.start()
        with self._api_lock:
            subscribed_tickers = self._subscribed_tickers
            subscribed_keys = self._subscribed_keys
        return self._get_snapshots_from_keys(subscribed_tickers, subscribed_keys)

    def get_subscribed_prices(self) -> List[Optional[float]]:
        return [
            None if record is None else record.last
            for record in self.get_subscribed_snapshots()
        ]

    def _get_snapshots_from_keys(
        self,
        tickers: Sequence[str],
        resolved_keys: Sequence[Optional[SnapshotRequestKey]],
    ) -> List[Optional[SnapshotRecord]]:
        _, proto = _load_vendor_bindings()
        results: List[Optional[SnapshotRecord]] = [None] * len(tickers)
        if not tickers:
            return results

        request_groups: Dict[Tuple[str, str], List[Tuple[int, SnapshotRequestKey]]] = {}
        for idx, key in enumerate(resolved_keys):
            if key is None:
                continue
            request_groups.setdefault((key.category, key.exch_id), []).append(
                (idx, key)
            )

        for (category, exch_id), items in request_groups.items():
            request = proto.StkSnapshotRequest()
            request.strCategory = category
            for _, key in items:
                condition = request.condition.add()
                condition.exchId = exch_id
                condition.stkCode = key.stk_code
            decoded = self._fetch_records(request, [key for _, key in items])
            decoded_iter = iter(decoded)
            for idx, _ in items:
                results[idx] = next(decoded_iter, None)
        return results

    def _resolve_runtime_request_key(
        self, ticker: str
    ) -> Optional[SnapshotRequestKey]:
        if not ticker:
            return None
        with self._api_lock:
            if ticker in self._request_key_cache:
                return self._request_key_cache[ticker]

        resolved = self._resolve_request_key(ticker)

        with self._api_lock:
            self._request_key_cache[ticker] = resolved
        return resolved

    def _fetch_records(
        self,
        request: Any,
        ordered_keys: Sequence[SnapshotRequestKey],
    ) -> List[Optional[SnapshotRecord]]:
        with self._api_lock:
            if self._api is None:
                raise RuntimeError("Quote client is not started.")
            frames = self._api.getSnapshot(request)
        decoded_records = self._decode_frames(frames, request.strCategory)

        # An empty ordered-key list means a wildcard request. Keep every
        # decoded record and derive a normalized display ticker from the
        # response category/exchange/code.
        if not ordered_keys:
            return [
                record
                for record in decoded_records
                if self._record_has_data(record)
            ]

        records_by_key: Dict[Tuple[str, str], List[SnapshotRecord]] = {}
        for record in decoded_records:
            response_key = (
                str(record.resp_exch_id).strip().upper(),
                normalize_quote_response_stk_code(
                    record.resp_exch_id, record.resp_stk_code
                ),
            )
            records_by_key.setdefault(response_key, []).append(record)

        results: List[Optional[SnapshotRecord]] = []
        for key in ordered_keys:
            response_key = (
                key.exch_id.upper(),
                normalize_quote_response_stk_code(key.exch_id, key.stk_code),
            )
            candidates = records_by_key.get(response_key, [])
            record = candidates.pop(0) if candidates else None
            if record is None or not self._record_has_data(record):
                results.append(None)
                continue
            record.ticker = key.ticker
            record.request_exch_id = key.exch_id
            record.request_stk_code = key.stk_code
            results.append(record)
        return results

    @staticmethod
    def _decode_frames(frames: Any, category: str) -> List[SnapshotRecord]:
        _, proto = _load_vendor_bindings()
        if not frames:
            return []
        decoded: List[SnapshotRecord] = []
        for payload in frames:
            if isinstance(payload, str):
                payload = payload.encode()
            message = proto.ProtoQuotationArray()
            message.ParseFromString(payload)
            items = list(message.quotationList)
            if message.HasField("quotation"):
                items.append(message.quotation)
            for quote in items:
                decoded.append(
                    SnapshotRecord(
                        ticker=_response_ticker(
                            category, quote.strExchId, quote.strCode
                        ),
                        category=str(category or "").upper(),
                        request_exch_id="",
                        request_stk_code="",
                        resp_exch_id=str(quote.strExchId or ""),
                        resp_stk_code=str(quote.strCode or ""),
                        trading_day=int(quote.iTradingDay or 0),
                        time=int(quote.iTime or 0),
                        status=int(quote.iStatus or 0),
                        last=float(quote.fLast or 0.0),
                        pre_close=float(quote.fPreClose or 0.0),
                        td_close=float(quote.fTdClose or 0.0),
                        open=float(quote.fOpen or 0.0),
                        high=float(quote.fHigh or 0.0),
                        low=float(quote.fLow or 0.0),
                        volume=int(quote.lVolume or 0),
                        turnover=float(
                            getattr(quote, "fTurnover", 0.0)
                            or getattr(quote, "lTurnover", 0.0)
                            or 0.0
                        ),
                        num_trades=int(quote.lNumTrades or 0),
                        local_time=int(quote.iLocalTime or 0),
                        ask_prices=tuple(float(value) for value in quote.fAskPrice),
                        ask_volumes=tuple(int(value) for value in quote.iAskVol),
                        bid_prices=tuple(float(value) for value in quote.fBidPrice),
                        bid_volumes=tuple(int(value) for value in quote.iBidVol),
                        open_interest=int(quote.iOpenInterest or 0),
                        pre_open_interest=int(quote.iPreOpenInterest or 0),
                        pre_settle_price=float(quote.fPreSettlePrice or 0.0),
                        settle_price=float(quote.fSettlePrice or 0.0),
                        lot_size=int(quote.iLotSize or 0),
                        trade_status=str(quote.strTradeStatus or ""),
                        close_flag=str(quote.strCloseFlag or ""),
                    )
                )
        return decoded

    @staticmethod
    def _record_has_data(record: SnapshotRecord) -> bool:
        if record.trading_day <= 0:
            return False
        return any(
            (
                abs(record.last) > 1.0e-12,
                abs(record.pre_close) > 1.0e-12,
                abs(record.open) > 1.0e-12,
                abs(record.high) > 1.0e-12,
                abs(record.low) > 1.0e-12,
                record.volume > 0,
                record.open_interest > 0,
                bool(record.ask_prices),
                bool(record.bid_prices),
            )
        )

    @classmethod
    def _resolve_request_key(
        cls, ticker: str
    ) -> Optional[SnapshotRequestKey]:
        normalized = normalize_quote_ticker(ticker)
        fields = resolve_quote_request_fields(
            normalized, request_usage=REQUEST_USAGE_SNAPSHOT
        )
        if fields is None:
            return None
        category, exch_id, stk_code = fields
        return SnapshotRequestKey(
            ticker=normalized or str(ticker or "").strip().upper(),
            category=category,
            exch_id=exch_id,
            stk_code=stk_code,
        )


def _first_positive(values: Sequence[float]) -> Optional[float]:
    for value in values:
        if float(value) > 0.0:
            return float(value)
    return None


def _response_ticker(category: str, exch_id: str, stk_code: str) -> str:
    code = str(stk_code or "").strip().upper()
    exchange = str(exch_id or "").strip().upper()
    category = str(category or "").strip().upper()
    if not code:
        return ""
    if category == "O" and exchange == "F":
        return f"{code}.CFE"
    suffix_by_exchange = {
        "0": "SH",
        "1": "SZ",
        "7": "BJ",
        "H": "HK",
        "US": "US",
        "JP": "JP",
        "KS": "KS",
        "KQ": "KQ",
    }
    suffix = suffix_by_exchange.get(exchange)
    return f"{code}.{suffix}" if suffix else code


__all__ = [
    "QuoteGatewaySnapshotClient",
    "SnapshotRecord",
    "SnapshotRequestKey",
]
