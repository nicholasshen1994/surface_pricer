"""Market data layer: raw snapshot containers plus provider implementations.

Everything that talks to an external quote source lives here.  The layer only
exposes plain containers (:mod:`surface_pricer.marketdata.data`) and provider
classes, so higher layers never depend on a specific data source.
"""

from .data import OptionQuoteRecord, RawSnapshot, SpotRecord
from .future_inputs import future_spots_by_expiry, future_tickers_for_option_records
from .gateway import QuoteGatewaySnapshotClient, SnapshotRecord
from .listed_contracts import (
    EXCHANGE_CFFEX,
    EXCHANGE_SSE,
    EXCHANGE_SZSE,
    ParsedListedOption,
    cffex_option_expiry_date,
    cffex_option_expiry_datetime,
    etf_option_expiry_date,
    etf_option_expiry_datetime,
    parse_etf_option_code,
    parse_index_option_ticker,
    parse_listed_option,
    underlying_future_ticker,
)
from .offline_quotes import OptionQuote, QuoteSlice
from .providers import DEFAULT_INDEX_RATE, MarketDataProvider, QuoteApiDataProvider
from .registry import UNDERLYING_SPECS, UnderlyingSpec, get_underlying_spec

__all__ = [
    "DEFAULT_INDEX_RATE",
    "EXCHANGE_CFFEX",
    "EXCHANGE_SSE",
    "EXCHANGE_SZSE",
    "MarketDataProvider",
    "OptionQuote",
    "OptionQuoteRecord",
    "ParsedListedOption",
    "QuoteApiDataProvider",
    "QuoteGatewaySnapshotClient",
    "QuoteSlice",
    "RawSnapshot",
    "SnapshotRecord",
    "SpotRecord",
    "UNDERLYING_SPECS",
    "UnderlyingSpec",
    "cffex_option_expiry_date",
    "cffex_option_expiry_datetime",
    "etf_option_expiry_date",
    "etf_option_expiry_datetime",
    "future_spots_by_expiry",
    "future_tickers_for_option_records",
    "get_underlying_spec",
    "parse_etf_option_code",
    "parse_index_option_ticker",
    "parse_listed_option",
    "underlying_future_ticker",
]
