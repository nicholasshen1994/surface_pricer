"""Standalone single-index, single-currency vanilla pricer with an EDS SABR
fit pipeline aligned with the edslib CN convention.

Layer map::

    core        dates, calendars, curves, market state, numerical primitives
    marketdata  quote snapshots and providers (gateway / registry / parsing)
    fitting     vol surface calibration (prepare -> engine -> surface)
    pricing     NPV and Greeks (vanilla pricer, bump-and-revalue risk)
    portfolio   term-sheet valuation for real trades
    reporting   text reports and charts
    io          JSON adapters
    apps        command line entry points

Public names are re-exported here so ``from surface_pricer import ...`` keeps
working across the re-organisation.
"""

from .api import fit_surface, price_json, price_vanilla, price_vanilla_with_risk
from .core.curves import ConstantRateCurve, PiecewiseRateCurve
from .core.daycount import BusinessCalendar, DateHelperBusinessCalendar, year_fraction
from .core.market import MarketState
from .fitting import (
    EDSSabrFitter,
    EDSSabrSlice,
    EDSSabrSurface,
    FitResult,
    FitSettings,
    OverrideConfig,
    PillarOverride,
    SliceData,
    SliceFitResult,
    build_market_state,
    prepare_slices,
)
from .marketdata import (
    UNDERLYING_SPECS,
    MarketDataProvider,
    OptionQuote,
    OptionQuoteRecord,
    ParsedListedOption,
    QuoteApiDataProvider,
    QuoteGatewaySnapshotClient,
    QuoteSlice,
    RawSnapshot,
    SnapshotRecord,
    SpotRecord,
    UnderlyingSpec,
    get_underlying_spec,
    parse_etf_option_code,
    parse_index_option_ticker,
    parse_listed_option,
    underlying_future_ticker,
)
from .pricing.vanilla import VanillaContract
from .pricing.results import PricingResult, RiskSettings

__all__ = [
    "BusinessCalendar",
    "ConstantRateCurve",
    "DateHelperBusinessCalendar",
    "EDSSabrFitter",
    "EDSSabrSlice",
    "EDSSabrSurface",
    "FitResult",
    "FitSettings",
    "MarketDataProvider",
    "MarketState",
    "OptionQuote",
    "OptionQuoteRecord",
    "OverrideConfig",
    "ParsedListedOption",
    "PillarOverride",
    "PiecewiseRateCurve",
    "PricingResult",
    "QuoteApiDataProvider",
    "QuoteGatewaySnapshotClient",
    "QuoteSlice",
    "RawSnapshot",
    "RiskSettings",
    "SliceData",
    "SliceFitResult",
    "SnapshotRecord",
    "SpotRecord",
    "UNDERLYING_SPECS",
    "UnderlyingSpec",
    "VanillaContract",
    "build_market_state",
    "fit_surface",
    "get_underlying_spec",
    "parse_etf_option_code",
    "parse_index_option_ticker",
    "parse_listed_option",
    "prepare_slices",
    "price_json",
    "price_vanilla",
    "price_vanilla_with_risk",
    "underlying_future_ticker",
    "year_fraction",
]
