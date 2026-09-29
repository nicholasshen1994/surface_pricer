"""Portfolio layer: term sheets, lifecycle handling and trade valuation.

Typical flow::

    terms = load_terms("trades.xlsx")                 # portfolio.terms
    portfolio = value_portfolio(terms, market)        # portfolio.valuation
    write_csv(portfolio, "out/trades.csv")            # portfolio.report

``MarketState`` (spot / curves / fitted surface) comes from ``core`` +
``fitting``; ``pricing`` provides the NPV / Greeks engine.
"""

from .report import (
    CSV_COLUMNS,
    format_summary,
    portfolio_to_dict,
    trades_to_rows,
    write_csv,
    write_json,
)
from .schedule import TradeSchedule, TradeStatus, build_schedule
from .terms import ObservationRecord, SUPPORTED_PRODUCT_TYPES, TradeTerms, load_terms
from .valuation import (
    PortfolioValuation,
    TradeValuation,
    value_portfolio,
    value_trade,
)

__all__ = [
    "CSV_COLUMNS",
    "ObservationRecord",
    "PortfolioValuation",
    "SUPPORTED_PRODUCT_TYPES",
    "TradeSchedule",
    "TradeStatus",
    "TradeTerms",
    "TradeValuation",
    "build_schedule",
    "format_summary",
    "load_terms",
    "portfolio_to_dict",
    "trades_to_rows",
    "value_portfolio",
    "value_trade",
    "write_csv",
    "write_json",
]
