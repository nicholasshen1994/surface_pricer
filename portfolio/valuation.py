"""Valuation of real trades: per-trade pricing plus portfolio aggregation.

A trade is priced on top of a :class:`~surface_pricer.core.market.MarketState`
(spot / curves / fitted surface).  Currency amounts scale linearly with the
term-sheet notional, and all Greeks carry the same scaling, so a portfolio
total is a plain sum.

Lifecycle handling follows :mod:`surface_pricer.portfolio.schedule`: expired
trades contribute nothing (their cash flows settled), trades that have not
started yet are reported without a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

from ..core.market import MarketState
from ..pricing.contracts import VanillaContract
from ..pricing.exotics import get_pricer
from ..pricing.greeks import calculate_greeks
from ..pricing.results import PricingResult, RiskSettings
from ..pricing.vanilla import VanillaPricer
from .schedule import TradeSchedule, TradeStatus, build_schedule
from .terms import TradeTerms

_GREEK_FIELDS = ("delta", "delta_cash", "delta_n", "gamma", "vega", "theta", "vanna", "volga", "rho", "rhoq")
_BUCKET_FIELDS = ("bucketed_vega", "bucketed_delta", "bucketed_rhoq", "bucketed_rho")
_BUCKET_SHORT = {
    "bucketed_vega": "vega",
    "bucketed_delta": "delta",
    "bucketed_rhoq": "rhoq",
    "bucketed_rho": "rho",
}


@dataclass
class TradeValuation:
    """One priced trade (or one skipped trade with the reason attached)."""

    terms: TradeTerms
    schedule: TradeSchedule
    status: str
    npv: Optional[float] = None
    result: Optional[PricingResult] = None
    message: str = ""

    @property
    def trade_id(self) -> str:
        return self.terms.trade_id

    @property
    def is_valued(self) -> bool:
        return self.result is not None or self.npv is not None

    def greeks(self) -> Dict[str, Optional[float]]:
        if self.result is None:
            return {name: None for name in _GREEK_FIELDS}
        return {name: getattr(self.result, name, None) for name in _GREEK_FIELDS}

    def buckets(self) -> Dict[str, Dict[str, float]]:
        if self.result is None:
            return {name: {} for name in _BUCKET_FIELDS}
        return {name: dict(getattr(self.result, name, {}) or {}) for name in _BUCKET_FIELDS}


@dataclass
class PortfolioValuation:
    """A list of :class:`TradeValuation` plus portfolio-level aggregation."""

    valuation_date: date
    trades: List[TradeValuation] = field(default_factory=list)

    @property
    def totals(self) -> Dict[str, float]:
        totals: Dict[str, float] = {"npv": 0.0}
        totals.update({name: 0.0 for name in _GREEK_FIELDS})
        for trade in self.trades:
            if trade.npv is not None:
                totals["npv"] += float(trade.npv)
            for name, value in trade.greeks().items():
                if value is not None:
                    totals[name] = totals.get(name, 0.0) + float(value)
        return totals

    def bucketed_totals(self) -> Dict[str, Dict[str, float]]:
        """Bucket sums keyed by ``vega`` / ``delta`` / ``rhoq`` / ``rho``."""
        totals: Dict[str, Dict[str, float]] = {
            short: {} for short in _BUCKET_SHORT.values()
        }
        for trade in self.trades:
            for name, buckets in trade.buckets().items():
                short = _BUCKET_SHORT[name]
                for label, value in buckets.items():
                    totals[short][label] = totals[short].get(label, 0.0) + float(value)
        return {name: dict(sorted(values.items())) for name, values in totals.items()}

    def counts_by_status(self) -> Dict[str, int]:
        counts = {status: 0 for status in TradeStatus.ALL}
        for trade in self.trades:
            counts[trade.status] = counts.get(trade.status, 0) + 1
        return counts

    def message_summary(self) -> List[str]:
        return [
            "{}: {}".format(trade.trade_id, trade.message)
            for trade in self.trades
            if trade.message
        ]


# ------------------------------------------------------------------ valuation
def value_trade(
    terms: TradeTerms,
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    with_risk: bool = True,
    bucketed_vega_pillars: Optional[Iterable] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
) -> TradeValuation:
    """Value one trade at the valuation date of ``market``."""
    schedule = build_schedule(terms, market.valuation_date, market.calendar)

    if schedule.is_not_started:
        return TradeValuation(
            terms=terms,
            schedule=schedule,
            status=schedule.status,
            npv=None,
            message="not started ({}): not valued".format(
                schedule.start_date.isoformat() if schedule.start_date else "?"
            ),
        )
    if schedule.is_expired:
        return TradeValuation(
            terms=terms,
            schedule=schedule,
            status=schedule.status,
            npv=0.0,
            message="expired ({}): settled, excluded from the portfolio".format(
                schedule.expiry_date.isoformat()
            ),
        )

    result = _price(terms, market, settings, with_risk, bucketed_vega_pillars, bucketed_delta_pillars)
    return TradeValuation(
        terms=terms,
        schedule=schedule,
        status=schedule.status,
        npv=float(result.npv),
        result=result,
    )


def value_portfolio(
    trades: Sequence[TradeTerms],
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    with_risk: bool = True,
    bucketed_vega_pillars: Optional[Iterable] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
) -> PortfolioValuation:
    """Value a whole term sheet on one market state."""
    valuations = [
        value_trade(
            terms,
            market,
            settings=settings,
            with_risk=with_risk,
            bucketed_vega_pillars=bucketed_vega_pillars,
            bucketed_delta_pillars=bucketed_delta_pillars,
        )
        for terms in trades
    ]
    return PortfolioValuation(
        valuation_date=market.valuation_date.date(),
        trades=valuations,
    )


def _price(
    terms: TradeTerms,
    market: MarketState,
    settings: Optional[RiskSettings],
    with_risk: bool,
    bucketed_vega_pillars: Optional[Iterable],
    bucketed_delta_pillars: Optional[Iterable],
) -> PricingResult:
    if not terms.is_vanilla:
        factory = get_pricer(terms.product_type)
        if factory is None:
            raise ValueError(
                "product_type {!r} of {} is not supported yet; available: vanilla "
                "(register exotic pricers in surface_pricer.pricing.exotics)".format(
                    terms.product_type, terms.trade_id
                )
            )
        pricer = factory(terms, market)
        if with_risk:
            return pricer.greeks(terms, market, settings=settings)
        return pricer.price(terms, market, settings=settings)

    contract = VanillaContract(
        expiry=terms.expiry_date,
        strike=terms.strike,
        option_type=terms.call_put,
        notional=terms.notional,
        strike_type=terms.strike_type,
    )
    if with_risk:
        return calculate_greeks(
            contract,
            market,
            settings=settings,
            bucketed_vega_pillars=bucketed_vega_pillars,
            bucketed_delta_pillars=bucketed_delta_pillars,
        )
    return VanillaPricer(market).price(contract)


__all__ = [
    "PortfolioValuation",
    "TradeValuation",
    "value_portfolio",
    "value_trade",
]
