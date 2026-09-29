"""Public high-level API for the standalone vanilla surface pricer."""

from typing import Optional

from .core.market import MarketState
from .fitting.pipeline import FitResult, build_market_state, fit_surface
from .io.serialization import contract_from_dict, market_from_dict, surface_from_dict
from .pricing.contracts import VanillaContract
from .pricing.greeks import calculate_greeks
from .pricing.results import PricingResult, RiskSettings
from .pricing.vanilla import price_vanilla as _price_vanilla


def price_vanilla(
    contract: VanillaContract,
    market: MarketState,
) -> PricingResult:
    return _price_vanilla(contract, market)


def price_vanilla_with_risk(
    contract: VanillaContract,
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    bucketed_vega_pillars=None,
    bucketed_delta_pillars=None,
) -> PricingResult:
    return calculate_greeks(
        contract,
        market,
        settings=settings,
        bucketed_vega_pillars=bucketed_vega_pillars,
        bucketed_delta_pillars=bucketed_delta_pillars,
    )


def price_json(
    contract_payload: dict,
    market_payload: dict,
    calendar_file: Optional[str] = None,
    with_risk: bool = False,
    risk_settings: Optional[RiskSettings] = None,
) -> PricingResult:
    market = market_from_dict(market_payload, calendar_file=calendar_file)
    contract = contract_from_dict(contract_payload)
    if with_risk:
        return price_vanilla_with_risk(contract, market, settings=risk_settings)
    return price_vanilla(contract, market)


__all__ = [
    "FitResult",
    "build_market_state",
    "fit_surface",
    "price_json",
    "price_vanilla",
    "price_vanilla_with_risk",
    "surface_from_dict",
]
