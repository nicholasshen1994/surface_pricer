"""Public high-level API for the standalone vanilla surface pricer."""

from typing import Optional

from .core.market import MarketState
from .fitting.pipeline import FitResult, build_market_state, fit_surface
from .io.serialization import contract_from_dict, market_from_dict, surface_from_dict
from .pricing.results import PricingResult, RiskSettings
from .pricing.vanilla import (
    VanillaContract,
    VanillaSpec,
    calculate_greeks,
    calculate_greeks_spec,
    resolve_spec,
)
from .pricing.vanilla import price_vanilla as _price_vanilla
from .pricing.vanilla import price_vanilla_spec as _price_vanilla_spec


def price_vanilla(
    contract: VanillaContract,
    market: MarketState,
) -> PricingResult:
    return _price_vanilla(contract, market)


def price_spec(
    spec: VanillaSpec,
    market: MarketState,
) -> PricingResult:
    """Price a resolved spec - the JSON layer's entry point."""
    return _price_vanilla_spec(spec, market)


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


def price_spec_with_risk(
    spec: VanillaSpec,
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    bucketed_vega_pillars=None,
    bucketed_delta_pillars=None,
) -> PricingResult:
    return calculate_greeks_spec(
        spec,
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
    """Price a JSON contract on a JSON market.

    The payload may be **raw terms** (``strike`` + ``strike_type``, resolved against
    the market here) or a resolved spec (``"kind": "vanilla_spec"``, or the
    ``contract`` block of a ``price-tool --json`` quote) - the latter is priced
    exactly as written.
    """
    market = market_from_dict(market_payload, calendar_file=calendar_file)
    if _is_resolved(contract_payload):
        spec = VanillaSpec.from_dict(contract_payload, market)
        if with_risk:
            return price_spec_with_risk(spec, market, settings=risk_settings)
        return price_spec(spec, market)

    contract = contract_from_dict(contract_payload)
    if with_risk:
        return price_vanilla_with_risk(contract, market, settings=risk_settings)
    return price_vanilla(contract, market)


def _is_resolved(payload: dict) -> bool:
    """A resolved spec (or a quote wrapping one) rather than raw terms."""
    if not isinstance(payload, dict):
        return False
    if payload.get("kind") == "vanilla_spec":
        return True
    inner = payload.get("contract")
    return isinstance(inner, dict) and inner.get("kind") == "vanilla_spec"


__all__ = [
    "FitResult",
    "VanillaSpec",
    "build_market_state",
    "fit_surface",
    "price_json",
    "price_spec",
    "price_spec_with_risk",
    "price_vanilla",
    "price_vanilla_with_risk",
    "resolve_spec",
    "surface_from_dict",
]
