"""Analytic vanilla pricing on top of an EDS SABR surface.

The pricer takes a resolved :class:`~surface_pricer.pricing.vanilla.spec.VanillaSpec`
(``price_spec`` / ``npv_spec``): absolute strike, absolute expiry, and the market
mapping Black-76 needs.  The raw-terms entry points (``price`` / ``npv``) simply
resolve the contract first, so both spellings price the same numbers.
"""

from typing import Tuple

from ...core.market import MarketState
from ...core.math.black import black_price
from ..results import PricingResult
from .contract import VanillaContract
from .spec import VanillaSpec, resolve_spec


class VanillaPricer:
    def __init__(self, market: MarketState):
        if market.surface is None:
            raise ValueError("MarketState.surface is required for surface vanilla pricing")
        self.market = market

    def _initial_forward(self, expiry):
        return self.market.forward(expiry, spot=self.market.surface.init_spot)

    # ------------------------------------------------------- spec (engine) path
    def details_spec(
        self, spec: VanillaSpec
    ) -> Tuple[float, float, float, float, float]:
        """``(forward, strike, tau, discount factor, vol)`` of a resolved option.

        The market mapping comes from the spec - it *is* the resolution - and only
        the vol is looked up live, so a vol bump moves it while a spot bump leaves
        the strike alone.
        """
        volatility = float(
            self.market.surface.implied_vol(
                spec.expiry_date,
                [spec.strike],
                current_forward=spec.forward,
                initial_forward=spec.initial_forward,
                valuation_date=self.market.valuation_date,
                forward_resolver=self.market.forward,
                initial_forward_resolver=self._initial_forward,
            )[0]
        )
        return (
            spec.forward,
            spec.strike,
            spec.year_fraction,
            spec.discount_factor,
            volatility,
        )

    def npv_spec(self, spec: VanillaSpec) -> float:
        forward, strike, tau, discount_factor, volatility = self.details_spec(spec)
        value = black_price(
            forward, strike, tau, volatility, discount_factor, spec.option_type
        )
        return float(value * spec.notional)

    def price_spec(self, spec: VanillaSpec) -> PricingResult:
        forward, strike, tau, discount_factor, volatility = self.details_spec(spec)
        value = black_price(
            forward, strike, tau, volatility, discount_factor, spec.option_type
        )
        return PricingResult(
            npv=float(value * spec.notional),
            forward=forward,
            discount_factor=discount_factor,
            implied_vol=volatility,
            strike=strike,
            year_fraction=tau,
            metadata={
                "option_type": spec.option_type,
                "strike_type": spec.strike_type,
                "strike_input": spec.strike_input,
                "expiry_date": spec.expiry_date.date().isoformat(),
            },
        )

    # ------------------------------------------------------- raw terms (front end)
    def details(self, contract: VanillaContract):
        return self.details_spec(resolve_spec(contract, self.market))

    def npv(self, contract: VanillaContract) -> float:
        return self.npv_spec(resolve_spec(contract, self.market))

    def price(self, contract: VanillaContract) -> PricingResult:
        return self.price_spec(resolve_spec(contract, self.market))


def price_vanilla(contract: VanillaContract, market: MarketState) -> PricingResult:
    return VanillaPricer(market).price(contract)


def price_vanilla_spec(spec: VanillaSpec, market: MarketState) -> PricingResult:
    """Price a resolved spec (the JSON layer's entry point)."""
    return VanillaPricer(market).price_spec(spec)


__all__ = ["VanillaPricer", "price_vanilla", "price_vanilla_spec"]
