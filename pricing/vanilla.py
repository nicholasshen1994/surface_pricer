"""Analytic vanilla pricing on top of an EDS SABR surface."""

from dataclasses import dataclass
from typing import Optional

from ..core.market import MarketState
from ..core.math.black import black_price
from .contracts import VanillaContract
from .results import PricingResult


class VanillaPricer:
    def __init__(self, market: MarketState):
        if market.surface is None:
            raise ValueError("MarketState.surface is required for surface vanilla pricing")
        self.market = market

    def _initial_forward(self, expiry):
        return self.market.forward(expiry, spot=self.market.surface.init_spot)

    def details(self, contract: VanillaContract):
        forward = self.market.forward(contract.expiry)
        strike = contract.absolute_strike(self.market.spot, forward)
        tau = self.market.year_fraction(contract.expiry)
        discount_factor = self.market.discount_factor(contract.expiry)
        initial_forward = self._initial_forward(contract.expiry)
        volatility = float(
            self.market.surface.implied_vol(
                contract.expiry,
                [strike],
                current_forward=forward,
                initial_forward=initial_forward,
                valuation_date=self.market.valuation_date,
                forward_resolver=self.market.forward,
                initial_forward_resolver=self._initial_forward,
            )[0]
        )
        return forward, strike, tau, discount_factor, volatility

    def npv(self, contract: VanillaContract) -> float:
        forward, strike, tau, discount_factor, volatility = self.details(contract)
        value = black_price(
            forward,
            strike,
            tau,
            volatility,
            discount_factor,
            contract.option_type,
        )
        return float(value * contract.notional)

    def price(self, contract: VanillaContract) -> PricingResult:
        forward, strike, tau, discount_factor, volatility = self.details(contract)
        value = black_price(
            forward,
            strike,
            tau,
            volatility,
            discount_factor,
            contract.option_type,
        )
        return PricingResult(
            npv=float(value * contract.notional),
            forward=forward,
            discount_factor=discount_factor,
            implied_vol=volatility,
            strike=strike,
            year_fraction=tau,
            metadata={"option_type": contract.option_type},
        )


def price_vanilla(contract: VanillaContract, market: MarketState) -> PricingResult:
    return VanillaPricer(market).price(contract)
