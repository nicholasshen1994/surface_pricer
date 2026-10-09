"""Pricing layer: NPV and Greeks on top of a fitted vol surface.

The layer depends on ``core`` (market state / math) and ``fitting`` (surface)
and is consumed by ``portfolio`` and ``apps``.  Inside the layer everything is
grouped by function, and the dependencies only ever point one way::

    results            result containers and the risk settings
    risk               bump objects + the shared bump-and-revalue driver
    models             coefficients implied by the surface (Dupire local vol)
    numerics           product-agnostic numerics (tridiagonal solves, log grids)
    rules              pricing rules (barrier shift) applied before an engine
    vanilla            product: European options (contract / pricer / greeks)
    exotics            product family: one package per exotic, engines inside

A product package owns its terms, its schedule / cash flows and its engines; it
imports the shared layers, never another product.  The exotic plugin contract
(``ExoticPricer`` / ``register_pricer`` / ``get_pricer``) stays in
:mod:`surface_pricer.pricing.exotics`.
"""

from .results import PricingResult, RiskSettings
from .vanilla import VanillaContract, VanillaPricer, calculate_greeks, price_vanilla

__all__ = [
    "PricingResult",
    "RiskSettings",
    "VanillaContract",
    "VanillaPricer",
    "calculate_greeks",
    "price_vanilla",
]
