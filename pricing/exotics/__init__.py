"""Exotic product family: the plugin contract, the registry and the products.

Each exotic is a **package** next to this module (``autocall/`` today, barrier /
accumulator later) holding its own terms, effective schedule, cash flows and
engines, so a new product never shares a file with an engine:

* a pricer is any object implementing :class:`ExoticPricer` (``price`` and
  ``greeks`` taking a contract, a :class:`~surface_pricer.core.market.MarketState`
  and optional :class:`~surface_pricer.pricing.results.RiskSettings`);
* :func:`register_pricer` maps a ``product_type`` string (the same value used in
  the term-sheet files) to a factory;
* :func:`get_pricer` / :func:`available_product_types` let callers dispatch by
  product type and fail with a clear message for unsupported ones.

Products register themselves on import (see the trailing import below).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

from ..results import PricingResult


@runtime_checkable
class ExoticPricer(Protocol):
    """Minimal interface a future exotic pricer must implement."""

    def price(self, contract, market, settings=None) -> PricingResult:
        """Return the NPV (and any analytic values) of one contract."""

    def greeks(self, contract, market, settings=None) -> PricingResult:
        """Return the NPV together with the requested Greeks."""


_PRICERS: Dict[str, Callable[..., ExoticPricer]] = {}


def register_pricer(product_type: str, factory: Callable[..., ExoticPricer]) -> None:
    """Register a pricer factory under a product-type key (e.g. ``"barrier"``)."""
    key = str(product_type or "").strip().lower()
    if not key:
        raise ValueError("product_type must not be empty")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _PRICERS[key] = factory


def get_pricer(product_type: str) -> Optional[Callable[..., ExoticPricer]]:
    """Return the factory registered for ``product_type`` (or ``None``)."""
    return _PRICERS.get(str(product_type or "").strip().lower())


def available_product_types() -> Tuple[str, ...]:
    """Product types with a registered pricer, sorted for stable reporting."""
    return tuple(sorted(_PRICERS))


__all__ = [
    "ExoticPricer",
    "available_product_types",
    "get_pricer",
    "register_pricer",
]


# Import the shipped product modules last so they can register themselves while
# importing this package for the registry (no circular import).
from . import autocall  # noqa: E402,F401  (registers autocallable / snowball / autocall)
