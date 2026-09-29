"""Contract definitions priced by the :mod:`surface_pricer.pricing` layer.

Exotic products (barrier / autocallable / accumulator) are added in a later
phase as extra contract classes plus an entry in
:mod:`surface_pricer.pricing.exotics`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.daycount import DateLike, to_datetime


@dataclass
class VanillaContract:
    expiry: DateLike
    strike: float
    option_type: str = "call"
    notional: float = 1.0
    strike_type: str = "absolute"

    def __post_init__(self):
        self.expiry = to_datetime(self.expiry)
        option_type = self.option_type.lower()
        if option_type in {"c", "call"}:
            self.option_type = "call"
        elif option_type in {"p", "put"}:
            self.option_type = "put"
        else:
            raise ValueError("option_type must be call or put")

    def absolute_strike(self, spot: float, fwd: float) -> float:
        kind = self.strike_type.lower()
        if kind in {"percentage", "spot_percentage", "spot_relative"}:
            return float(self.strike * spot)
        if kind in {"fwd_percentage", "forward_percentage", "forward_relative"}:
            return float(self.strike * fwd)
        return float(self.strike)


__all__ = ["VanillaContract"]
