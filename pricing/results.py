"""Pricing / risk result containers and the risk calculation settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RiskSettings:
    """Bump sizes and reporting conventions of the bump-and-revalue Greeks.

    Defaults follow the edslib risk convention (``greeks/greeks.py``):
    spot bumps are 1% relative, vol bumps 0.5 vol points, rate / borrow bumps
    1bp; vega-like Greeks are reported per 1 vol point and rho-like Greeks per
    1% rate move (edslib ``get_greek_scaling()`` of 100).
    """

    delta_bump_pct: float = 0.01
    gamma_bump_pct: float = 0.01
    vega_bump: float = 0.005
    volga_bump: float = 0.005
    vanna_vol_bump: float = 0.005
    theta_days: int = 1
    rate_bump: float = 0.001
    borrow_bump: float = 0.001
    report_vega_per_vol_point: bool = True
    report_rho_per_pct: bool = True
    bucketed_vega_method: str = "point_by_point"


@dataclass
class PricingResult:
    npv: float
    forward: float
    discount_factor: float
    implied_vol: float
    strike: float
    year_fraction: float
    delta: Optional[float] = None
    delta_cash: Optional[float] = None
    delta_n: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    vanna: Optional[float] = None
    volga: Optional[float] = None
    rho: Optional[float] = None
    rhoq: Optional[float] = None
    bucketed_vega: Dict[str, float] = field(default_factory=dict)
    bucketed_rhoq: Dict[str, float] = field(default_factory=dict)
    bucketed_rho: Dict[str, float] = field(default_factory=dict)
    bucketed_delta: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["PricingResult", "RiskSettings"]
