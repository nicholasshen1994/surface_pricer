"""Pricing / risk result containers and the risk calculation settings."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence

#: The scalar Greek fields :meth:`PricingResult.zero_greeks` clears.
_GREEK_FIELDS = (
    "delta",
    "delta_cash",
    "delta_n",
    "gamma",
    "gamma_cash",
    "vega",
    "theta",
    "vanna",
    "volga",
    "rho",
    "rhoq",
)


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
    #: Which Greeks a risk run computes.  ``None`` -> all (the library default,
    #: so existing callers keep their full risk run); ``()`` -> none, i.e. NPV
    #: only, which is what ``price_autocall`` does unless ``--greeks`` asks for
    #: more.  Names are validated by :func:`..risk.diff.parse_greeks`.
    greeks: Optional[Sequence[str]] = None
    #: Coarse-bucket policy of the **auto** curve grids (borrow / rate): up to this
    #: tenor every pillar keeps a bucket of its own, the rest merge into one bucket
    #: per year of tenor, and a pillar past the trade's horizon - which the pricer
    #: passes in - is dropped, keeping only the one that brackets it.  A coarse
    #: grid is what makes ``bucketed_delta`` affordable on a long-dated snowball:
    #: a 15-pillar curve with a 2Y trade costs a handful of bump pairs instead of
    #: thirty valuations.  ``None`` keeps a bucket per pillar (the pre-2026-10
    #: grid, and what ``--full-bucket-grid`` asks for).  A pinned pillar list is
    #: always used as given.
    bucket_group_after: Optional[str] = "1Y"

    # ---- exotic engines (autocallable) -------------------------------------
    # Barrier smoothing keeps the discrete KO/KI indicators continuous so the
    # bump-and-revalue Greeks stay stable (the payoff is continuous in spot).
    # ``barrier_smooth_width`` is the band half-width relative to the level;
    # ``barrier_smooth_floor`` is an absolute lower bound for that band.
    barrier_smooth: bool = True
    barrier_smooth_width: float = 0.01
    barrier_smooth_floor: float = 0.0
    # Monte Carlo defaults (edslib: 65535 paths, single seed).  Six steps per
    # observation segment keep the local-vol discretisation error small - one
    # step biases the knock-in probability by ~17% on a quarterly snowball.
    mc_paths: int = 65536
    mc_seed: int = 0
    mc_steps_per_observation: int = 6
    # Greeks run many bumped valuations; the common random numbers keep the
    # finite differences paired, so the CLI uses **one** path count (``--paths``)
    # for both the price and the bumps - there is no separate ``--greek-paths`` any
    # more (2026-10).  ``None`` means "same as :attr:`mc_paths`"; a library caller
    # may still pin a smaller number here.
    mc_greek_paths: Optional[int] = None
    # PDE defaults (edslib: ~500 nodes, 0.2 vol-time steps).  Time discretisation
    # now follows the MC engine: every observation date is a node and each
    # window carries ``pde_steps_per_observation`` equal vol-time sub-steps.
    # There is no global step-length cap any more (the old ``pde_time_step``
    # only ever bound windows wider than ~1.2 vol-years - i.e. long-dated
    # annual trades - and duplicated this knob).
    pde_nodes: int = 601
    pde_steps_per_observation: int = 6
    pde_theta: float = 0.5


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
    #: ``gamma * spot^2 / 100`` - the gamma in cash terms, i.e. the NPV change per
    #: (1% spot move)^2 (edslib's dollar gamma / 100).  Reported next to ``gamma``
    #: because that is how a desk sizes the convexity of a position.
    gamma_cash: Optional[float] = None
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

    def zero_greeks(self, greeks: Optional[Sequence[str]] = None) -> "PricingResult":
        """Every Greek cleared - what a **settled** (knocked-out) trade reports.

        Once the trade has knocked out it is a single discounted cash flow: the NPV
        stays (that is the accrued coupon), but there is nothing left to bump, so
        delta, gamma, vega, theta, vanna, volga and the curve Greeks are all zero
        and the bucketed tables are empty.  ``greeks`` names the selection a run
        asked for (a settled trade never computes one, so the fields have to be
        filled in as zero); a Greek nobody asked for stays ``None``.
        """
        wanted = set(_GREEK_FIELDS) if greeks is None else {str(name) for name in greeks}
        cleared = replace(
            self,
            bucketed_vega={},
            bucketed_rhoq={},
            bucketed_rho={},
            bucketed_delta={},
        )
        for name in _GREEK_FIELDS:
            if name in wanted or getattr(cleared, name) is not None:
                setattr(cleared, name, 0.0)
        return cleared


__all__ = ["PricingResult", "RiskSettings"]
