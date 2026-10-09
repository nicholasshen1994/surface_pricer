"""Risk layer: the bump objects and stencils every Greek is built from.

``bumps`` holds the market-state bumps (public, shared by both products),
``diff`` the product-agnostic bump-and-revalue driver used by the exotic
engines, ``buckets`` the per-pillar (bucketed) Greeks shared by the vanilla
pricer and the exotic engines, and ``slide`` the spot ladder built on top of
them.  The vanilla *parallel* stencils live in
:mod:`surface_pricer.pricing.vanilla.greeks`.
"""

from .buckets import (
    BUCKET_NAMES,
    bucket_greeks,
    convert_bucketed_delta,
    curve_bucket,
    vega_bucket,
    vol_pillars,
)
from .bumps import curve_bump, parallel_bump, require_surface, spot_bump, vol_bump
from .diff import GREEK_CONVENTION, GREEK_NAMES, ValueFn, bump_greeks, parse_greeks
from .slide import SlideRow, run_slide, spot_ladder

__all__ = [
    "BUCKET_NAMES",
    "GREEK_CONVENTION",
    "GREEK_NAMES",
    "SlideRow",
    "ValueFn",
    "bucket_greeks",
    "bump_greeks",
    "convert_bucketed_delta",
    "curve_bucket",
    "curve_bump",
    "parallel_bump",
    "parse_greeks",
    "require_surface",
    "run_slide",
    "spot_bump",
    "spot_ladder",
    "vega_bucket",
    "vol_bump",
    "vol_pillars",
]
