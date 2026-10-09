"""Command line entry points.

Every CLI of the package lives here, one module per command, each exposing
``main(argv) -> int`` and reachable through the unified launcher, e.g.
``python -m surface_pricer fit`` / ``... price`` / ``... build-json`` /
``... price-json`` / ``... slide``.  A module can also be run directly as a
file: it re-enters itself through the package (see the ``__package__`` guard at
the top of each entry point).
"""

from .build_borrow_curve import main as build_borrow_curve_main
from .build_ir_curve import main as build_ir_curve_main
from .fit_surface import main as fit_surface_main

__all__ = [
    "build_borrow_curve_main",
    "build_ir_curve_main",
    "fit_surface_main",
    "price_autocall_main",
    "price_json_main",
    "price_vanilla_main",
]
