"""Command line entry points.

Each module exposes ``main(argv) -> int`` and is reachable through the unified
launcher, e.g. ``python -m surface_pricer fit`` / ``... price``.  Every module
can also be run directly as a file: it re-enters itself through the package
(see the ``__package__`` guard at the top of each entry point).
"""

from .build_borrow_curve import main as build_borrow_curve_main
from .build_ir_curve import main as build_ir_curve_main
from .fit_surface import main as fit_surface_main
from .price_trades import main as price_trades_main

__all__ = [
    "build_borrow_curve_main",
    "build_ir_curve_main",
    "fit_surface_main",
    "price_trades_main",
]
