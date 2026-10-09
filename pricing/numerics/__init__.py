"""Product-agnostic numerical primitives.

Nothing here knows about a product: ``fdm`` only provides tridiagonal solves, log
grids and theta-scheme coefficients, so the same toolbox can back another PDE
engine (or a tree) without touching this package.
"""

from .fdm import (
    apply_tridiagonal,
    bsm_coefficients,
    build_log_grid,
    theta_step,
    thomas_solve,
)

__all__ = [
    "apply_tridiagonal",
    "bsm_coefficients",
    "build_log_grid",
    "theta_step",
    "thomas_solve",
]
