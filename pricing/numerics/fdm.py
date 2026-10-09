"""Finite-difference helpers for the 1-D autocallable PDE engine.

Tridiagonal arrays use the full-length convention: ``lower[i]`` is
``A[i, i-1]`` (``lower[0]`` ignored), ``upper[i]`` is ``A[i, i+1]``
(``upper[-1]`` ignored).  Boundary rows can therefore be replaced in place,
which is how the Dirichlet conditions are applied.

The spatial operator is the Black-Scholes generator in log-spot coordinates
``x = ln S`` on a possibly non-uniform grid::

    0.5 sigma^2 d2/dx2 + (r - q - 0.5 sigma^2) d/dx - r

and time is advanced backwards with the theta scheme (``theta = 0.5`` is
Crank-Nicolson, the default; edslib's TR-BDF2 is an alternative we do not need
here because the engine enforces observation dates as time nodes).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np


def thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system in O(n) (Thomas algorithm)."""
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float).copy()
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float).copy()
    size = diag.size
    if lower.size != size or upper.size != size or rhs.size != size:
        raise ValueError("tridiagonal arrays must share one length")
    if size == 1:
        return rhs / diag

    for index in range(1, size):
        factor = lower[index] / diag[index - 1]
        diag[index] -= factor * upper[index - 1]
        rhs[index] -= factor * rhs[index - 1]

    solution = np.empty(size, dtype=float)
    solution[-1] = rhs[-1] / diag[-1]
    for index in range(size - 2, -1, -1):
        solution[index] = (
            rhs[index] - upper[index] * solution[index + 1]
        ) / diag[index]
    return solution


def apply_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Matrix-vector product with a tridiagonal matrix (full-length arrays)."""
    values = np.asarray(vector, dtype=float)
    result = np.asarray(diag, dtype=float) * values
    result = result.copy()
    result[1:] += np.asarray(lower, dtype=float)[1:] * values[:-1]
    result[:-1] += np.asarray(upper, dtype=float)[:-1] * values[1:]
    return result


def build_log_grid(
    spot_min: float,
    spot_max: float,
    nodes: int,
    crucial_levels: Iterable[float] = (),
) -> np.ndarray:
    """Uniform log grid with the crucial levels pinned as exact nodes.

    Barriers must sit on grid nodes: missing them by half a step is the classic
    source of PDE bias for autocallables.  Each level is inserted together with
    a quarter-step refinement on either side.
    """
    spot_min = float(spot_min)
    spot_max = float(spot_max)
    if spot_min <= 0.0 or spot_max <= spot_min:
        raise ValueError("require 0 < spot_min < spot_max")
    count = max(int(nodes), 5)
    base = np.linspace(math.log(spot_min), math.log(spot_max), count)
    step = base[1] - base[0]

    extra = []
    for level in crucial_levels:
        if level is None:
            continue
        value = float(level)
        if not math.isfinite(value) or value <= 0.0:
            continue
        x = math.log(value)
        if x <= base[0] or x >= base[-1]:
            continue
        extra.extend([x - 0.25 * step, x, x + 0.25 * step])
    if not extra:
        return base
    return np.unique(np.concatenate([base, np.asarray(extra, dtype=float)]))


def _as_vector(value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(size, float(array))
    if array.size != size:
        raise ValueError("coefficient array has the wrong length")
    return array


def bsm_coefficients(
    x_grid: np.ndarray,
    sigma,
    rate,
    dividend,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black-Scholes generator coefficients on a (non-uniform) log grid.

    ``sigma`` / ``rate`` / ``dividend`` may be scalars or one value per node.
    Boundary rows are left at zero so the caller can replace them with
    Dirichlet rows.
    """
    x = np.asarray(x_grid, dtype=float)
    size = x.size
    if size < 3:
        raise ValueError("need at least three grid nodes")
    sigma = _as_vector(sigma, size)
    rate = _as_vector(rate, size)
    dividend = _as_vector(dividend, size)

    left = x[1:-1] - x[:-2]
    right = x[2:] - x[1:-1]
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("x_grid must be strictly increasing")

    variance = sigma ** 2
    diff = 0.5 * variance
    drift = rate - dividend - diff
    scale = 2.0 / (left + right)

    lower = np.zeros(size, dtype=float)
    diag = np.zeros(size, dtype=float)
    upper = np.zeros(size, dtype=float)
    lower[1:-1] = scale * diff[1:-1] / left - drift[1:-1] / (left + right)
    upper[1:-1] = scale * diff[1:-1] / right + drift[1:-1] / (left + right)
    diag[1:-1] = -scale * diff[1:-1] * (1.0 / left + 1.0 / right) - rate[1:-1]
    return lower, diag, upper


def theta_step(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    values: np.ndarray,
    *,
    dt: float,
    theta: float,
    low_boundary: float,
    high_boundary: float,
) -> np.ndarray:
    """One backward step with the theta scheme and Dirichlet boundaries."""
    if dt <= 0.0:
        return np.asarray(values, dtype=float).copy()
    theta = float(theta) if 0.0 <= float(theta) <= 1.0 else 0.5

    rhs = np.asarray(values, dtype=float) + (1.0 - theta) * dt * apply_tridiagonal(
        lower, diag, upper, values
    )
    lhs_lower = -theta * dt * np.asarray(lower, dtype=float)
    lhs_diag = 1.0 - theta * dt * np.asarray(diag, dtype=float)
    lhs_upper = -theta * dt * np.asarray(upper, dtype=float)

    lhs_lower[0] = 0.0
    lhs_diag[0] = 1.0
    lhs_upper[0] = 0.0
    rhs[0] = float(low_boundary)
    lhs_lower[-1] = 0.0
    lhs_diag[-1] = 1.0
    lhs_upper[-1] = 0.0
    rhs[-1] = float(high_boundary)
    return thomas_solve(lhs_lower, lhs_diag, lhs_upper, rhs)


__all__ = [
    "apply_tridiagonal",
    "bsm_coefficients",
    "build_log_grid",
    "theta_step",
    "thomas_solve",
]
