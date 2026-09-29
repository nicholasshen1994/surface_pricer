"""Configuration of the standalone EDS SABR fit.

The defaults mirror the edslib CN setup (``ML.IDX.PRICING.MID``) so that a
single from-scratch calibration run reproduces the same parameterisation,
bounds, penalties and time convention.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .overrides import OverrideConfig

# The six EDS SABR smile parameters, in fit order.
SMILE_PARAMETER_NAMES = (
    "skew",
    "conv",
    "left_skew_1",
    "left_skew_2",
    "right_skew_1",
    "right_skew_2",
)

# edslib CN parameter bounds:
#   skew (-2, 2), conv (0.001, 2),
#   left/right skew_1 (0, 10), left/right skew_2 (0, 40).
CN_PARAM_BOUNDS = (
    (-2.0, 2.0),
    (0.001, 2.0),
    (0.0, 10.0),
    (0.0, 40.0),
    (0.0, 10.0),
    (0.0, 40.0),
)

_WEIGHT_MODES = {"vega", "atm_vega", "equal", "spread"}
_FORWARD_SOURCES = {"parity", "future"}


@dataclass
class FitSettings:
    """All tunable knobs of the standalone fit pipeline."""

    # --- market time convention (edslib CN: 243 trading days, 5% holidays) ---
    trading_days_per_year: float = 243.0
    holiday_weight: float = 0.05
    min_expiry_business_days: int = 2
    max_expiry_calendar_days: int = 730

    # --- quote cleaning ---
    spread_mad_factor: float = 3.0
    min_strikes_per_expiry: int = 4

    # --- forward ---
    # "parity" (edslib default) implies the forward from put-call parity.
    # "future" keeps the exchange future price as the forward.
    forward_source: str = "parity"

    # --- weights ---
    weight_mode: str = "vega"

    # --- optimizer (iteration budget exposed on purpose) ---
    max_iterations: int = 500
    gradient_tolerance: float = 1e-9
    function_tolerance: float = 1e-9
    optimizer_method_priority: Tuple[str, ...] = ("L-BFGS-B", "SLSQP")
    param_zero_threshold: float = 1e-4
    zero_initialization: bool = False

    # --- penalties (edslib CN) ---
    mid_vol_penalty_factor: float = 10.0
    out_of_bid_ask_penalty_factor: float = 100.0
    calendar_penalty_factor: float = 1.0
    arb_check_points: int = 11
    arb_check_std_range: float = 2.0

    # --- adjacent-tenor sticking (edslib CN: next-tenor factors are all 0) ---
    sticky_to_pre_tenor_skew_factor: float = 0.0005
    sticky_to_pre_tenor_conv_factor: float = 0.0005
    sticky_to_pre_tenor_left_skew_1_factor: float = 0.0001
    sticky_to_pre_tenor_right_skew_1_factor: float = 0.0001
    sticky_to_next_tenor_skew_factor: float = 0.0
    sticky_to_next_tenor_conv_factor: float = 0.0
    sticky_to_next_tenor_left_skew_1_factor: float = 0.0
    sticky_to_next_tenor_right_skew_1_factor: float = 0.0

    # --- optional reference surface (enables sticky-to-reference penalties) ---
    # Pass an ``EDSSabrSurface`` (for example the previous official fit) to
    # keep this run close to that reference.  Single-shot runs leave it None.
    reference_surface: Optional[object] = None
    sticky_to_reference_skew_factor: float = 0.001
    sticky_to_reference_conv_factor: float = 0.001
    sticky_to_reference_left_skew_1_factor: float = 0.0001
    sticky_to_reference_right_skew_1_factor: float = 0.0001

    # --- smile parameterisation ---
    param_bounds: Sequence[Sequence[float]] = CN_PARAM_BOUNDS
    param_scaling_floor: float = 0.3

    # --- hand overrides / synthetic tenors (see ``surface_pricer.fitting.overrides``) ---
    # Pinned pillars keep their hand values exactly: the corresponding
    # dimensions are removed from the optimizer.  Other pillars - and the
    # unpinned fields of a partially pinned pillar - are fitted as usual.
    # ``None`` keeps the historical from-scratch behaviour untouched.
    override_config: Optional["OverrideConfig"] = None

    def __post_init__(self):
        self.weight_mode = str(self.weight_mode or "vega").strip().lower()
        if self.weight_mode not in _WEIGHT_MODES:
            raise ValueError(
                "weight_mode must be one of {}, got {!r}".format(
                    sorted(_WEIGHT_MODES), self.weight_mode
                )
            )
        self.forward_source = str(self.forward_source or "parity").strip().lower()
        if self.forward_source not in _FORWARD_SOURCES:
            raise ValueError(
                "forward_source must be one of {}, got {!r}".format(
                    sorted(_FORWARD_SOURCES), self.forward_source
                )
            )
        if len(self.param_bounds) != len(SMILE_PARAMETER_NAMES):
            raise ValueError(
                "param_bounds must contain one (lower, upper) pair per smile parameter"
            )
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

    def lower_bounds(self) -> np.ndarray:
        return np.asarray([pair[0] for pair in self.param_bounds], dtype=float)

    def upper_bounds(self) -> np.ndarray:
        return np.asarray([pair[1] for pair in self.param_bounds], dtype=float)


__all__ = ["FitSettings", "CN_PARAM_BOUNDS", "SMILE_PARAMETER_NAMES"]
