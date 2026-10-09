"""Pricing rules: business conventions applied before an engine runs.

A rule is part of the *pricing* of a trade (not a risk pre-process), so it is
resolved here and expanded into the effective terms exactly once, by
``build_schedule`` of the product.  The packaged standard lives in
``surface_pricer/config/barrier_shift.json``.
"""

from .barrier_shift import (
    CONFIG_ENV_VAR,
    MODE_ADDITIVE,
    MODE_NONE,
    MODE_RELATIVE,
    PACKAGE_CONFIG_PATH,
    BarrierShiftSpec,
    ShiftConfig,
    ShiftConfigError,
    clear_shift_config_cache,
    contract_max_gap,
    expand_shift,
    load_shift_config,
    resolve_shift,
)

__all__ = [
    "BarrierShiftSpec",
    "CONFIG_ENV_VAR",
    "MODE_ADDITIVE",
    "MODE_NONE",
    "MODE_RELATIVE",
    "PACKAGE_CONFIG_PATH",
    "ShiftConfig",
    "ShiftConfigError",
    "clear_shift_config_cache",
    "contract_max_gap",
    "expand_shift",
    "load_shift_config",
    "resolve_shift",
]
