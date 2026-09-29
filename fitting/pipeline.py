"""Public entry point of the standalone EDS SABR fit pipeline.

    raw snapshot -> quote cleaning / parity forward -> sequential fit -> surface

The data source lives behind :mod:`providers`; everything else (preparation,
calibration, surface assembly) is self contained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.daycount import to_datetime
from ..core.market import MarketState
from ..marketdata.data import RawSnapshot
from .engine import EDSSabrFitter, SliceFitResult
from .overrides import SMILE_FIELD_TO_SURFACE, extend_and_apply
from .prepare import SliceData, prepare_slices
from .settings import SMILE_PARAMETER_NAMES, FitSettings
from .surface import EDSSabrSurface


@dataclass
class FitResult:
    """Surface plus per-expiry calibration details."""

    surface: EDSSabrSurface
    slices: List[SliceFitResult]
    spot: float
    valuation_datetime: datetime
    underlying: str = ""
    forward_overrides: Dict[str, float] = field(default_factory=dict)
    settings: FitSettings = field(default_factory=FitSettings)
    market: Optional[MarketState] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    # Hand override configuration actually used, kept for reproducibility.
    override_config: Optional[object] = None

    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        return {
            result.expiry.date().isoformat(): {
                "tau": result.tau,
                "forward": result.forward,
                "atm_vol": result.atm_vol,
                "rmse": result.rmse,
                "weighted_rmse": result.weighted_rmse,
                "n_quotes": float(len(result.slice_info.strikes)),
                "out_of_bid_ask": float(len(result.out_of_bid_ask)),
                "fixed": 1.0 if result.is_override else 0.0,
                "synthetic": 1.0 if result.is_synthetic else 0.0,
            }
            for result in self.slices
        }


def build_market_state(
    snapshot: RawSnapshot,
    settings: Optional[FitSettings] = None,
) -> MarketState:
    """Wrap a raw snapshot into the market state consumed by the fit."""
    settings = settings or FitSettings()
    return MarketState(
        valuation_date=snapshot.valuation_datetime,
        spot=snapshot.spot,
        rate_curve=snapshot.rate_curve,
        borrow_curve=getattr(snapshot, "borrow_curve", None),
        calendar=snapshot.calendar,
        trading_days_per_year=settings.trading_days_per_year,
        holiday_weight=settings.holiday_weight,
    )


def fit_surface(
    snapshot: RawSnapshot,
    settings: Optional[FitSettings] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> FitResult:
    """Run one from-scratch EDS SABR calibration on a raw snapshot.

    ``progress`` receives human readable status messages (pass ``print`` for
    command line runs).
    """
    settings = settings or FitSettings()
    _report(
        progress,
        "snapshot {} | {} option records | {} spot records | valuation={:%Y-%m-%d %H:%M:%S}".format(
            snapshot.underlying,
            len(snapshot.option_records),
            len(snapshot.spot_records),
            snapshot.valuation_datetime,
        ),
    )
    market = build_market_state(snapshot, settings)
    _report(progress, "preparing slices (MAD clean / parity forward / OTM IV / weights) ...")
    started = time.perf_counter()
    slices, forward_overrides = prepare_slices(snapshot, settings, market)
    if not slices:
        raise ValueError(
            "no fit-ready slices produced from the snapshot of {}".format(
                snapshot.underlying
            )
        )
    _report(
        progress,
        "prepared {} slices in {:.2f}s".format(len(slices), time.perf_counter() - started),
    )
    for slice_info in slices:
        _report(
            progress,
            "  {} | {:2d} quotes | fwd={:.4f} | tau={:.4f} | forward={}".format(
                slice_info.expiry.date().isoformat(),
                len(slice_info.strikes),
                slice_info.forward,
                slice_info.tau,
                slice_info.diagnostics.get("forward_source", ""),
            ),
        )
    _report(
        progress,
        "fitting with weight={} maxiter={} ...".format(
            settings.weight_mode, settings.max_iterations
        ),
    )
    market = market.clone(forward_overrides=forward_overrides)
    fitter = EDSSabrFitter(slices, settings, progress=progress)
    surface, results = fitter.fit(market)
    surface, results = _apply_overrides(surface, results, fitter, market, settings, progress)
    market = market.clone(surface=surface)
    return FitResult(
        surface=surface,
        slices=results,
        spot=float(snapshot.spot),
        valuation_datetime=snapshot.valuation_datetime,
        underlying=snapshot.underlying,
        forward_overrides=forward_overrides,
        settings=settings,
        market=market,
        diagnostics=dict(snapshot.diagnostics),
        override_config=getattr(settings, "override_config", None),
    )


def _apply_overrides(
    surface: EDSSabrSurface,
    results: List[SliceFitResult],
    fitter: EDSSabrFitter,
    market: MarketState,
    settings: FitSettings,
    progress: Optional[Callable[[str], None]],
) -> Tuple[EDSSabrSurface, List[SliceFitResult]]:
    """Extend the fitted surface with hand pillars / synthetic tenors.

    Pinned pillars with listed quotes were already handled inside the engine
    (their dimensions were dropped from the optimiser); this step adds the
    pillars that do not exist in the listed market and records them.
    """
    resolution = getattr(fitter, "override_resolution", None)
    if resolution is None or not resolution.has_work:
        return surface, list(results)

    extended, applied, new_pillars = extend_and_apply(surface, resolution)
    extended_results = list(results)
    for expiry in new_pillars:
        extended_results.append(_synthetic_slice_result(extended, expiry, market, settings))
    extended_results.sort(key=lambda item: item.expiry)

    if applied or new_pillars:
        _report(
            progress,
            "hand overrides | {} pillar(s) pinned | {} pillar(s) added beyond the listed expiries".format(
                len(applied), len(new_pillars)
            ),
        )
    return extended, extended_results


def _synthetic_slice_result(
    surface: EDSSabrSurface,
    expiry,
    market: MarketState,
    settings: FitSettings,
    note: str = "hand / synthetic pillar, values interpolated from the fitted surface",
) -> SliceFitResult:
    """Wrap an added pillar into a :class:`SliceFitResult` for reporting."""
    expiry = to_datetime(expiry)
    tau = float(market.year_fraction(expiry))
    forward = float(market.forward(expiry))
    index = surface.pillar_index(expiry)
    scale = max(
        float(settings.param_scaling_floor),
        float(np.sqrt(max(tau, 1.0e-12))),
    )
    if index is None:
        params = np.zeros(len(SMILE_PARAMETER_NAMES), dtype=float)
    else:
        params = np.asarray(
            [
                float(getattr(surface, SMILE_FIELD_TO_SURFACE[name])[index]) / scale
                for name in SMILE_PARAMETER_NAMES
            ],
            dtype=float,
        )
    slice_info = SliceData(
        expiry=expiry,
        tau=tau,
        forward=forward,
        discount_factor=float(market.discount_factor(expiry)),
        strikes=np.asarray([], dtype=float),
        ln_moneyness=np.asarray([], dtype=float),
        vols=np.asarray([], dtype=float),
        bid_vols=np.asarray([], dtype=float),
        ask_vols=np.asarray([], dtype=float),
        weights=np.asarray([], dtype=float),
        option_types=(),
        diagnostics={"forward_source": "interpolated", "n_quotes": 0.0},
    )
    return SliceFitResult(
        slice_info=slice_info,
        params=params,
        fitted=surface.get_slice(expiry, current_forward=forward),
        atm_vol=float(surface.get_atm_vol(surface.get_vol_time(expiry))),
        rmse=0.0,
        weighted_rmse=0.0,
        converged=True,
        message=note,
        optimizer_method="synthetic",
        iterations=0,
        function_evaluations=0,
        is_synthetic=True,
        override_source="synthetic",
    )


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)


__all__ = ["FitResult", "build_market_state", "fit_surface"]
