"""Sequential EDS SABR calibration engine.

Ported from the edslib CN fitter (``EDSSabrSimpleVolFitter``):

* ``liquidity_order_fit`` scheduling - slices are fitted from the most liquid
  one to the least liquid one, and every slice is constrained by its nearest
  more-liquid neighbours through calendar-arbitrage and parameter penalties;
* ``_calculate_slice_penalties`` - mid-vol fit error (x10), out-of-bid-ask
  error (x100), calendar arbitrage smoothing, adjacent-tenor sticking and the
  optional sticky-to-reference term;
* ``_adjust_initial_guess`` - zero the guess when the initial cost is bad and
  estimate skew/conv from the 95%/100%/105% quotes;
* L-BFGS-B with an SLSQP fallback, small parameters clamped to zero, and the
  final surface parameters scaled by ``max(0.3, sqrt(tau))``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ..core.daycount import to_date
from ..core.market import MarketState
from .eds_slice import EDSSabrSlice
from .overrides import surface_to_slice_values, validate_slice_values
from .prepare import SliceData
from .settings import SMILE_PARAMETER_NAMES, FitSettings
from .surface import EDSSabrSurface

_EXP_OVERFLOW_LIMIT = 700.0


@dataclass
class SliceFitResult:
    """Calibration result of one expiry."""

    slice_info: SliceData
    params: np.ndarray
    fitted: EDSSabrSlice
    atm_vol: float
    rmse: float
    weighted_rmse: float
    out_of_bid_ask: Dict[float, float] = field(default_factory=dict)
    converged: bool = True
    message: str = ""
    optimizer_method: str = ""
    iterations: int = 0
    function_evaluations: int = 0
    elapsed_seconds: float = 0.0
    # ``is_override`` marks a pillar with hand-pinned values (partial or full),
    # ``is_synthetic`` a pillar added on top of the fitted expiries.
    is_override: bool = False
    is_synthetic: bool = False
    override_source: str = ""

    @property
    def expiry(self) -> datetime:
        return self.slice_info.expiry

    @property
    def tau(self) -> float:
        return self.slice_info.tau

    @property
    def forward(self) -> float:
        return self.slice_info.forward


class EDSSabrFitter:
    """One-shot (from-scratch) sequential EDS SABR fitter."""

    def __init__(
        self,
        slices: Sequence[SliceData],
        settings: FitSettings,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.slices = sorted(slices, key=lambda item: item.expiry)
        self.settings = settings
        self._progress = progress
        self._arb_grid: Dict[datetime, Tuple[np.ndarray, np.ndarray]] = {}
        self._reference_params: Dict[datetime, np.ndarray] = {}
        self._slice_by_date: Dict[date, SliceData] = {
            to_date(slice_info.expiry): slice_info for slice_info in self.slices
        }
        # Hand overrides, resolved lazily in ``fit`` (tenor specs need the
        # valuation date and the calendar of the market state).
        self.override_resolution = None
        self._fixed_atm: Dict[datetime, float] = {}
        self._fixed_params: Dict[datetime, np.ndarray] = {}
        self._fixed_indices: Dict[datetime, Tuple[int, ...]] = {}
        self._override_sources: Dict[datetime, str] = {}
        self._build_arb_grids()
        self._build_reference_params()

    def _report(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)

    # ----------------------------------------------------------------- public
    def fit(
        self, market: MarketState
    ) -> Tuple[EDSSabrSurface, List[SliceFitResult]]:
        """Run the sequential calibration and build the SABR surface."""
        settings = self.settings
        self._prepare_overrides(market)
        schedule = self._liquidity_schedule()
        self._report(
            "sequential fit order (most liquid first): {}".format(
                ", ".join(
                    self.slices[index].expiry.date().isoformat()
                    for index, _ in schedule
                )
            )
        )
        self._report(
            "optimizer: methods={}, maxiter={}, gtol={:g}, ftol={:g}".format(
                "/".join(settings.optimizer_method_priority),
                settings.max_iterations,
                settings.gradient_tolerance,
                settings.function_tolerance,
            )
        )

        results: Dict[int, SliceFitResult] = {}
        fitted: Dict[int, EDSSabrSlice] = {}
        total = len(schedule)

        for position, (index, constraints) in enumerate(schedule, start=1):
            slice_info = self.slices[index]
            self._report(
                "[{}/{}] fitting {} | {} quotes | fwd={:.3f} | tau={:.4f}".format(
                    position,
                    total,
                    slice_info.expiry.date().isoformat(),
                    len(slice_info.strikes),
                    slice_info.forward,
                    slice_info.tau,
                )
            )
            pre_slice, next_slice = None, None
            for constraint in constraints:
                if constraint < index and constraint in fitted:
                    pre_slice = fitted[constraint]
                elif constraint > index and constraint in fitted:
                    next_slice = fitted[constraint]

            started = time.perf_counter()
            result = self._fit_one_slice(self.slices[index], pre_slice, next_slice)
            elapsed = time.perf_counter() - started
            if result is None:
                self._report(
                    "[{}/{}] {} FAILED after {:.1f}s".format(
                        position, total, slice_info.expiry.date().isoformat(), elapsed
                    )
                )
                continue
            result.elapsed_seconds = elapsed
            results[index] = result
            fitted[index] = result.fitted
            self._report(
                "[{}/{}] {} done | rmse={:.5f} | out-of-ba={} | {} nit={} nfev={} | {:.1f}s".format(
                    position,
                    total,
                    slice_info.expiry.date().isoformat(),
                    result.rmse,
                    len(result.out_of_bid_ask),
                    result.optimizer_method,
                    result.iterations,
                    result.function_evaluations,
                    elapsed,
                )
            )

        ordered = [results[index] for index in sorted(results)]
        if not ordered:
            raise ValueError("all slices failed to calibrate; check quotes and bounds")
        surface = self._build_surface(market, ordered)
        self._report("surface built | {} expiries".format(len(ordered)))
        return surface, ordered

    # ----------------------------------------------------------- overrides
    def _prepare_overrides(self, market: MarketState) -> None:
        """Resolve hand overrides and lock the pinned optimiser dimensions."""
        config = getattr(self.settings, "override_config", None)
        if config is None:
            return
        resolution = config.resolve(
            market.valuation_date,
            market.calendar,
            [slice_info.expiry for slice_info in self.slices],
        )
        self.override_resolution = resolution
        scaling_floor = float(
            getattr(config, "scaling_floor", None) or self.settings.param_scaling_floor
        )
        bounds = list(zip(self.settings.lower_bounds(), self.settings.upper_bounds()))
        atm_pinned = False
        for item in resolution.overrides:
            if not item.matched:
                # Hand pillars without listed quotes are appended after the
                # fit by ``surface_pricer.fitting.overrides.extend_and_apply``.
                continue
            slice_info = self._slice_by_date.get(to_date(item.expiry))
            if slice_info is None:
                continue
            slice_values = surface_to_slice_values(item.smile, slice_info.tau, scaling_floor)
            validate_slice_values(slice_values, bounds, slice_info.expiry)
            if slice_values:
                params = np.zeros(len(SMILE_PARAMETER_NAMES), dtype=float)
                indices: List[int] = []
                for name, value in slice_values.items():
                    index = SMILE_PARAMETER_NAMES.index(name)
                    params[index] = float(value)
                    indices.append(index)
                self._fixed_params[slice_info.expiry] = params
                self._fixed_indices[slice_info.expiry] = tuple(sorted(indices))
            if item.atm_vol is not None:
                self._fixed_atm[slice_info.expiry] = float(item.atm_vol)
                atm_pinned = True
            self._override_sources[slice_info.expiry] = "manual"
            self._report(
                "override {} | pinned {}".format(
                    slice_info.expiry.date().isoformat(),
                    ", ".join(item.fields) or "-",
                )
            )
        if resolution.synthetic_dates:
            self._report(
                "synthetic tenors added after the fit: {}".format(
                    ", ".join(value.date().isoformat() for value in resolution.synthetic_dates)
                )
            )
        if atm_pinned:
            # the arbitrage grid is built from the ATM vol, refresh it
            self._build_arb_grids()

    # ----------------------------------------------------------- scheduling
    def _slice_atm_vol(self, slice_info: SliceData) -> float:
        pinned = self._fixed_atm.get(slice_info.expiry)
        if pinned is not None:
            return float(pinned)
        return float(
            np.interp(
                slice_info.forward,
                slice_info.strikes,
                slice_info.vols,
            )
        )

    def _liquidity_schedule(self) -> List[Tuple[int, List[int]]]:
        """Liquidity ordered schedule with the Kahn topological sort."""
        count = len(self.slices)
        liquidity = np.zeros(count, dtype=float)
        for index, slice_info in enumerate(self.slices):
            atm_vol = self._slice_atm_vol(slice_info)
            sqrt_tau = np.sqrt(max(slice_info.tau, 1.0e-12))
            left_bound = np.exp(-2.0 * atm_vol * sqrt_tau)
            right_bound = 1.0 / left_bound
            grid = slice_info.forward * np.linspace(left_bound, right_bound, 51)
            buckets = max(len(grid) - 1, 1)
            filled = 0
            for bucket in range(1, len(grid)):
                inside = (slice_info.strikes >= grid[bucket - 1]) & (
                    slice_info.strikes < grid[bucket]
                )
                if np.any(inside):
                    filled += 1
            coverage = filled / buckets
            median_spread = float(np.median(slice_info.ask_vols - slice_info.bid_vols))
            if median_spread == 0.0:
                median_spread = 1.0
            liquidity[index] = coverage / median_spread

        dependencies: Dict[int, List[int]] = {}
        for index in range(count):
            deps: List[int] = []
            for left in range(index - 1, -1, -1):
                if liquidity[left] > liquidity[index]:
                    deps.append(left)
                    break
            for right in range(index + 1, count):
                if liquidity[right] > liquidity[index]:
                    deps.append(right)
                    break
            dependencies[index] = deps

        in_degree = [len(dependencies[index]) for index in range(count)]
        adjacency: Dict[int, List[int]] = {index: [] for index in range(count)}
        for index, deps in dependencies.items():
            for dep in deps:
                adjacency[dep].append(index)

        queue = [index for index in range(count) if in_degree[index] == 0]
        schedule: List[Tuple[int, List[int]]] = []
        fitted_set = set()
        while queue:
            current = queue.pop(0)
            constraints = [dep for dep in dependencies[current] if dep in fitted_set]
            schedule.append((current, constraints))
            fitted_set.add(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return schedule

    # ------------------------------------------------------------- arbitrage
    def _build_arb_grids(self) -> None:
        settings = self.settings
        for slice_info in self.slices:
            atm_vol = self._slice_atm_vol(slice_info)
            sigma = atm_vol * np.sqrt(max(slice_info.tau, 1.0e-12))
            grid = np.linspace(
                -float(settings.arb_check_std_range),
                float(settings.arb_check_std_range),
                int(settings.arb_check_points),
            )
            ln_strike = grid * sigma
            d1 = -ln_strike / max(sigma, 1.0e-8) + 0.5 * sigma
            weights = norm.pdf(d1)
            weights = weights / weights.sum()
            self._arb_grid[slice_info.expiry] = (np.exp(ln_strike), weights)

    def _build_reference_params(self) -> None:
        surface = self.settings.reference_surface
        if surface is None:
            return
        for slice_info in self.slices:
            try:
                reference_slice = surface.get_slice(
                    slice_info.expiry,
                    current_forward=slice_info.forward,
                )
            except Exception:
                continue
            self._reference_params[slice_info.expiry] = np.asarray(
                [
                    reference_slice.skew,
                    reference_slice.conv,
                    reference_slice.left_skew_1,
                    reference_slice.left_skew_2,
                    reference_slice.right_skew_1,
                    reference_slice.right_skew_2,
                ],
                dtype=float,
            )

    # ------------------------------------------------------------ fit slices
    def _make_slice(
        self,
        slice_info: SliceData,
        params: np.ndarray,
        *,
        atm_vol: Optional[float] = None,
    ) -> EDSSabrSlice:
        vol_atmf = self._slice_atm_vol(slice_info) if atm_vol is None else float(atm_vol)
        return EDSSabrSlice(
            ref_strike=slice_info.forward,
            forward=slice_info.forward,
            vol_atmf=vol_atmf,
            tau=max(slice_info.tau, 1.0e-12),
            skew=float(params[0]),
            conv=float(params[1]),
            left_skew_1=float(params[2]),
            left_skew_2=float(params[3]),
            right_skew_1=float(params[4]),
            right_skew_2=float(params[5]),
        )

    def _smooth_penalty(self, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > _EXP_OVERFLOW_LIMIT:
            value = _EXP_OVERFLOW_LIMIT
        return float(np.exp(value) - value - 1.0)

    def _calendar_penalty(
        self,
        fitted: EDSSabrSlice,
        slice_info: SliceData,
        other: EDSSabrSlice,
        *,
        direction: str,
    ) -> float:
        moneyness, grid_weights = self._arb_grid[slice_info.expiry]
        current_var = (
            fitted.get_implied_vol(moneyness * slice_info.forward) ** 2
            * slice_info.tau
        )
        other_var = other.get_implied_vol(moneyness * other.forward) ** 2 * other.tau
        if direction == "pre":
            violation = other_var - current_var
        else:
            violation = current_var - other_var
        penalty = 0.0
        for value, weight in zip(violation, grid_weights):
            penalty += self._smooth_penalty(float(value)) * float(weight)
        return penalty * float(self.settings.calendar_penalty_factor)

    def _tenor_penalty(
        self,
        other: EDSSabrSlice,
        params: np.ndarray,
        *,
        skew_factor: float,
        conv_factor: float,
        left_skew_1_factor: float,
        right_skew_1_factor: float,
    ) -> float:
        penalty = 0.0
        penalty += skew_factor * (other.skew - params[0]) ** 2
        penalty += conv_factor * (other.conv - params[1]) ** 2
        penalty += left_skew_1_factor * (other.left_skew_1 - params[2]) ** 2
        penalty += right_skew_1_factor * (other.right_skew_1 - params[4]) ** 2
        return penalty

    def _reference_penalty(self, slice_info: SliceData, params: np.ndarray) -> float:
        reference = self._reference_params.get(slice_info.expiry)
        if reference is None:
            return 0.0
        settings = self.settings
        penalty = 0.0
        penalty += settings.sticky_to_reference_skew_factor * (reference[0] - params[0]) ** 2
        penalty += settings.sticky_to_reference_conv_factor * (reference[1] - params[1]) ** 2
        penalty += (
            settings.sticky_to_reference_left_skew_1_factor * (reference[2] - params[2]) ** 2
        )
        penalty += (
            settings.sticky_to_reference_right_skew_1_factor * (reference[4] - params[4]) ** 2
        )
        return penalty

    def _slice_penalties(
        self,
        params: np.ndarray,
        slice_info: SliceData,
        pre_slice: Optional[EDSSabrSlice],
        next_slice: Optional[EDSSabrSlice],
    ) -> float:
        settings = self.settings
        fitted = self._make_slice(slice_info, params)
        fitted_vols = fitted.get_implied_vol(slice_info.strikes)
        diff = fitted_vols - slice_info.vols
        penalty = float(np.dot(diff ** 2, slice_info.weights)) * settings.mid_vol_penalty_factor

        outside = np.maximum(slice_info.bid_vols - fitted_vols, 0.0) + np.maximum(
            fitted_vols - slice_info.ask_vols, 0.0
        )
        penalty += (
            float(np.dot(outside ** 2, slice_info.weights))
            * settings.out_of_bid_ask_penalty_factor
        )

        if pre_slice is not None:
            penalty += self._calendar_penalty(
                fitted, slice_info, pre_slice, direction="pre"
            )
            penalty += self._tenor_penalty(
                pre_slice,
                params,
                skew_factor=settings.sticky_to_pre_tenor_skew_factor,
                conv_factor=settings.sticky_to_pre_tenor_conv_factor,
                left_skew_1_factor=settings.sticky_to_pre_tenor_left_skew_1_factor,
                right_skew_1_factor=settings.sticky_to_pre_tenor_right_skew_1_factor,
            )
        if next_slice is not None:
            penalty += self._calendar_penalty(
                fitted, slice_info, next_slice, direction="next"
            )
            penalty += self._tenor_penalty(
                next_slice,
                params,
                skew_factor=settings.sticky_to_next_tenor_skew_factor,
                conv_factor=settings.sticky_to_next_tenor_conv_factor,
                left_skew_1_factor=settings.sticky_to_next_tenor_left_skew_1_factor,
                right_skew_1_factor=settings.sticky_to_next_tenor_right_skew_1_factor,
            )
        penalty += self._reference_penalty(slice_info, params)
        if not np.isfinite(penalty):
            return float("inf")
        return float(penalty)

    def _initial_params(self, slice_info: SliceData) -> np.ndarray:
        reference = self._reference_params.get(slice_info.expiry)
        if reference is not None:
            return np.asarray(reference, dtype=float).copy()
        return np.zeros(len(SMILE_PARAMETER_NAMES), dtype=float)

    def _adjust_initial_guess(
        self,
        xint: np.ndarray,
        slice_info: SliceData,
        pre_slice: Optional[EDSSabrSlice],
        next_slice: Optional[EDSSabrSlice],
    ) -> np.ndarray:
        settings = self.settings
        lower, upper = settings.lower_bounds(), settings.upper_bounds()
        if settings.zero_initialization:
            xint = np.zeros_like(xint)
        else:
            try_loss = self._slice_penalties(xint, slice_info, pre_slice, next_slice)
            if not np.isfinite(try_loss) or try_loss > 0.5:
                xint = np.zeros_like(xint)

        strikes = slice_info.strikes
        vols = slice_info.vols
        forward = slice_info.forward
        if xint[0] == 0.0 or xint[1] == 0.0:
            idx_100 = int(np.searchsorted(strikes, forward))
            idx_105 = int(np.searchsorted(strikes, forward * 1.05))
            idx_95 = int(np.searchsorted(strikes, forward * 0.95))
            idx_95 = max(idx_95, 0)
            if idx_105 == 0:
                idx_105 = 2
                idx_100 = 1
            if idx_100 == 0:
                idx_100 = 1
                if idx_105 == idx_100:
                    idx_105 = idx_100 + 1
            if idx_105 == len(strikes):
                idx_105 -= 1
            if idx_100 >= idx_105:
                idx_100 = idx_105 - 1
            if idx_95 >= idx_100:
                idx_95 = idx_100 - 1
            risk_reversal = vols[idx_105] - vols[idx_95]
            if xint[0] == 0.0:
                xint[0] = risk_reversal / vols[idx_100] if risk_reversal != 0.0 else 0.1
            if xint[1] == 0.0:
                butterfly = vols[idx_95] - 2.0 * vols[idx_100] + vols[idx_105]
                if butterfly != 0.0:
                    xint[1] = butterfly / vols[idx_100]
                else:
                    xint[1] = 0.1
        return np.clip(xint, lower, upper)

    def _optimize(
        self,
        cost,
        x0: np.ndarray,
        bounds: Sequence[Tuple[float, float]],
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        settings = self.settings
        last_stats: Dict[str, Any] = {}
        for method in settings.optimizer_method_priority:
            options: Dict[str, Any] = {
                "maxiter": int(settings.max_iterations),
                "ftol": float(settings.function_tolerance),
                "disp": False,
            }
            if method == "L-BFGS-B":
                options["gtol"] = float(settings.gradient_tolerance)
            try:
                result = minimize(
                    cost,
                    np.asarray(x0, dtype=float),
                    method=method,
                    bounds=list(bounds),
                    options=options,
                )
            except Exception:
                continue
            last_stats = {
                "method": method,
                "iterations": int(getattr(result, "nit", 0) or 0),
                "function_evaluations": int(getattr(result, "nfev", 0) or 0),
                "message": str(getattr(result, "message", "")),
            }
            if result.success and np.all(np.isfinite(result.x)):
                return True, np.asarray(result.x, dtype=float), last_stats
        return False, np.asarray(x0, dtype=float), last_stats

    def _fit_one_slice(
        self,
        slice_info: SliceData,
        pre_slice: Optional[EDSSabrSlice],
        next_slice: Optional[EDSSabrSlice],
    ) -> Optional[SliceFitResult]:
        settings = self.settings
        pinned = self._fixed_params.get(slice_info.expiry)
        pinned_indices = self._fixed_indices.get(slice_info.expiry, ())
        atm_vol = self._fixed_atm.get(slice_info.expiry)

        xint = self._initial_params(slice_info)
        if pinned is not None:
            xint = np.asarray(pinned, dtype=float).copy()
        xint = self._adjust_initial_guess(xint, slice_info, pre_slice, next_slice)
        if pinned is not None:
            xint[list(pinned_indices)] = np.asarray(pinned, dtype=float)[list(pinned_indices)]

        free_indices = tuple(
            index
            for index in range(len(SMILE_PARAMETER_NAMES))
            if index not in set(pinned_indices)
        )

        if not free_indices:
            # every smile parameter is pinned: skip the optimiser entirely
            stats = {
                "method": "pinned",
                "iterations": 0,
                "function_evaluations": 0,
                "message": "all smile parameters pinned by a hand override",
            }
            self._report(
                "  {} fully pinned | optimiser skipped".format(
                    slice_info.expiry.date().isoformat()
                )
            )
            return self._build_result(slice_info, np.asarray(pinned, dtype=float), stats)

        bounds = list(zip(settings.lower_bounds(), settings.upper_bounds()))

        def expand(x_free: np.ndarray) -> np.ndarray:
            full = np.asarray(pinned, dtype=float).copy() if pinned is not None else xint.copy()
            full[list(free_indices)] = np.asarray(x_free, dtype=float)
            return full

        def cost(x_free: np.ndarray) -> float:
            return self._slice_penalties(expand(x_free), slice_info, pre_slice, next_slice)

        free_bounds = [bounds[index] for index in free_indices]
        success, x_free, stats = self._optimize(cost, xint[list(free_indices)], free_bounds)
        if not success:
            return None
        full_params = expand(x_free)
        params = np.where(
            np.abs(full_params) <= settings.param_zero_threshold, 0.0, full_params
        )
        if pinned is not None:
            # hand values must survive the small-parameter clamp untouched
            params[list(pinned_indices)] = np.asarray(pinned, dtype=float)[list(pinned_indices)]
        return self._build_result(slice_info, params, stats)

    def _build_result(
        self,
        slice_info: SliceData,
        params: np.ndarray,
        stats: Dict[str, Any],
    ) -> SliceFitResult:
        """Assemble a :class:`SliceFitResult` (pinned and fitted slices alike)."""
        fitted = self._make_slice(slice_info, params)
        fitted_vols = fitted.get_implied_vol(slice_info.strikes)
        diff = fitted_vols - slice_info.vols
        outside = np.maximum(slice_info.bid_vols - fitted_vols, 0.0) + np.maximum(
            fitted_vols - slice_info.ask_vols, 0.0
        )
        out_of_bid_ask = {
            float(strike): float(value)
            for strike, value in zip(slice_info.strikes, outside)
            if value > 0.0
        }
        source = self._override_sources.get(slice_info.expiry, "")
        return SliceFitResult(
            slice_info=slice_info,
            params=np.asarray(params, dtype=float),
            fitted=fitted,
            atm_vol=self._slice_atm_vol(slice_info),
            rmse=float(np.sqrt(np.mean(diff ** 2))),
            weighted_rmse=float(np.sqrt(np.mean(diff ** 2 * slice_info.weights))),
            out_of_bid_ask=out_of_bid_ask,
            converged=True,
            message=str(stats.get("message", "")),
            optimizer_method=str(stats.get("method", "")),
            iterations=int(stats.get("iterations", 0)),
            function_evaluations=int(stats.get("function_evaluations", 0)),
            is_override=bool(source),
            override_source=source,
        )

    # ------------------------------------------------------------- surface
    def _build_surface(
        self,
        market: MarketState,
        results: Sequence[SliceFitResult],
    ) -> EDSSabrSurface:
        settings = self.settings
        expiry_dates = [result.expiry for result in results]
        atm_vols = [result.atm_vol for result in results]

        def scaled(index: int) -> List[float]:
            values = []
            for result in results:
                scale = max(
                    float(settings.param_scaling_floor),
                    float(np.sqrt(max(result.tau, 1.0e-12))),
                )
                values.append(float(result.params[index] * scale))
            return values

        return EDSSabrSurface(
            init_date=market.valuation_date,
            init_spot=market.spot,
            expiry_dates=expiry_dates,
            atm_vols=atm_vols,
            skews=scaled(0),
            convs=scaled(1),
            left_skews_1=scaled(2),
            left_skews_2=scaled(3),
            right_skews_1=scaled(4),
            right_skews_2=scaled(5),
            stickiness_ratio=0.0,
            calendar=market.calendar,
            trading_days_per_year=settings.trading_days_per_year,
            holiday_weight=settings.holiday_weight,
        )


__all__ = ["EDSSabrFitter", "SliceFitResult"]
