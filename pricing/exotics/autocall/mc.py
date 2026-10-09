"""Monte Carlo engine for autocallables (standard snowball).

Stability mechanisms (ported from the edslib risk stack):

* **common random numbers** - the normal matrix comes from a deterministic
  Sobol sequence (fixed seed) and is cached per *interval*, so every Greek bump
  differences paths that are bit-for-bit the same - including the theta date
  shift, whose daily grid drops its leading interval;
* **barrier smoothing** - the discrete KO/KI indicators are smoothed over a
  narrow band (:func:`~surface_pricer.pricing.exotics.autocall.smooth_indicator`,
  shared with the PDE engine) so the payoff is continuous in the spot and
  delta/gamma stop being dominated by paths crossing the barrier;
* **theta alignment** - the cached matrix is sliced by interval, so a
  valuation-date bump reuses the same numbers for the days it still has and the
  theta difference compares like with like.

The martingale-preserving stepping uses the curve forwards::

    S_{i+1} = S_i * (F_{i+1} / F_i) * exp(-0.5 sigma_i^2 dt + sigma_i sqrt(dt) Z)

with ``sigma_i`` the Dupire local vol at ``(t_i, S_i)`` - the same coefficients
the PDE engine consumes, which is what makes the two methods comparable.  The
forward ratio is applied on **every** step (some grid nodes carry no vol time
but the forward still moves).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

try:  # scipy >= 1.7
    from scipy.stats import qmc
except ImportError:  # pragma: no cover - legacy scipy fallback
    qmc = None

from ....core.daycount import to_date
from ...models.localvol import (
    LOCAL_VOL_NODES,
    LOCAL_VOL_SLICE_STEP,
    DupireLocalVol,
    LocalVolCache,
)
from ...results import PricingResult, RiskSettings
from ...risk.buckets import bucket_greeks, bucket_grid_info
from ...risk.diff import GREEK_CONVENTION, apply_greeks, bump_greeks
from .cashflows import expiry_cash_flow, ko_cash_flow
from .contract import AutocallContract
from .grid import TimeGrid, build_time_grid, smooth_indicator
from .schedule import AutocallSchedule, build_schedule, resolve_trigger_basis


def _normals(n_steps: int, n_paths: int, seed: int) -> np.ndarray:
    """Deterministic standard normals, shape ``(n_paths, n_steps)``.

    ``scramble=True`` with a fixed seed keeps the sequence reproducible while
    smoothing the Sobol net; the fallback keeps old scipy usable without losing
    reproducibility.
    """
    if qmc is not None:
        sampler = qmc.Sobol(d=max(n_steps, 1), scramble=True, seed=int(seed))
        uniform = sampler.random(n_paths)
    else:  # pragma: no cover - legacy scipy fallback
        generator = np.random.default_rng(int(seed))
        uniform = generator.random((n_paths, max(n_steps, 1)))
    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniform)


def _suffix_offset(dates: Tuple[Any, ...], target: Tuple[Any, ...]) -> Optional[int]:
    """Leading intervals of ``dates`` that ``target`` does not have (``None`` if unrelated).

    A valuation-date bump drops leading intervals, so the shifted grid is a
    suffix of the original one; anything else is a different grid.  Nodes are
    compared by **date**: the first node carries the valuation timestamp
    (``15:34:28``) while the monitoring nodes are midnight stamps, so the same
    calendar date would otherwise never match.
    """
    offset = len(dates) - len(target)
    if offset < 0:
        return None
    if tuple(to_date(item) for item in dates[offset:]) != tuple(
        to_date(item) for item in target
    ):
        return None
    return offset


class AutocallMonteCarlo:
    """Monte Carlo pricer implementing :class:`ExoticPricer`.

    Instances are cheap; all market data comes in through ``price`` /
    ``greeks`` so a bumped market simply rebuilds the local-vol surface.
    """

    def __init__(
        self,
        *,
        paths: Optional[int] = None,
        seed: Optional[int] = None,
        steps_per_observation: Optional[int] = None,
        smooth: Optional[bool] = None,
        smooth_width: Optional[float] = None,
        smooth_floor: Optional[float] = None,
        local_vol_nodes: int = LOCAL_VOL_NODES,
        local_vol_step: float = LOCAL_VOL_SLICE_STEP,
        local_vol_cache: Optional[LocalVolCache] = None,
        ko_shift: Optional[Any] = None,
        ki_shift: Optional[Any] = None,
        shift_config: Optional[Any] = None,
        contractual: bool = False,
        trigger_basis: str = "contractual",
    ):
        self.paths = None if paths is None else int(paths)
        self.seed = None if seed is None else int(seed)
        self.steps_per_observation = (
            None if steps_per_observation is None else int(steps_per_observation)
        )
        self.smooth = smooth
        self.smooth_width = smooth_width
        self.smooth_floor = smooth_floor
        self.local_vol_nodes = max(int(local_vol_nodes), 5)
        # vol-time spacing of the local-vol table (0 -> build a slice per node)
        self.local_vol_step = max(float(local_vol_step), 0.0)
        # Caller-supplied table provider (``make_table_cache``): it owns the
        # discretisation and the on-disk cache, and one of them serves every market
        # of a bump run or a spot ladder.
        self.local_vol_cache = local_vol_cache
        self.ko_shift = ko_shift
        self.ki_shift = ki_shift
        self.shift_config = shift_config
        self.contractual = bool(contractual)
        self.trigger_basis = resolve_trigger_basis(trigger_basis)

    # ------------------------------------------------------------- public API
    def price(
        self,
        contract: AutocallContract,
        market: Any,
        settings: Optional[RiskSettings] = None,
    ) -> PricingResult:
        """Value a contract (its resolved schedule is built first)."""
        return self.price_schedule(self._schedule(contract, market), market, settings)

    def price_schedule(
        self,
        schedule: AutocallSchedule,
        market: Any,
        settings: Optional[RiskSettings] = None,
    ) -> PricingResult:
        """Value a resolved schedule - the JSON layer's entry point.

        ``AutocallSchedule.from_dict(payload, market)`` turns a hand-editable
        JSON contract into the object this expects.
        """
        settings = self._effective_settings(settings)
        paths, seed, steps = self._sizes(settings)
        npv, error, grid = self._value(schedule, market, settings, {})
        return self._result(npv, error, schedule, market, paths, seed, grid, settings)

    def greeks(
        self,
        contract: AutocallContract,
        market: Any,
        settings: Optional[RiskSettings] = None,
    ) -> PricingResult:
        """Greeks of a contract (its resolved schedule is built first)."""
        return self.greeks_schedule(self._schedule(contract, market), market, settings)

    def greeks_schedule(
        self,
        schedule: AutocallSchedule,
        market: Any,
        settings: Optional[RiskSettings] = None,
    ) -> PricingResult:
        """Greeks of a resolved schedule - the JSON layer's risk entry point.

        A bump only changes the market, so the bumped valuations reuse the same
        contractual terms through :meth:`AutocallSchedule.rebased`.  A **settled**
        trade (knocked out) is one discounted cash flow: the NPV is still reported,
        every Greek is zero, and no bump has to be paid for.
        """
        settings = self._effective_settings(settings)
        if schedule.is_settled:
            # one discounted cash flow: the NPV reports, every Greek the caller
            # asked for is zero, and not a single bump is paid for
            return self.price_schedule(schedule, market, settings).zero_greeks(
                settings.greeks
            )
        # bumped valuations may use fewer paths: the common random numbers keep
        # the finite differences paired, so the extra noise cancels in the
        # difference instead of biasing it
        paths, seed, steps = self._sizes(
            settings, override_paths=settings.mc_greek_paths
        )
        normals: Dict[Any, np.ndarray] = {}
        # the local-vol table is a model coefficient: pin its anchor to the base
        # spot so a spot bump only moves the query point, not the surface, and
        # build it once per (surface, time grid) instead of once per bump
        tables = self.local_vol_cache or LocalVolCache(
            nodes=self.local_vol_nodes,
            slice_step=self.local_vol_step,
            spot_anchor=float(market.spot),
        )

        base_npv, base_error, grid = self._value(
            schedule, market, settings, normals, tables=tables
        )

        def value(bumped_market: Any) -> float:
            bumped_schedule = schedule.rebased(bumped_market)
            npv, _, _ = self._value(
                bumped_schedule,
                bumped_market,
                settings,
                normals,
                tables=tables,
            )
            return npv

        greeks = bump_greeks(value, market, settings, base=base_npv)
        buckets = bucket_greeks(
            value,
            market,
            settings,
            delta_cash=greeks["delta_cash"],
            horizon=schedule.expiry_payment_date,
        )

        result = apply_greeks(
            self._result(
                base_npv, base_error, schedule, market, paths, seed, grid, settings
            ),
            greeks,
        )
        result.bucketed_vega = buckets["bucketed_vega"]
        result.bucketed_rhoq = buckets["bucketed_rhoq"]
        result.bucketed_rho = buckets["bucketed_rho"]
        result.bucketed_delta = buckets["bucketed_delta"]
        result.metadata["bucket_grid"] = bucket_grid_info(
            market, settings, horizon=schedule.expiry_payment_date
        )
        result.metadata["greek_convention"] = dict(
            GREEK_CONVENTION,
            delta="bump-and-revalue dNPV/dSpot (common random numbers)",
            theta="valuation date +1D (same normals when the grid is unchanged)",
        )
        return result

    # ------------------------------------------------------------ internals
    def _effective_settings(self, settings: Optional[RiskSettings]) -> RiskSettings:
        settings = settings or RiskSettings()
        changes = {}
        if self.smooth is not None:
            changes["barrier_smooth"] = bool(self.smooth)
        if self.smooth_width is not None:
            changes["barrier_smooth_width"] = float(self.smooth_width)
        if self.smooth_floor is not None:
            changes["barrier_smooth_floor"] = float(self.smooth_floor)
        return replace(settings, **changes) if changes else settings

    def _sizes(
        self, settings: RiskSettings, override_paths: Optional[int] = None
    ) -> Tuple[int, int, int]:
        if self.paths is not None:
            requested = int(self.paths)
        elif override_paths is not None:
            requested = int(override_paths)
        else:
            requested = int(settings.mc_paths)
        requested = max(requested, 2)
        # Sobol nets are balanced on powers of two - round up, keep the request
        effective = 1 << max(int(math.ceil(math.log2(requested))), 1)
        seed = int(self.seed if self.seed is not None else settings.mc_seed)
        steps = int(
            self.steps_per_observation
            if self.steps_per_observation is not None
            else settings.mc_steps_per_observation
        )
        return effective, seed, max(steps, 1)

    def _schedule(
        self, contract: AutocallContract, market: Any
    ) -> AutocallSchedule:
        return build_schedule(
            contract,
            market,
            ko_shift=self.ko_shift,
            ki_shift=self.ki_shift,
            shift_config=self.shift_config,
            contractual=self.contractual,
            trigger_basis=self.trigger_basis,
        )

    def _local_table(
        self,
        market: Any,
        grid: TimeGrid,
        spot_anchor: Optional[float],
        tables: Optional[LocalVolCache],
    ) -> DupireLocalVol:
        """Local-vol coefficients for ``grid``, cached across the risk run."""
        cache = tables if tables is not None else self.local_vol_cache
        if cache is not None:
            return cache.table(market, grid.dates)
        local = DupireLocalVol(
            market,
            nodes=self.local_vol_nodes,
            spot_anchor=spot_anchor,
            slice_step=self.local_vol_step,
        )
        local.prepare(grid.dates)
        return local

    def _value(
        self,
        schedule: AutocallSchedule,
        market: Any,
        settings: RiskSettings,
        normals: Dict[Any, np.ndarray],
        spot_anchor: Optional[float] = None,
        tables: Optional[LocalVolCache] = None,
    ) -> Tuple[float, float, Optional[TimeGrid]]:
        if schedule.is_settled:
            npv = float(schedule.knocked_out_cash) * float(
                schedule.knocked_out_discount_factor
            )
            return npv, 0.0, None

        paths, seed, steps = self._sizes(settings)
        grid = build_time_grid(schedule, market, steps_per_observation=steps)
        z = self._normals_for(grid, paths, seed, normals)

        npv, error = self._simulate(
            schedule,
            market,
            grid,
            z,
            settings,
            spot_anchor=spot_anchor,
            tables=tables,
        )
        return npv, error, grid

    def _normals_for(
        self,
        grid: TimeGrid,
        paths: int,
        seed: int,
        cache: Dict[str, Any],
    ) -> np.ndarray:
        """Sobol normals aligned by **interval**, not by column position.

        Every Greek bump reuses one draw, but a valuation-date bump rebuilds the
        grid: with daily knock-in monitoring the day-0 interval disappears, so
        the shifted grid has one step less.  Handing both the same matrix from
        column zero would apply the base day-0 shock to the day-1 interval, and
        drawing a fresh matrix would leave theta unpaired - its two valuations
        would then carry independent standard errors (~700 on a 1M notional at
        65k paths), and the difference would measure that noise instead of the
        day roll.  The cache keeps the interval order next to the matrix and
        returns the slice that lines the intervals up.
        """
        entry = cache.get("normals")
        if entry is not None:
            dates, matrix = entry
            offset = _suffix_offset(dates, grid.dates)
            if offset is not None and matrix.shape[0] == paths:
                return matrix[:, offset : offset + grid.n_steps]
        matrix = _normals(grid.n_steps, paths, seed)
        cache["normals"] = (grid.dates, matrix)
        return matrix

    def _simulate(
        self,
        schedule: AutocallSchedule,
        market: Any,
        grid: TimeGrid,
        z: np.ndarray,
        settings: RiskSettings,
        spot_anchor: Optional[float] = None,
        tables: Optional[LocalVolCache] = None,
    ) -> Tuple[float, float]:
        n_paths = z.shape[0]
        local = self._local_table(market, grid, spot_anchor, tables)

        spots = np.full(n_paths, float(market.spot))
        survival = np.ones(n_paths)
        no_knock_in = np.ones(n_paths)
        total = np.zeros(n_paths)

        ko_events = {step: index for index, step in enumerate(grid.observation_steps)}
        if grid.ki_steps:
            ki_events = {step: index for index, step in enumerate(grid.ki_steps)}
            ki_levels = schedule.ki_monitor_levels
        else:  # a grid built outside build_time_grid: test together with the KO
            ki_events = ko_events
            ki_levels = schedule.ki_levels

        for step in range(1, len(grid.dates)):
            dt = grid.vol_times[step] - grid.vol_times[step - 1]
            ratio = grid.forwards[step] / grid.forwards[step - 1]
            if dt > 0.0:
                sigma = local.local_vols(grid.dates[step - 1], spots)
                spots = spots * ratio * np.exp(
                    -0.5 * sigma * sigma * dt + sigma * math.sqrt(dt) * z[:, step - 1]
                )
            else:
                spots = spots * ratio

            # knock-in first: it is observed on every monitoring date (daily by
            # default), so a path that dips between two knock-out dates - or
            # between two knock-out and expiry - still knocks in
            ki_index = ki_events.get(step)
            if ki_index is not None:
                ki_level = ki_levels[ki_index]
                if ki_level > 0.0:
                    ki_prob = smooth_indicator(
                        ki_level - spots,
                        ki_level,
                        enabled=settings.barrier_smooth,
                        width=settings.barrier_smooth_width,
                        floor=settings.barrier_smooth_floor,
                    )
                    no_knock_in *= 1.0 - ki_prob

            ko_index = ko_events.get(step)
            if ko_index is None:
                continue
            # The knock-out stays live after the knock-in (the KI state is not
            # absorbing): ``survival`` is unconditional, so a knocked-in path that
            # reaches a knock-out level still takes the coupon.  On a tied date the
            # knock-out wins - applied after the knock-in above.
            ko_level = schedule.ko_levels[ko_index]
            ko_prob = smooth_indicator(
                spots - ko_level,
                ko_level,
                enabled=settings.barrier_smooth,
                width=settings.barrier_smooth_width,
                floor=settings.barrier_smooth_floor,
            )
            cash = ko_cash_flow(schedule, ko_index) * schedule.discount_factors[ko_index]
            total += survival * ko_prob * cash
            survival *= 1.0 - ko_prob

        knocked_in_prob = 1.0 - no_knock_in
        without = expiry_cash_flow(schedule, spots, False)
        with_ki = expiry_cash_flow(schedule, spots, True)
        total += (
            survival
            * (knocked_in_prob * with_ki + (1.0 - knocked_in_prob) * without)
            * schedule.expiry_discount_factor
        )

        npv = float(total.mean())
        error = float(total.std(ddof=1) / math.sqrt(n_paths)) if n_paths > 1 else 0.0
        return npv, error

    def _result(
        self,
        npv: float,
        error: float,
        schedule: AutocallSchedule,
        market: Any,
        paths: int,
        seed: int,
        grid: Optional[TimeGrid],
        settings: RiskSettings,
    ) -> PricingResult:
        expiry = schedule.expiry_date
        metadata: Dict[str, Any] = {
            "method": "monte_carlo",
            "paths": int(paths),
            "seed": int(seed),
            "steps": int(grid.n_steps) if grid is not None else 0,
            "std_error": float(error),
            "spot0": float(schedule.spot0),
            "anchored_on": schedule.anchored_on,
            "observation_dates": [
                day.date().isoformat() for day in schedule.observation_dates
            ],
            "ko_levels": [float(value) for value in schedule.ko_levels],
            "ki_levels": [float(value) for value in schedule.ki_levels],
            "ko_shift": schedule.ko_shift.describe(),
            "ki_shift": schedule.ki_shift.describe(),
            "shift_notes": list(schedule.notes),
            "barrier_smooth": bool(settings.barrier_smooth),
        }
        if schedule.is_settled:
            metadata["knocked_out"] = True
            metadata["knocked_out_index"] = schedule.knocked_out_index
        return PricingResult(
            npv=float(npv),
            forward=float(market.forward(expiry)),
            discount_factor=float(market.discount_factor(expiry)),
            implied_vol=0.0,
            strike=0.0,
            year_fraction=float(market.year_fraction(expiry)),
            metadata=metadata,
        )


__all__ = ["AutocallMonteCarlo"]
