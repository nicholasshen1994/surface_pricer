"""One-dimensional PDE engine for autocallables (standard snowball).

Backward induction in log-spot space, sharing every input with the MC engine -
the Dupire local-vol coefficients, the effective (already shifted) barriers and
the cash-flow rules - so a cross-check compares the methods rather than two
setups:

* the time grid is forced through every observation date, with
  ``RiskSettings.pde_steps_per_observation`` equal vol-time sub-steps per window
  - the same time-discretisation concept as the MC engine;
* knock-out is a value condition at each observation, smoothed with the same
  indicator the MC engine uses;
* knock-in is a **two-state** induction: ``V_noki`` and ``V_ki`` are both
  carried backwards and mixed at each observation with the knock-in
  probability - the PDE counterpart of edslib's auxiliary contract;
* coupons ride inside the knock-out and expiry settlements, so no separate
  accrual pass is needed;
* barriers are pinned as exact grid nodes (``build_log_grid``), which removes
  the classic "barrier between nodes" PDE bias.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from ...models.localvol import (
    LOCAL_VOL_NODES,
    LOCAL_VOL_SLICE_STEP,
    DupireLocalVol,
    LocalVolCache,
)
from ...numerics.fdm import bsm_coefficients, build_log_grid, theta_step
from ...results import PricingResult, RiskSettings
from ...risk.buckets import bucket_greeks, bucket_grid_info
from ...risk.diff import GREEK_CONVENTION, apply_greeks, bump_greeks
from .cashflows import expiry_cash_flow, ko_cash_flow, rebate_cash_flow
from .contract import AutocallContract
from .grid import build_time_grid, smooth_indicator
from .schedule import AutocallSchedule, build_schedule, resolve_trigger_basis


class AutocallPDE:
    """PDE pricer implementing :class:`ExoticPricer`."""

    def __init__(
        self,
        *,
        nodes: Optional[int] = None,
        theta: Optional[float] = None,
        sigmas: float = 6.0,
        local_vol_nodes: int = LOCAL_VOL_NODES,
        local_vol_step: float = LOCAL_VOL_SLICE_STEP,
        local_vol_cache: Optional[LocalVolCache] = None,
        smooth: Optional[bool] = None,
        smooth_width: Optional[float] = None,
        smooth_floor: Optional[float] = None,
        ko_shift: Optional[Any] = None,
        ki_shift: Optional[Any] = None,
        shift_config: Optional[Any] = None,
        contractual: bool = False,
        trigger_basis: str = "contractual",
    ):
        self.nodes = None if nodes is None else max(int(nodes), 5)
        self.theta = None if theta is None else float(theta)
        self.sigmas = float(sigmas)
        self.local_vol_nodes = max(int(local_vol_nodes), 5)
        # vol-time spacing of the local-vol table (0 -> build a slice per node)
        self.local_vol_step = max(float(local_vol_step), 0.0)
        # A caller-supplied table provider (``make_table_cache``): it owns the
        # discretisation and, with a store wired in, the on-disk cache.  Handing one
        # in is what makes a bump run - or a whole spot ladder - build a single
        # table instead of one per market.
        self.local_vol_cache = local_vol_cache
        self.smooth = smooth
        self.smooth_width = smooth_width
        self.smooth_floor = smooth_floor
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
        npv, info = self._value(schedule, market, settings)
        return self._result(npv, schedule, market, info, settings)

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
        # One grid and one local-vol anchor for the whole risk run: neither the
        # discretisation nor the model coefficients may follow the bumped spot,
        # or the bumps cancel against the shift instead of measuring it.
        tables = self.local_vol_cache or LocalVolCache(
            nodes=self.local_vol_nodes,
            slice_step=self.local_vol_step,
            spot_anchor=float(market.spot),
        )
        base_npv, info = self._value(schedule, market, settings, tables=tables)
        x_grid = info.get("x_grid")

        def value(bumped_market: Any) -> float:
            bumped_schedule = schedule.rebased(bumped_market)
            return self._value(
                bumped_schedule,
                bumped_market,
                settings,
                x_grid=x_grid,
                tables=tables,
            )[0]

        greeks = bump_greeks(value, market, settings, base=base_npv)
        buckets = bucket_greeks(
            value,
            market,
            settings,
            delta_cash=greeks["delta_cash"],
            horizon=schedule.expiry_payment_date,
        )
        result = apply_greeks(
            self._result(base_npv, schedule, market, info, settings), greeks
        )
        result.bucketed_vega = buckets["bucketed_vega"]
        result.bucketed_rhoq = buckets["bucketed_rhoq"]
        result.bucketed_rho = buckets["bucketed_rho"]
        result.bucketed_delta = buckets["bucketed_delta"]
        result.metadata["bucket_grid"] = bucket_grid_info(
            market, settings, horizon=schedule.expiry_payment_date
        )
        result.metadata["greek_convention"] = dict(GREEK_CONVENTION)
        return result

    # ------------------------------------------------------------ internals
    def _effective_settings(self, settings: Optional[RiskSettings]) -> RiskSettings:
        settings = settings or RiskSettings()
        changes = {}
        if self.nodes is not None:
            changes["pde_nodes"] = int(self.nodes)
        if self.theta is not None:
            changes["pde_theta"] = float(self.theta)
        if self.smooth is not None:
            changes["barrier_smooth"] = bool(self.smooth)
        if self.smooth_width is not None:
            changes["barrier_smooth_width"] = float(self.smooth_width)
        if self.smooth_floor is not None:
            changes["barrier_smooth_floor"] = float(self.smooth_floor)
        return replace(settings, **changes) if changes else settings

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

    def _space_grid(
        self, schedule: AutocallSchedule, market: Any, nodes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        expiry = schedule.expiry_date
        forward = float(market.forward(expiry))
        vol_time = max(float(schedule.expiry_vol_time), 1e-8)
        atm = 0.2
        if getattr(market, "surface", None) is not None:
            atm = max(float(market.surface.get_atm_vol(vol_time)), 1e-4)
        spread = self.sigmas * atm * math.sqrt(vol_time)
        low = forward * math.exp(-spread)
        high = forward * math.exp(spread)

        levels = [
            float(value)
            for value in list(schedule.ko_levels) + list(schedule.ki_levels)
            if float(value) > 0.0
        ]
        if levels:
            low = min(low, min(levels) * 0.95)
            high = max(high, max(levels) * 1.05)
        low = max(low, 1e-8)

        crucial = levels + [float(schedule.spot0)]
        x_grid = build_log_grid(low, high, max(int(nodes), 5), crucial_levels=crucial)
        return x_grid, np.exp(x_grid)

    def _local_table(
        self,
        market: Any,
        grid: Any,
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
        x_grid: Optional[np.ndarray] = None,
        spot_anchor: Optional[float] = None,
        tables: Optional[LocalVolCache] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve backwards on ``x_grid`` (rebuilt only when not supplied).

        The grid is a pure discretisation of the numerical scheme, so the
        Greeks **pass the base grid in** for every bumped market: rebuilding it
        around each bumped forward would drag the nodes along with the spot and
        cancel most of the sensitivity (bump delta ~1 against a local grid
        slope of ~56).  Barriers stay put - they are anchored on the start spot
        by :func:`build_schedule` - so a fixed grid leaves the spot moving
        *relative to* the barriers, which is exactly what the Greeks describe.
        """
        if schedule.is_settled:
            npv = float(schedule.knocked_out_cash) * float(
                schedule.knocked_out_discount_factor
            )
            return npv, {"steps": 0, "nodes": 0, "knocked_out": True}

        theta = float(settings.pde_theta)
        grid = build_time_grid(
            schedule,
            market,
            steps_per_observation=settings.pde_steps_per_observation,
        )
        local = self._local_table(market, grid, spot_anchor, tables)
        if x_grid is None:
            x_grid, spots = self._space_grid(schedule, market, settings.pde_nodes)
        else:
            x_grid = np.asarray(x_grid, dtype=float)
            spots = np.exp(x_grid)
        size = x_grid.size

        notional = float(schedule.notional)
        # the "neither knocked out nor knocked in" settlement, already resolved
        principal_coupon = rebate_cash_flow(schedule)
        put_payoff = expiry_cash_flow(schedule, spots, True)
        low_payoff = notional * max(
            1.0 - float(schedule.ki_gearing), float(schedule.protected_principal)
        )

        discount_to_expiry = float(schedule.expiry_discount_factor)
        # The lower boundary is "knocked in" only when it really sits below the
        # knock-in level: a grid that stops above the KI can still expire
        # untouched there and collect principal plus coupon.
        ki_floor = min(
            [float(value) for value in schedule.ki_levels if float(value) > 0.0],
            default=0.0,
        )
        if ki_floor > 0.0 and float(spots[0]) <= ki_floor:
            low_value_at_expiry = low_payoff * discount_to_expiry
        else:
            low_value_at_expiry = principal_coupon * discount_to_expiry
        no_ko_value_at_expiry = principal_coupon * discount_to_expiry
        # knock-out settlement discounted to its payment date (valuation-date
        # present value); the engine divides by the discount factor of the
        # observation itself so the -rV term does the remaining discounting
        ko_value_at_payment = [
            ko_cash_flow(schedule, index) * float(df)
            for index, df in enumerate(schedule.discount_factors)
        ]

        # Terminal conditions.  When the expiry is itself an observation date
        # (the usual snowball layout) the final knock-in / knock-out decision
        # happens at expiry too, so both state curves must carry it; otherwise
        # the last observation was already applied on the previous grid step
        # and the expiry simply pays.
        last_observation = (
            schedule.observation_dates[-1] if schedule.observation_dates else None
        )
        observation_at_expiry = (
            last_observation is not None and last_observation == schedule.expiry_date
        )
        if observation_at_expiry:
            ko_last = schedule.ko_levels[-1]
            ki_last = schedule.ki_levels[-1]
            terminal_ko = smooth_indicator(
                spots - ko_last,
                ko_last,
                enabled=settings.barrier_smooth,
                width=settings.barrier_smooth_width,
                floor=settings.barrier_smooth_floor,
            )
            if ki_last > 0.0:
                terminal_ki = smooth_indicator(
                    ki_last - spots,
                    ki_last,
                    enabled=settings.barrier_smooth,
                    width=settings.barrier_smooth_width,
                    floor=settings.barrier_smooth_floor,
                )
            else:
                terminal_ki = np.zeros(size, dtype=float)
            value_noki = (1.0 - terminal_ki) * principal_coupon + terminal_ki * put_payoff
            value_ki = terminal_ko * principal_coupon + (1.0 - terminal_ko) * put_payoff
        else:
            value_noki = np.full(size, principal_coupon, dtype=float)
            value_ki = put_payoff

        events = {step: index for index, step in enumerate(grid.observation_steps)}
        if grid.ki_steps:
            ki_events = {step: index for index, step in enumerate(grid.ki_steps)}
            ki_levels = schedule.ki_monitor_levels
        else:  # a grid built outside build_time_grid: test together with the KO
            ki_events = events
            ki_levels = schedule.ki_levels
        # the first observation at or after each time node -> upper boundary
        upcoming: list = [None] * len(grid.dates)
        current: Optional[int] = None
        for step in range(len(grid.dates) - 1, -1, -1):
            if step in events:
                current = events[step]
            upcoming[step] = current

        for step in range(len(grid.dates) - 1, 0, -1):
            when = grid.dates[step - 1]
            base_df = float(market.discount_factor(when))
            if base_df <= 0.0:  # pragma: no cover - defensive
                base_df = 1.0
            dt = grid.vol_times[step] - grid.vol_times[step - 1]
            if dt > 0.0:
                sigma = local.local_vols(when, spots)
                # The drift has to reproduce the curve forward exactly: the MC
                # engine steps by F_{i+1}/F_i, so the PDE takes the same ratio,
                # and discounting uses the interval's average short rate (zero
                # rates would bias a sloped curve).
                mu = math.log(grid.forwards[step] / grid.forwards[step - 1]) / dt
                df_later = float(market.discount_factor(grid.dates[step]))
                df_earlier = float(market.discount_factor(when))
                if df_later > 0.0 and df_earlier > 0.0:
                    # DF decreases with maturity, so the average short rate is
                    # -ln(DF_later / DF_earlier) / dt
                    rate = -math.log(df_later / df_earlier) / dt
                else:  # pragma: no cover - defensive
                    rate = 0.0
                lower, diag, upper = bsm_coefficients(
                    x_grid, sigma, rate, rate - mu
                )

                index = upcoming[step - 1]
                if index is None:
                    high_boundary = no_ko_value_at_expiry / base_df
                else:
                    high_boundary = ko_value_at_payment[index] / base_df

                value_noki = theta_step(
                    lower, diag, upper, value_noki,
                    dt=dt, theta=theta,
                    low_boundary=low_value_at_expiry / base_df,
                    high_boundary=high_boundary,
                )
                value_ki = theta_step(
                    lower, diag, upper, value_ki,
                    dt=dt, theta=theta,
                    low_boundary=low_value_at_expiry / base_df,
                    high_boundary=high_boundary,
                )

            # knock-in: every monitoring date carries its own constraint (daily
            # business days by default, so each node can test the barrier)
            ki_index = ki_events.get(step - 1)
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
                    value_noki = ki_prob * value_ki + (1.0 - ki_prob) * value_noki

            ko_index = events.get(step - 1)
            if ko_index is None:
                continue
            ko_level = schedule.ko_levels[ko_index]
            ko_prob = smooth_indicator(
                spots - ko_level,
                ko_level,
                enabled=settings.barrier_smooth,
                width=settings.barrier_smooth_width,
                floor=settings.barrier_smooth_floor,
            )
            cash = ko_value_at_payment[ko_index] / base_df

            # The knock-out stays live after the knock-in - the KI state is not
            # absorbing, so a knocked-in path that later reaches a knock-out level
            # still knocks out and takes the coupon.  On a tied date the knock-out
            # wins, which is why it is applied *after* the knock-in transfer.
            value_noki = ko_prob * cash + (1.0 - ko_prob) * value_noki
            value_ki = ko_prob * cash + (1.0 - ko_prob) * value_ki

        curve = value_ki if schedule.knocked_in_before else value_noki
        target = math.log(float(market.spot))
        npv = float(np.interp(target, x_grid, curve))

        # local derivative from the solved nodes (the same second-order stencil
        # as the scheme).  A global spline is unreliable here: with a barrier
        # pinned next to the spot the curve has a kink and the spline rings.
        index = int(np.searchsorted(x_grid, target))
        index = min(max(index, 1), size - 2)
        span = x_grid[index + 1] - x_grid[index - 1]
        if span > 0.0:
            grid_delta = (
                (curve[index + 1] - curve[index - 1]) / span / float(market.spot)
            )
        else:  # pragma: no cover - degenerate grid
            grid_delta = 0.0

        info = {
            "steps": int(grid.n_steps),
            "nodes": int(size),
            "theta": theta,
            "grid_delta": grid_delta,
            "knocked_out": False,
            "x_grid": x_grid,  # reused by the Greeks, not part of the report
        }
        return npv, info

    def _result(
        self,
        npv: float,
        schedule: AutocallSchedule,
        market: Any,
        info: Dict[str, Any],
        settings: RiskSettings,
    ) -> PricingResult:
        expiry = schedule.expiry_date
        metadata: Dict[str, Any] = {
            "method": "pde",
            "steps": int(info.get("steps", 0)),
            "nodes": int(info.get("nodes", 0)),
            "theta": float(info.get("theta", settings.pde_theta)),
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
        if "grid_delta" in info:
            metadata["grid_delta"] = float(info["grid_delta"])
        if info.get("knocked_out") or schedule.is_settled:
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


__all__ = ["AutocallPDE"]
