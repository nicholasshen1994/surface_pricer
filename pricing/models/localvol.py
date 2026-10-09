"""Dupire local volatility implied by the fitted EDS SABR surface.

This is the **single source of coefficients** for both exotic engines: the MC
path evolution and the PDE coefficients all come from
:meth:`DupireLocalVol.local_vols`, so the two methods price the same model.

The formula follows the strike form of Gatheral's Dupire equation (mirroring
``standalone_localvol/src/localvol/localvol/dupire.py``)::

    sigma_lv^2 = (dw/dT) / [ (1 + K d1 sqrt(t) dsigma/dK)^2
                             + sigma t K^2 (d2sigma/dK2 - d1 sqrt(t) (dsigma/dK)^2) ]

with total variance ``w = sigma_iv^2 * t``, ``d1 = (ln(F/K) + sigma^2 t / 2) /
(sigma sqrt(t))`` and ``dw/dT`` taken at **fixed log-moneyness** (strikes are
rescaled by ``F(T_up) / F(T_dn)``), exactly like the reference implementation.

Robustness: ``sigma_lv`` is floored at ``floor`` and capped at ``cap_multiple``
times the largest ATM vol, and every slice reports arbitrage diagnostics
(calendar / butterfly / negative variance / clipping ratios) instead of
failing silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
import bisect
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core.daycount import DateLike, to_date, to_datetime
from ...core.market import MarketState

#: The table's discretisation.  They are module-level because the **cache** owns
#: them: a cache (in memory or on disk) built with one discretisation must not
#: serve a table built with another, and the apps create the cache while the
#: engines keep their defaults from here.
LOCAL_VOL_NODES = 61
LOCAL_VOL_SLICE_STEP = 0.01


@dataclass(frozen=True)
class LocalVolDiagnostics:
    """Arbitrage / robustness counters for one local-vol slice."""

    vol_time: float
    atm_vol: float
    calendar_arb_ratio: float
    butterfly_arb_ratio: float
    negative_ratio: float
    clipped_ratio: float
    grid_points: int

    def describe(self) -> str:
        return (
            "t={:.6f} atm={:.4%} calendar={:.2%} butterfly={:.2%} "
            "negative={:.2%} clipped={:.2%} nodes={}".format(
                self.vol_time,
                self.atm_vol,
                self.calendar_arb_ratio,
                self.butterfly_arb_ratio,
                self.negative_ratio,
                self.clipped_ratio,
                self.grid_points,
            )
        )


@dataclass(frozen=True)
class LocalVolSlice:
    """Local vol on one expiry, sampled on a log-moneyness grid."""

    expiry: datetime
    vol_time: float
    forward: float
    x_grid: np.ndarray  # ln(K / F)
    local_vols: np.ndarray
    diagnostics: LocalVolDiagnostics

    def at_moneyness(self, x: np.ndarray) -> np.ndarray:
        """Interpolate in ``x = ln(S / F)``; the grid edges hold outside."""
        values = np.atleast_1d(np.asarray(x, dtype=float))
        return np.interp(values, self.x_grid, self.local_vols)

    def at_spots(self, spots: np.ndarray) -> np.ndarray:
        """Interpolate at absolute spot levels (same convention as the engines)."""
        values = np.atleast_1d(np.asarray(spots, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(values / self.forward)
        return self.at_moneyness(x)


class DupireLocalVol:
    """``sigma_lv(T, S)`` from the fitted surface; one instance per market.

    Slices are cached per expiry, so the engines can call :meth:`local_vols`
    once per time step without redoing the IV surface sampling.  Rebuild the
    object after bumping the market (the cache is tied to the snapshot).
    """

    def __init__(
        self,
        market: MarketState,
        *,
        nodes: int = LOCAL_VOL_NODES,
        sigmas: float = 6.0,
        cap_multiple: float = 5.0,
        floor: float = 1e-4,
        dt_rel: float = 0.02,
        min_vol_time: float = 1e-8,
        slice_step: float = 0.01,
        spot_anchor: Optional[float] = None,
    ):
        if market.surface is None:
            raise ValueError(
                "local volatility requires a fitted surface "
                "(MarketState.surface is None)"
            )
        self.market = market
        self.surface = market.surface
        # The table is quoted in log-moneyness relative to this spot: the
        # coefficients are a property of the *model*, not of where the spot
        # happens to be, so the Greeks pin the base spot here while the spot
        # bump only moves the query point.  Rebuilding the table around each
        # bumped forward would drag the whole surface along and cancel most of
        # the sensitivity.
        self.spot_anchor = float(market.spot if spot_anchor is None else spot_anchor)
        self.nodes = max(int(nodes), 5)
        self.sigmas = float(sigmas)
        self.cap_multiple = float(cap_multiple)
        self.floor = float(floor)
        self.dt_rel = float(dt_rel)
        self.min_vol_time = float(min_vol_time)
        # Vol-time spacing of the prepared table: the coefficients are smooth in
        # ``t``, so slices are built every ``slice_step`` vol years and the
        # engines interpolate in between.  Building one slice is dominated by its
        # **grid construction** (``EDSSabrSlice`` rebuilds it on every calibration
        # iteration), not by the implied-surface sampling: a profile of a 2Y daily
        # grid (431 dates -> 145 slices) spent ~16s of 20s in
        # ``_generate_xy_grid`` and ~2s in the surface sampling, which is why the
        # quadrature there is batched (2026-10, 2.7x) rather than the sampler.
        self.slice_step = max(float(slice_step), 0.0)
        atm = np.asarray(self.surface.atm_vols, dtype=float)
        self._cap = self.cap_multiple * float(np.max(atm)) if atm.size else self.floor

        # one fixed log-moneyness grid for every slice: sampling the implied
        # surface is by far the dominant cost, so the total-variance samples
        # must be reusable across maturities
        atm_max = float(np.max(atm)) if atm.size else 0.2
        times = np.asarray(getattr(self.surface, "expiry_times", []), dtype=float)
        horizon = float(np.max(times)) if times.size else 1.0
        self._x_max = max(
            self.sigmas * max(atm_max, 1e-3) * math.sqrt(max(horizon, 1e-8)),
            self.floor,
        )
        self._x_grid = np.linspace(-self._x_max, self._x_max, self.nodes)
        self._cache: Dict[datetime, LocalVolSlice] = {}
        self._w_cache: Dict[datetime, Tuple[float, np.ndarray]] = {}
        self._prepared: List[datetime] = []

    # ------------------------------------------------------------- accessors
    @property
    def cap(self) -> float:
        return self._cap

    def slice_at(self, expiry: DateLike) -> LocalVolSlice:
        """Local-vol slice for ``expiry`` (cached).

        ``dw/dT`` uses a central difference at fixed log-moneyness here (three
        implied-surface samplings); the engines instead call :meth:`prepare`
        which walks the time grid once and reuses the previous total-variance
        sample - a single sampling per date.
        """
        key = to_datetime(expiry)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._build_slice(key)
        self._cache[key] = result
        return result

    def prepare(self, expiries: Sequence[DateLike]) -> None:
        """Pre-build slices in time order using backward differences.

        Every requested date then costs exactly one implied-surface sampling.
        The one-sided difference is fine: the local-vol grid is sampled at the
        engine's own time nodes, which are dense (weeks apart).

        Dates closer together than ``slice_step`` in vol time are *not* built -
        :meth:`local_vols` interpolates between the neighbours, which is exact
        to ``O(slice_step**2 d2 sigma/dt2)`` and keeps a daily monitoring grid
        (250 nodes a year) at a fifth of the sampling cost.  The first and last
        requested dates are always kept.
        """
        ordered = sorted({to_datetime(value) for value in expiries})
        keep: List[datetime] = []
        for index, expiry in enumerate(ordered):
            if index in (0, len(ordered) - 1):
                keep.append(expiry)
                continue
            if self._vol_time(expiry) - self._vol_time(keep[-1]) >= self.slice_step:
                keep.append(expiry)
        previous: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        for expiry in keep:
            if expiry not in self._cache:
                self._cache[expiry] = self._build_slice(expiry, previous=previous)
            previous = self._w_cache.get(expiry)
        self._prepared = keep

    # ------------------------------------------------------------ persistence
    def to_payload(self) -> Dict[str, Any]:
        """The prepared table as JSON-able data - what the on-disk cache stores.

        One entry per prepared slice: the coefficients plus the numbers they were
        evaluated at, so a reloaded table answers exactly like a rebuilt one.  The
        *inputs* (surface, curves, time grid) are the cache's business - they key
        the file and are stored next to it.
        """
        prepared = [to_datetime(value) for value in self._prepared]
        return {
            "nodes": int(self.nodes),
            "slice_step": float(self.slice_step),
            "spot_anchor": float(self.spot_anchor),
            "prepared": [_stamp(value) for value in prepared],
            "slices": [
                self._slice_payload(self._cache[value])
                for value in prepared
                if value in self._cache
            ],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], market: MarketState
    ) -> "DupireLocalVol":
        """Rebuild a table stored by :meth:`to_payload` on ``market``.

        The market supplies the surface and the vol-time scale, the payload the
        coefficients; the stored discretisation is restored too, so a cache hit and
        a rebuild are interchangeable (that is the whole point of the file).
        """
        table = cls(
            market,
            nodes=int(payload.get("nodes", LOCAL_VOL_NODES)),
            slice_step=float(payload.get("slice_step", LOCAL_VOL_SLICE_STEP)),
            spot_anchor=payload.get("spot_anchor"),
        )
        for item in payload.get("slices") or ():
            item = dict(item)
            diagnostics = dict(item.get("diagnostics") or {})
            vol_time = float(item["vol_time"])
            slice_ = LocalVolSlice(
                expiry=to_datetime(item["expiry"]),
                vol_time=vol_time,
                forward=float(item["forward"]),
                x_grid=np.asarray(item["x_grid"], dtype=float),
                local_vols=np.asarray(item["local_vols"], dtype=float),
                diagnostics=LocalVolDiagnostics(
                    vol_time=float(diagnostics.get("vol_time", vol_time)),
                    atm_vol=float(diagnostics.get("atm_vol", 0.0)),
                    calendar_arb_ratio=float(diagnostics.get("calendar_arb_ratio", 0.0)),
                    butterfly_arb_ratio=float(diagnostics.get("butterfly_arb_ratio", 0.0)),
                    negative_ratio=float(diagnostics.get("negative_ratio", 0.0)),
                    clipped_ratio=float(diagnostics.get("clipped_ratio", 0.0)),
                    grid_points=int(diagnostics.get("grid_points", table.nodes)),
                ),
            )
            table._cache[slice_.expiry] = slice_
        stored = payload.get("prepared")
        if stored:
            table._prepared = [to_datetime(value) for value in stored]
        else:  # pragma: no cover - a hand-written payload may skip the list
            table._prepared = list(table._cache)
        return table

    @staticmethod
    def _slice_payload(item: LocalVolSlice) -> Dict[str, Any]:
        stats = item.diagnostics
        return {
            "expiry": _stamp(item.expiry),
            "vol_time": float(item.vol_time),
            "forward": float(item.forward),
            "x_grid": [float(value) for value in np.asarray(item.x_grid)],
            "local_vols": [float(value) for value in np.asarray(item.local_vols)],
            "diagnostics": {
                "vol_time": float(stats.vol_time),
                "atm_vol": float(stats.atm_vol),
                "calendar_arb_ratio": float(stats.calendar_arb_ratio),
                "butterfly_arb_ratio": float(stats.butterfly_arb_ratio),
                "negative_ratio": float(stats.negative_ratio),
                "clipped_ratio": float(stats.clipped_ratio),
                "grid_points": int(stats.grid_points),
            },
        }

    def _bracket(self, expiry: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Prepared slices surrounding ``expiry`` (``(None, None)`` if unknown)."""
        if not self._prepared:
            return None, None
        index = bisect.bisect_left(self._prepared, expiry)
        if index < len(self._prepared) and self._prepared[index] == expiry:
            return expiry, expiry
        lower = self._prepared[index - 1] if index > 0 else None
        upper = self._prepared[index] if index < len(self._prepared) else None
        return lower, upper

    def local_vols(self, expiry: DateLike, spots: np.ndarray) -> np.ndarray:
        """Vectorised ``sigma_lv`` at absolute spot levels (the engines' entry).

        Exact when the date was prepared, otherwise linearly interpolated in vol
        time between the surrounding slices (evaluated at ``spots`` first, so a
        degenerate near-zero slice with its own grid is handled too).
        """
        key = to_datetime(expiry)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.at_spots(spots)
        lower, upper = self._bracket(key)
        if lower is None or upper is None or lower == upper:
            return self.slice_at(key).at_spots(spots)
        target = self._vol_time(key)
        t0 = self._vol_time(lower)
        t1 = self._vol_time(upper)
        weight = (target - t0) / (t1 - t0) if t1 > t0 else 0.0
        weight = min(max(weight, 0.0), 1.0)
        below = self._cache[lower].at_spots(spots)
        if weight <= 0.0:
            return below
        above = self._cache[upper].at_spots(spots)
        return (1.0 - weight) * below + weight * above

    def local_vol(self, expiry: DateLike, spot: float) -> float:
        """Scalar convenience wrapper."""
        return float(self.local_vols(expiry, np.asarray([spot], dtype=float))[0])

    def diagnostics(self) -> Dict[str, float]:
        """Worst-case diagnostics across every slice built so far."""
        if not self._cache:
            return {}
        slices: List[LocalVolSlice] = list(self._cache.values())
        return {
            "slices": float(len(slices)),
            "calendar_arb_ratio": max(item.diagnostics.calendar_arb_ratio for item in slices),
            "butterfly_arb_ratio": max(item.diagnostics.butterfly_arb_ratio for item in slices),
            "negative_ratio": max(item.diagnostics.negative_ratio for item in slices),
            "clipped_ratio": max(item.diagnostics.clipped_ratio for item in slices),
            "floor": self.floor,
            "cap": self._cap,
        }

    def describe(self) -> str:
        """One-line summary for reports (worst case over the built slices)."""
        stats = self.diagnostics()
        if not stats:
            return "local vol: no slices built"
        return (
            "local vol: {} slices | worst calendar={:.2%} butterfly={:.2%} "
            "clipped={:.2%} | floor={:.4f} cap={:.4f}".format(
                int(stats["slices"]),
                stats["calendar_arb_ratio"],
                stats["butterfly_arb_ratio"],
                stats["clipped_ratio"],
                stats["floor"],
                stats["cap"],
            )
        )

    # ------------------------------------------------------------- internals
    def _vol_time(self, expiry: DateLike) -> float:
        return float(
            self.surface.get_vol_time(expiry, valuation_date=self.market.valuation_date)
        )

    def _forward(self, expiry: DateLike) -> float:
        """Forward at the anchored spot (constant across the Greeks)."""
        return float(self.market.forward(expiry, spot=self.spot_anchor))

    def _implied_vol(
        self, expiry: DateLike, strikes: np.ndarray, forward: float
    ) -> np.ndarray:
        """Surface IV at absolute strikes, priced consistently with ``VanillaPricer``."""
        initial_forward = float(
            self.market.forward(expiry, spot=self.surface.init_spot)
        )
        values = self.surface.implied_vol(
            expiry,
            strikes,
            current_forward=float(forward),
            initial_forward=initial_forward,
            valuation_date=self.market.valuation_date,
        )
        return np.asarray(values, dtype=float)

    def _time_bracket(self, expiry: datetime) -> Tuple[datetime, datetime]:
        """Dates used for the ``dw/dT`` central difference at fixed moneyness."""
        days = max((to_date(expiry) - to_date(self.market.valuation_date)).days, 1)
        step = max(1, int(round(self.dt_rel * days)))
        up = expiry + timedelta(days=step)
        down = expiry - timedelta(days=step)
        earliest = to_datetime(self.market.valuation_date) + timedelta(days=1)
        if down < earliest:
            down = earliest
        return up, down

    @staticmethod
    def _strike_derivatives(
        ivs: np.ndarray, strikes: np.ndarray, x_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``d sigma/dK`` and ``d2 sigma/dK2`` from a uniform log-strike grid.

        Same stencils as the reference implementation: central inside, one-sided
        second-order at both edges, then converted from log-strike to strike
        derivatives (``d/dK = d/dlnK / K``).
        """
        step = float(x_grid[1] - x_grid[0])
        count = ivs.size
        first = np.empty(count, dtype=float)
        second = np.empty(count, dtype=float)

        first[1:-1] = (ivs[2:] - ivs[:-2]) / (2.0 * step)
        first[0] = (-3.0 * ivs[0] + 4.0 * ivs[1] - ivs[2]) / (2.0 * step)
        first[-1] = (3.0 * ivs[-1] - 4.0 * ivs[-2] + ivs[-3]) / (2.0 * step)

        second[1:-1] = (ivs[2:] - 2.0 * ivs[1:-1] + ivs[:-2]) / step ** 2
        second[0] = (2.0 * ivs[0] - 5.0 * ivs[1] + 4.0 * ivs[2] - ivs[3]) / step ** 2
        second[-1] = (
            2.0 * ivs[-1] - 5.0 * ivs[-2] + 4.0 * ivs[-3] - ivs[-4]
        ) / step ** 2

        sigma_k = first / strikes
        sigma_kk = (second - first) / strikes ** 2
        return sigma_k, sigma_kk

    def _w_at(self, expiry: datetime) -> Tuple[float, np.ndarray, np.ndarray]:
        """``(vol_time, total variance, implied vols)`` on the shared grid."""
        cached = self._w_cache.get(expiry)
        if cached is not None:
            return cached
        vol_time = self._vol_time(expiry)
        forward = self._forward(expiry)
        ivs = self._implied_vol(expiry, forward * np.exp(self._x_grid), forward)
        values = (float(vol_time), ivs ** 2 * vol_time, ivs)
        self._w_cache[expiry] = values
        return values

    def _flat_slice(self, expiry: datetime, vol_time: float) -> LocalVolSlice:
        """Degenerate slice at ``t -> 0``: flat at the ATM vol."""
        atm = float(self.surface.get_atm_vol(max(vol_time, 0.0)))
        x_grid = np.linspace(-self.floor, self.floor, 3)
        values = np.full_like(x_grid, atm)
        diagnostics = LocalVolDiagnostics(
            vol_time=float(vol_time),
            atm_vol=atm,
            calendar_arb_ratio=0.0,
            butterfly_arb_ratio=0.0,
            negative_ratio=0.0,
            clipped_ratio=0.0,
            grid_points=3,
        )
        return LocalVolSlice(
            expiry=expiry,
            vol_time=float(vol_time),
            forward=self._forward(expiry),
            x_grid=x_grid,
            local_vols=values,
            diagnostics=diagnostics,
        )

    def _build_slice(
        self,
        expiry: datetime,
        previous: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
    ) -> LocalVolSlice:
        vol_time, w, ivs = self._w_at(expiry)
        if vol_time <= self.min_vol_time:
            return self._flat_slice(expiry, vol_time)

        atm = float(self.surface.get_atm_vol(vol_time))
        if not (atm > 0.0):
            return self._flat_slice(expiry, vol_time)

        forward = self._forward(expiry)
        x_grid = self._x_grid
        strikes = forward * np.exp(x_grid)

        # dw/dT at fixed log-moneyness.  Walking the engine's time grid, the
        # previous total-variance sample is reused (one sampling per date);
        # standalone lookups fall back to a central difference.
        if previous is not None and vol_time > previous[0] + self.min_vol_time:
            dw_dt = (w - previous[1]) / (vol_time - previous[0])
        else:
            up, down = self._time_bracket(expiry)
            t_up, w_up, _ = self._w_at(up)
            t_down, w_down, _ = self._w_at(down)
            dw_dt = (w_up - w_down) / (t_up - t_down)

        sigma_k, sigma_kk = self._strike_derivatives(ivs, strikes, x_grid)
        sqrt_t = math.sqrt(vol_time)
        d1 = (-x_grid + 0.5 * ivs ** 2 * vol_time) / (ivs * sqrt_t)
        denominator = (1.0 + strikes * d1 * sqrt_t * sigma_k) ** 2 + ivs * vol_time * (
            strikes ** 2
        ) * (sigma_kk - d1 * sqrt_t * sigma_k ** 2)

        with np.errstate(divide="ignore", invalid="ignore"):
            variance = dw_dt / denominator
        negative = (~np.isfinite(variance)) | (variance < 0.0)
        local = np.sqrt(np.maximum(variance, 0.0))
        clipped = (local < self.floor) | (local > self._cap)
        local = np.clip(local, self.floor, self._cap)

        diagnostics = LocalVolDiagnostics(
            vol_time=float(vol_time),
            atm_vol=atm,
            calendar_arb_ratio=float(np.mean(dw_dt < 0.0)),
            butterfly_arb_ratio=float(np.mean(denominator < 0.0)),
            negative_ratio=float(np.mean(negative)),
            clipped_ratio=float(np.mean(clipped)),
            grid_points=int(self.nodes),
        )
        return LocalVolSlice(
            expiry=expiry,
            vol_time=float(vol_time),
            forward=forward,
            x_grid=x_grid,
            local_vols=np.asarray(local, dtype=float),
            diagnostics=diagnostics,
        )


class LocalVolCache:
    """Builds the Dupire table once per ``(surface, time grid)`` in a risk run.

    A table is a **model coefficient**: within one bump-and-revalue run the spot,
    rate and borrow bumps must not move it - they move the drifts and the query
    point while the coefficients stay put (sticky moneyness, see
    :class:`DupireLocalVol`).  A vol bump carries a new surface and the theta
    bump a new time grid, so only those two rebuild.

    This is also the dominant cost of a bumper run: a 1Y daily-monitored table
    takes ~5s, more than the path loop of a 16k-path Monte Carlo valuation, so a
    full risk run would otherwise pay for ten tables instead of four.
    """

    def __init__(
        self,
        *,
        nodes: int = LOCAL_VOL_NODES,
        slice_step: float = LOCAL_VOL_SLICE_STEP,
        spot_anchor: Optional[float] = None,
        store: Optional[Any] = None,
        fingerprint: Optional[Mapping[str, Any]] = None,
    ):
        self.nodes = max(int(nodes), 5)
        self.slice_step = max(float(slice_step), 0.0)
        self.spot_anchor = None if spot_anchor is None else float(spot_anchor)
        #: Optional store (``read(fingerprint) -> payload|None`` /
        #: ``write(fingerprint, payload)``, see
        #: :mod:`surface_pricer.io.local_vol_cache`) plus the JSON-able description
        #: of the **inputs** it keys on - the surface, the curves, the time grid and
        #: the discretisation.  Without both, everything stays in memory.
        self.store = store
        self.fingerprint = dict(fingerprint or {})
        self.builds = 0  #: tables actually built (cost diagnostics)
        self.loads = 0  #: tables served from the store
        self.stores = 0  #: tables written to the store
        self.stale = 0  #: stored tables refused because they were built at another spot
        self.source = ""  #: ``"built"`` or ``"cache"`` - what the last table cost
        self._base_surface_id: Optional[int] = None
        self._resolved_anchor: Optional[float] = None
        self._tables: Dict[Tuple[int, Tuple[datetime, ...]], Tuple[Any, "DupireLocalVol"]] = {}

    def table(self, market: MarketState, dates: Sequence[DateLike]) -> "DupireLocalVol":
        """The prepared local-vol table for this market's surface on ``dates``.

        The market that *first* asks for a given pair supplies the coefficients
        (the base market in a risk run), so the anchored forward does not follow
        a rate or borrow bump either.  With a store wired in, a miss *builds and
        stores* and a hit *loads* - the coefficients are identical either way.

        A stored table is only served when it was built around the **same anchor**:
        the coefficients are quoted in log-moneyness relative to it, so a table
        from another spot is another model, and it is rebuilt instead
        (``self.stale`` counts those).  The anchor is the one this cache resolved
        (pinned, or the first market's spot) - never the spot a *later* market
        happens to be at, which is what a bump or a ladder rung moves.
        """
        anchor = self._anchor_for(market)
        key = (id(market.surface), tuple(to_datetime(value) for value in dates))
        entry = self._tables.get(key)
        if entry is not None:
            return entry[1]
        fingerprint = self._fingerprint(market, dates)
        payload = self.store.read(fingerprint) if fingerprint is not None else None
        if payload is not None:
            table = DupireLocalVol.from_payload(payload, market)
            if not _same_spot(table.spot_anchor, anchor):
                # the file says a different spot than the one this run is anchored
                # to: refuse it (and rewrite it below) rather than price a model
                # whose coefficients belong to another level
                self.stale += 1
                payload = None
        if payload is not None:
            self.loads += 1
            self.source = "cache"
        else:
            table = DupireLocalVol(
                market,
                nodes=self.nodes,
                slice_step=self.slice_step,
                spot_anchor=anchor,
            )
            table.prepare(dates)
            self.builds += 1
            self.source = "built"
            if fingerprint is not None:
                self.store.write(fingerprint, table.to_payload())
                self.stores += 1
        # the surface is pinned on purpose: ``id`` is the cache key, so
        # holding the reference rules out an id being recycled by a later
        # bumped surface (every vol bump creates a new one)
        self._tables[key] = (market.surface, table)
        return table

    def _anchor_for(self, market: MarketState) -> float:
        """The spot the coefficients are anchored to: pinned, else resolved once.

        The apps pin the base spot; a caller that pins nothing gets the **first**
        market's spot (the same rule the base surface follows just above).  Once
        resolved it stays put: a spot bump - or a ladder rung - moves the query
        point, not the coefficients.
        """
        if self.spot_anchor is not None:
            return float(self.spot_anchor)
        if self._resolved_anchor is None:
            self._resolved_anchor = float(market.spot)
        return self._resolved_anchor

    def _fingerprint(
        self, market: MarketState, dates: Sequence[DateLike]
    ) -> Optional[Dict[str, Any]]:
        """The store key - the inputs, or ``None`` when there is nothing to key on.

        Only the **base** surface is served from disk (the market that first asks,
        exactly like the in-memory pin): a vol bump is a different model, and
        writing its table under the base key would poison the cache for every
        later run.  The first surface seen owns the file; the rest stay in memory.
        """
        if self.store is None or not self.fingerprint or market.surface is None:
            return None
        surface_id = id(market.surface)
        if self._base_surface_id is None:
            self._base_surface_id = surface_id
        elif surface_id != self._base_surface_id:
            return None
        return {
            **self.fingerprint,
            "spot_anchor": self._anchor_for(market),
            "nodes": int(self.nodes),
            "slice_step": float(self.slice_step),
            "dates": [to_datetime(value).isoformat(sep=" ") for value in dates],
        }


def make_table_cache(
    *,
    spot_anchor: Optional[float] = None,
    store: Optional[Any] = None,
    fingerprint: Optional[Mapping[str, Any]] = None,
) -> LocalVolCache:
    """A table provider with the module's default discretisation.

    The apps build the cache (they know the run, the curves and where the output
    root is) and hand it to the engine, so the *cache* owns ``nodes`` /
    ``slice_step``: passing a cache means accepting its table, defaults included.
    """
    return LocalVolCache(
        nodes=LOCAL_VOL_NODES,
        slice_step=LOCAL_VOL_SLICE_STEP,
        spot_anchor=spot_anchor,
        store=store,
        fingerprint=fingerprint,
    )


def _stamp(value: datetime) -> str:
    """The payload's date spelling (``to_datetime`` reads it straight back)."""
    return to_datetime(value).isoformat(sep=" ")


def _same_spot(left: Any, right: Any) -> bool:
    """Two spots are the same spot, up to the rounding of a JSON round trip."""
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return abs(left_value - right_value) <= 1e-9 * max(1.0, abs(right_value))


__all__ = [
    "LOCAL_VOL_NODES",
    "LOCAL_VOL_SLICE_STEP",
    "DupireLocalVol",
    "LocalVolCache",
    "LocalVolDiagnostics",
    "LocalVolSlice",
    "make_table_cache",
]
