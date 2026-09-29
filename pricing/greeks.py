"""Bump-based vanilla Greeks, aligned with the edslib risk convention.

Reference: edslib ``greeks/greeks.py`` (the ``Valuation`` op plus the ``Greek``
classes).  The standalone port keeps the same bump objects, difference
stencils and reporting scalings:

==============  ============================  =================  ==============================
Greek           bump                          difference         reported unit
==============  ============================  =================  ==============================
delta           spot (relative)               central            dNPV/dSpot
delta_cash      spot (relative)               central            dNPV/dSpot * spot  (edslib Delta($))
delta_n         spot (relative)               central            dNPV/dSpot / spot  (edslib Delta Shares)
gamma           spot (relative)               second central     d2NPV/dSpot2
vega            vol surface parallel          central            per 1 vol point
volga           vol surface parallel          second central     per (1 vol point)^2
vanna           spot x vol                    cross (4 states)   per 1 vol point
theta           valuation date +1D            forward            NPV change per calendar day
rho             rate curve parallel           central            per 1% rate
rhoq            borrow curve parallel         central            per 1% borrow
bucketed_vega   one vol pillar at a time      central/backward   per 1 vol point
bucketed_rhoq   one borrow pillar at a time   central            per 1% borrow
bucketed_rho    one rate pillar at a time     central            per 1% rate
bucketed_delta  ``bucketed_rhoq`` distributed over the spot delta_cash
==============  ============================  =================  ==============================

``bucketed_delta`` splits the spot ``delta_cash`` over the borrow buckets by
their sensitivity share - ``delta_bucket = rhoQ * 100 / -tau`` with one shared
effective time ``tau = -100 * sum(rhoQ) / delta_cash`` - so the buckets always
add up to ``delta_cash``.  (edslib's
``RhoQBucketed._transfer_to_bucketed_delta`` divides by each bucket's own dcf
instead; the conventions coincide for ATM listed options with the expiry on a
bucket.)

Flat curves have no pillar of their own, so the bucketed rate/borrow rows are
computed after rebuilding the curve on the bucket grid (edslib's
``rebuild_ql_curve_by_tenors``); a curve that already carries pillars - the one
from ``build-borrow-curve`` - is bumped as it stands.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..core.daycount import shift_tenor, to_datetime
from ..core.market import MarketState
from .contracts import VanillaContract
from .results import PricingResult, RiskSettings
from .vanilla import VanillaPricer

# edslib's ``TimeBucketedGreek._DEFAULT_TIME_BUCKETED_TENORS``: the grid a flat
# curve is rebuilt on (and bucketed over) when it carries no pillars itself.
_DEFAULT_TIME_BUCKETED_TENORS = ("1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")

_CUMULATIVE_METHODS = {"cumulative", "cumulative_backward"}


# ---------------------------------------------------------------- bump helpers
def _value(contract: VanillaContract, market: MarketState) -> float:
    return VanillaPricer(market).npv(contract)


def _require_surface(market: MarketState):
    if market.surface is None:
        raise ValueError("MarketState.surface is required for vol-related Greeks")
    return market.surface


def _spot_bump(market: MarketState, amount: float) -> MarketState:
    return market.clone(spot=market.spot + amount)


def _vol_bump(market: MarketState, amount: float) -> MarketState:
    return market.clone(surface=_require_surface(market).bump_parallel(amount))


def _curve_bump(market: MarketState, curve_name: str, pillar, amount: float) -> MarketState:
    curve = market.rate_curve if curve_name == "rate" else market.borrow_curve
    if curve is None:
        return market
    bumped = curve.bump_pillar(pillar, amount)
    return market.clone(**{curve_name + "_curve": bumped})


def _parallel_bump(market: MarketState, curve_name: str, amount: float) -> MarketState:
    """Shift a whole curve in parallel (falling back to its single pillar)."""
    curve = market.rate_curve if curve_name == "rate" else market.borrow_curve
    if curve is None:
        return market
    rates = getattr(curve, "rates", None)
    if rates is not None and hasattr(curve, "with_rates"):
        bumped = curve.with_rates(np.asarray(rates, dtype=float) + amount)
        return market.clone(**{curve_name + "_curve": bumped})
    return market.clone(**{curve_name + "_curve": curve.bump_pillar(0, amount)})


# ------------------------------------------------------------- bucket plumbing
def _vol_pillars(market: MarketState, pillars: Optional[Iterable]) -> List:
    if pillars is not None:
        return list(pillars)
    return list(_require_surface(market).expiry_dates)


def _default_bucket_grid(market: MarketState) -> List:
    """edslib's tenor grid, shifted to the valuation date (flat curves only)."""
    return [
        shift_tenor(market.valuation_date, tenor, market.calendar)
        for tenor in _DEFAULT_TIME_BUCKETED_TENORS
    ]


def _borrow_buckets(market: MarketState, pillars: Optional[Iterable]) -> List:
    if pillars is not None:
        return list(pillars)
    curve = market.borrow_curve
    if curve is None:
        return []
    pillar_dates = list(getattr(curve, "pillar_dates", None) or [])
    return pillar_dates if pillar_dates else _default_bucket_grid(market)


def _rate_pillars(market: MarketState) -> List:
    curve = market.rate_curve
    if curve is None:
        return []
    pillar_dates = list(getattr(curve, "pillar_dates", None) or [])
    return pillar_dates if pillar_dates else _default_bucket_grid(market)


def _with_bucketed_curves(
    market: MarketState,
    borrow_pillars: List,
    rate_pillars: List,
) -> MarketState:
    """Expand flat curves onto the bucket grid before bumping them.

    edslib's ``TimeBucketedGreek._build_time_bucketed_curve`` rebuilds the curve
    on the bucket tenors first (``rebuild_ql_curve_by_tenors``) and bumps a
    single pillar afterwards.  Curves that already carry pillars (the borrow
    curve built by ``build-borrow-curve``, a piecewise rate curve) are used as
    they are.
    """
    updates = {}
    for name, pillars in (("borrow_curve", borrow_pillars), ("rate_curve", rate_pillars)):
        curve = getattr(market, name, None)
        if curve is None or not pillars:
            continue
        if list(getattr(curve, "pillar_dates", None) or []):
            continue
        rebuild = getattr(curve, "rebuild", None)
        if rebuild is None:
            continue
        updates[name] = rebuild(pillars, anchor=market.valuation_date)
    return market.clone(**updates) if updates else market


def _vega_bucket(

    contract: VanillaContract,
    market: MarketState,
    pillar,
    settings: RiskSettings,
) -> float:
    step = settings.vega_bump
    surface = _require_surface(market)
    if settings.bucketed_vega_method.strip().lower() in _CUMULATIVE_METHODS:
        up = _value(contract, market.clone(surface=surface.bump_cumulative_backward(pillar, step)))
        down = _value(contract, market.clone(surface=surface.bump_cumulative_backward(pillar, -step)))
    else:
        up = _value(contract, market.clone(surface=surface.bump_pillar(pillar, step)))
        down = _value(contract, market.clone(surface=surface.bump_pillar(pillar, -step)))
    bucket = (up - down) / (2.0 * step)
    if settings.report_vega_per_vol_point:
        bucket /= 100.0
    return float(bucket)


def _curve_bucket(
    contract: VanillaContract,
    market: MarketState,
    pillar,
    step: float,
    curve_name: str,
) -> float:
    up = _value(contract, _curve_bump(market, curve_name, pillar, step))
    down = _value(contract, _curve_bump(market, curve_name, pillar, -step))
    return float((up - down) / (2.0 * step))


def convert_bucketed_delta(
    bucketed_rhoq: Dict[str, float],
    delta_cash: float,
) -> Dict[str, float]:
    """Split the spot cash delta over the borrow buckets by sensitivity share.

    ``delta_bucket = delta_cash * rhoQ_bucket / sum(rhoQ_buckets)``, written
    here as ``rhoQ_per_1pct * 100 / -tau`` with one shared *effective*
    conversion time ``tau = -100 * sum(rhoQ) / delta_cash`` - so the buckets
    add up to ``delta_cash`` exactly, by construction.

    Anchoring ``tau`` on the priced ``delta_cash`` is deliberate: next to the
    pure forward channel (``-d lnF / dq``, the curve's own time) the spot and
    borrow bumps move moneyness - and hence the smile vol - slightly
    differently, and the spot delta itself carries its finite-difference
    stencil error (~2% for an off-the-money strike at the 1% default bump).
    Folding both into the shared time keeps the buckets consistent with the
    reported ``delta_cash``.  edslib's
    ``RhoQBucketed._transfer_to_bucketed_delta`` divides by each bucket's own
    dcf instead; the conventions coincide for ATM listed options with the
    expiry on a bucket.
    """
    total = float(sum(bucketed_rhoq.values()))
    if total == 0.0 or not delta_cash:
        return {}
    share = float(delta_cash) / total
    return {label: float(bucket) * share for label, bucket in bucketed_rhoq.items()}


# ----------------------------------------------------------------- public API
def calculate_greeks(
    contract: VanillaContract,
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    bucketed_vega_pillars: Optional[Iterable] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
) -> PricingResult:
    """Price ``contract`` and fill the full bump-and-revalue Greek set.

    ``bucketed_vega_pillars`` default to the surface expiries;
    ``bucketed_delta_pillars`` are borrow-curve bucket dates (defaults to the
    curve pillars, or edslib's tenor grid for a constant curve) and drive both
    ``bucketed_rhoq`` and the derived ``bucketed_delta``.
    """
    settings = settings or RiskSettings()
    base = VanillaPricer(market).price(contract)
    base_value = base.npv

    # ---- delta / gamma (spot) -------------------------------------------------
    spot_step = abs(market.spot) * settings.delta_bump_pct
    if spot_step <= 0.0:
        raise ValueError("spot must be positive")
    up_spot = _value(contract, _spot_bump(market, spot_step))
    down_spot = _value(contract, _spot_bump(market, -spot_step))
    raw_delta = (up_spot - down_spot) / (2.0 * spot_step)
    base.delta = float(raw_delta)
    base.delta_cash = float(raw_delta * market.spot)
    base.delta_n = float(raw_delta / market.spot)

    gamma_step = abs(market.spot) * settings.gamma_bump_pct
    if gamma_step <= 0.0:
        raise ValueError("spot must be positive")
    up_gamma = _value(contract, _spot_bump(market, gamma_step))
    down_gamma = _value(contract, _spot_bump(market, -gamma_step))
    base.gamma = float((up_gamma - 2.0 * base_value + down_gamma) / (gamma_step ** 2))

    # ---- vega / volga ---------------------------------------------------------
    vol_step = settings.vega_bump
    up_vol = _value(contract, _vol_bump(market, vol_step))
    down_vol = _value(contract, _vol_bump(market, -vol_step))
    base.vega = float((up_vol - down_vol) / (2.0 * vol_step))

    volga_step = settings.volga_bump
    if abs(volga_step - vol_step) < 1.0e-15:
        volga_up, volga_down = up_vol, down_vol
    else:
        volga_up = _value(contract, _vol_bump(market, volga_step))
        volga_down = _value(contract, _vol_bump(market, -volga_step))
    base.volga = float((volga_up - 2.0 * base_value + volga_down) / (volga_step ** 2))

    if settings.report_vega_per_vol_point:
        base.vega /= 100.0
        base.volga /= 10000.0

    # ---- theta ----------------------------------------------------------------
    theta_date = market.valuation_date + timedelta(days=settings.theta_days)
    if theta_date < contract.expiry:
        theta_market = market.clone(valuation_date=theta_date)
        base.theta = float(
            (_value(contract, theta_market) - base_value) / max(settings.theta_days, 1)
        )
    else:
        base.theta = float(-base_value)

    # ---- vanna ----------------------------------------------------------------
    vanna_vol_step = settings.vanna_vol_bump
    mixed_up_up = _value(contract, _vol_bump(_spot_bump(market, spot_step), vanna_vol_step))
    mixed_up_down = _value(contract, _vol_bump(_spot_bump(market, spot_step), -vanna_vol_step))
    mixed_down_up = _value(contract, _vol_bump(_spot_bump(market, -spot_step), vanna_vol_step))
    mixed_down_down = _value(contract, _vol_bump(_spot_bump(market, -spot_step), -vanna_vol_step))
    base.vanna = float(
        (mixed_up_up - mixed_up_down - mixed_down_up + mixed_down_down)
        / (4.0 * spot_step * vanna_vol_step)
    )
    if settings.report_vega_per_vol_point:
        base.vanna /= 100.0

    # ---- rho (rate) / rhoQ (borrow) -------------------------------------------
    rate_step = settings.rate_bump
    rate_up = _value(contract, _parallel_bump(market, "rate", rate_step))
    rate_down = _value(contract, _parallel_bump(market, "rate", -rate_step))
    base.rho = float((rate_up - rate_down) / (2.0 * rate_step))

    borrow_step = settings.borrow_bump
    borrow_up = _value(contract, _parallel_bump(market, "borrow", borrow_step))
    borrow_down = _value(contract, _parallel_bump(market, "borrow", -borrow_step))
    base.rhoq = float((borrow_up - borrow_down) / (2.0 * borrow_step))

    if settings.report_rho_per_pct:
        base.rho /= 100.0
        base.rhoq /= 100.0

    # ---- bucketed vega --------------------------------------------------------
    for pillar in _vol_pillars(market, bucketed_vega_pillars):
        label = to_datetime(pillar).date().isoformat()
        base.bucketed_vega[label] = _vega_bucket(contract, market, pillar, settings)

    # ---- bucketed rhoQ / delta ------------------------------------------------
    # Flat curves are first expanded onto the bucket grid so every bucket is a
    # single-pillar bump instead of a parallel shift (edslib's
    # ``_build_time_bucketed_curve`` + ``bump_given_tenor`` sequence).
    borrow_pillars = _borrow_buckets(market, bucketed_delta_pillars)
    rate_pillars = _rate_pillars(market)
    bucketed_market = _with_bucketed_curves(market, borrow_pillars, rate_pillars)

    for pillar in borrow_pillars:
        label = to_datetime(pillar).date().isoformat()
        bucket = _curve_bucket(contract, bucketed_market, pillar, borrow_step, "borrow")
        if settings.report_rho_per_pct:
            bucket /= 100.0
        base.bucketed_rhoq[label] = bucket
    base.bucketed_delta = convert_bucketed_delta(base.bucketed_rhoq, base.delta_cash)

    # ---- bucketed rho ---------------------------------------------------------
    for pillar in rate_pillars:
        label = to_datetime(pillar).date().isoformat()
        bucket = _curve_bucket(contract, bucketed_market, pillar, rate_step, "rate")
        if settings.report_rho_per_pct:
            bucket /= 100.0
        base.bucketed_rho[label] = bucket

    base.metadata["greek_convention"] = {
        "delta": "dNPV/dSpot",
        "delta_cash": "dNPV/dSpot * spot",
        "delta_n": "dNPV/dSpot / spot",
        "gamma": "d2NPV/dSpot2",
        "vega": "dNPV/dVol, reported per 1 vol point"
        if settings.report_vega_per_vol_point
        else "dNPV/dVol",
        "volga": "d2NPV/dVol2, reported per (1 vol point)^2"
        if settings.report_vega_per_vol_point
        else "d2NPV/dVol2",
        "theta": "NPV(t + theta_days) - NPV(t), per day",
        "vanna": "d2NPV/(dSpot dVol), reported per 1 vol point"
        if settings.report_vega_per_vol_point
        else "d2NPV/(dSpot dVol)",
        "rho": "dNPV/dRate, reported per 1%"
        if settings.report_rho_per_pct
        else "dNPV/dRate",
        "rhoq": "dNPV/dBorrow, reported per 1%"
        if settings.report_rho_per_pct
        else "dNPV/dBorrow",
        "bucketed_vega": "dNPV/dVol[pillar], method={}".format(settings.bucketed_vega_method),
        "bucketed_rhoq": (
            "dNPV/dBorrow[pillar] (single-pillar bump; flat curves are rebuilt "
            "on the bucket grid first), reported per 1%"
        ),
        "bucketed_rho": (
            "dNPV/dRate[pillar] (single-pillar bump on the rebuilt grid), reported per 1%"
        ),
        "bucketed_delta": (
            "delta_cash split over the borrow buckets by sensitivity share "
            "(sum(buckets) == delta_cash)"
        ),
    }
    return base


__all__ = ["calculate_greeks", "convert_bucketed_delta"]
