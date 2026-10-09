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
gamma_cash      spot (relative)               second central     d2NPV/dSpot2 * spot^2 / 100
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

Every stencil re-prices the **resolved** option (``VanillaSpec``), so a bump moves
the market and never the trade: a ``percentage`` strike is resolved once, and the
bump then measures a fixed strike.  ``calculate_greeks`` resolves the contract
first and hands over to ``calculate_greeks_spec``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Optional

from ...core.market import MarketState
from ..results import PricingResult, RiskSettings
from ..risk.bumps import parallel_bump, spot_bump, vol_bump
from ..risk.buckets import (
    BUCKET_NAMES,
    bucket_greeks,
    bucket_grid_info,
    convert_bucketed_delta,  # noqa: F401  (kept importable from this module)
)
from ..risk.diff import SPOT_GREEKS, selected_greeks
from .contract import VanillaContract
from .pricer import VanillaPricer
from .spec import VanillaSpec, resolve_spec


# --------------------------------------------------------------- value helper
def _value(spec: VanillaSpec, market: MarketState) -> float:
    """One vanilla repricing on ``market``; the bumps come from ``risk.bumps``.

    The **resolved** terms are carried over (``spec.rebased``), so a bump moves the
    market only: a percentage strike stays the absolute strike it was resolved at.
    """
    return VanillaPricer(market).npv_spec(spec.rebased(market))


# ----------------------------------------------------------------- public API
def calculate_greeks(
    contract: VanillaContract,
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    bucketed_vega_pillars: Optional[Iterable] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
) -> PricingResult:
    """Price ``contract`` and fill the full bump-and-revalue Greek set.

    The raw terms are resolved **once** and every bump re-prices that same
    resolved option on the bumped market - see :func:`calculate_greeks_spec`.
    """
    return calculate_greeks_spec(
        resolve_spec(contract, market),
        market,
        settings=settings,
        bucketed_vega_pillars=bucketed_vega_pillars,
        bucketed_delta_pillars=bucketed_delta_pillars,
    )


def calculate_greeks_spec(
    spec: VanillaSpec,
    market: MarketState,
    settings: Optional[RiskSettings] = None,
    bucketed_vega_pillars: Optional[Iterable] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
) -> PricingResult:
    """Price a resolved spec and fill the full bump-and-revalue Greek set.

    This is the JSON layer's risk entry point (``VanillaSpec.from_dict`` builds the
    spec).  ``bucketed_vega_pillars`` default to the surface expiries;
    ``bucketed_delta_pillars`` are borrow-curve bucket dates (defaults to the
    curve pillars, or edslib's tenor grid for a constant curve) and drive both
    ``bucketed_rhoq`` and the derived ``bucketed_delta``.
    """
    settings = settings or RiskSettings()
    base = VanillaPricer(market).price_spec(spec)
    base_value = base.npv

    # ``settings.greeks`` selects which bumps run at all; ``None`` means ``all`` -
    # the parallel Greeks including volga / vanna - on **both** pricers (2026-10).
    # The buckets are opt-in everywhere now (each is a bump pair of its own), so a
    # caller that wants them spells them out: ``greeks=("all", "buckets")``.
    wanted = selected_greeks(settings)
    buckets_wanted = bool(wanted.intersection(BUCKET_NAMES))

    # ---- the spot pair: delta family, gamma family, and vanna's spot leg ------
    # One pair carries delta, its two rescalings, gamma and cash gamma.  It also
    # runs for the buckets, because bucketed delta converts ``delta_cash`` into the
    # borrow buckets (the converted value is local then: ``base.delta_cash`` stays
    # unset unless the delta family was asked for).
    spot_step = abs(market.spot) * settings.delta_bump_pct
    if spot_step <= 0.0:
        raise ValueError("spot must be positive")
    spot_wanted = wanted.intersection(SPOT_GREEKS)
    delta_cash: Optional[float] = None
    if spot_wanted or buckets_wanted or "vanna" in wanted:
        up_spot = _value(spec, spot_bump(market, spot_step))
        down_spot = _value(spec, spot_bump(market, -spot_step))
        raw_delta = (up_spot - down_spot) / (2.0 * spot_step)
        if spot_wanted.intersection(("delta", "delta_cash", "delta_n")):
            base.delta = float(raw_delta)
            base.delta_cash = float(raw_delta * market.spot)
            base.delta_n = float(raw_delta / market.spot)
        if buckets_wanted:
            delta_cash = float(raw_delta * market.spot)
        if spot_wanted.intersection(("gamma", "gamma_cash")):
            gamma_step = abs(market.spot) * settings.gamma_bump_pct
            if gamma_step <= 0.0:
                raise ValueError("spot must be positive")
            up_gamma = _value(spec, spot_bump(market, gamma_step))
            down_gamma = _value(spec, spot_bump(market, -gamma_step))
            gamma = (up_gamma - 2.0 * base_value + down_gamma) / (gamma_step ** 2)
            # the family rides together, exactly like delta / delta_cash / delta_n:
            # asking for gamma hands back its cash form too, and vice versa
            base.gamma = float(gamma)
            base.gamma_cash = float(gamma * market.spot ** 2 / 100.0)

    # ---- vega / volga ---------------------------------------------------------
    vol_step = settings.vega_bump
    if wanted.intersection(("vega", "volga")):
        up_vol = _value(spec, vol_bump(market, vol_step))
        down_vol = _value(spec, vol_bump(market, -vol_step))
        if "vega" in wanted:
            base.vega = float((up_vol - down_vol) / (2.0 * vol_step))
        if "volga" in wanted:
            volga_step = settings.volga_bump
            if abs(volga_step - vol_step) < 1.0e-15:
                volga_up, volga_down = up_vol, down_vol
            else:
                volga_up = _value(spec, vol_bump(market, volga_step))
                volga_down = _value(spec, vol_bump(market, -volga_step))
            base.volga = float(
                (volga_up - 2.0 * base_value + volga_down) / (volga_step ** 2)
            )

    # ---- theta ----------------------------------------------------------------
    if "theta" in wanted:
        theta_date = market.valuation_date + timedelta(days=settings.theta_days)
        if theta_date < spec.expiry_date:
            theta_market = market.clone(valuation_date=theta_date)
            base.theta = float(
                (_value(spec, theta_market) - base_value) / max(settings.theta_days, 1)
            )
        else:
            base.theta = float(-base_value)

    # ---- vanna ----------------------------------------------------------------
    if "vanna" in wanted:
        vanna_vol_step = settings.vanna_vol_bump
        mixed_up_up = _value(spec, vol_bump(spot_bump(market, spot_step), vanna_vol_step))
        mixed_up_down = _value(spec, vol_bump(spot_bump(market, spot_step), -vanna_vol_step))
        mixed_down_up = _value(spec, vol_bump(spot_bump(market, -spot_step), vanna_vol_step))
        mixed_down_down = _value(
            spec, vol_bump(spot_bump(market, -spot_step), -vanna_vol_step)
        )
        base.vanna = float(
            (mixed_up_up - mixed_up_down - mixed_down_up + mixed_down_down)
            / (4.0 * spot_step * vanna_vol_step)
        )

    # ---- rho (rate) / rhoQ (borrow) -------------------------------------------
    if "rho" in wanted:
        rate_step = settings.rate_bump
        rate_up = _value(spec, parallel_bump(market, "rate", rate_step))
        rate_down = _value(spec, parallel_bump(market, "rate", -rate_step))
        base.rho = float((rate_up - rate_down) / (2.0 * rate_step))

    if "rhoq" in wanted:
        borrow_step = settings.borrow_bump
        borrow_up = _value(spec, parallel_bump(market, "borrow", borrow_step))
        borrow_down = _value(spec, parallel_bump(market, "borrow", -borrow_step))
        base.rhoq = float((borrow_up - borrow_down) / (2.0 * borrow_step))

    # ---- reporting scalings ---------------------------------------------------
    # Applied once at the end so a Greek is reported in its unit whether or not its
    # neighbours were asked for.
    if settings.report_vega_per_vol_point:
        if base.vega is not None:
            base.vega /= 100.0
        if base.volga is not None:
            base.volga /= 10000.0
        if base.vanna is not None:
            base.vanna /= 100.0
    if settings.report_rho_per_pct:
        if base.rho is not None:
            base.rho /= 100.0
        if base.rhoq is not None:
            base.rhoq /= 100.0

    # ---- bucketed vega / rhoQ / delta / rho -----------------------------------
    # Shared with the exotic engines (``pricing.risk.buckets``): the vega buckets
    # are the surface expiries, the rho / rhoQ buckets the curve pillars (a flat
    # curve is rebuilt on edslib's tenor grid first so every bucket stays a
    # single-pillar bump), and bucketed delta distributes the spot cash delta
    # over the borrow buckets.
    def value(bumped_market: MarketState) -> float:
        return _value(spec, bumped_market)

    bucket_which = tuple(name for name in BUCKET_NAMES if name in wanted)
    if "bucketed_delta" in bucket_which and "bucketed_rhoq" not in bucket_which:
        # bucketed delta is derived from the borrow buckets: compute its input,
        # but only report what was asked for
        bucket_which += ("bucketed_rhoq",)
    buckets = bucket_greeks(
        value,
        market,
        settings,
        delta_cash=delta_cash,
        which=bucket_which,
        bucketed_vega_pillars=bucketed_vega_pillars,
        bucketed_delta_pillars=bucketed_delta_pillars,
        horizon=spec.expiry_date,
    )
    for name in BUCKET_NAMES:
        if name in wanted:
            setattr(base, name, buckets[name])
    if bucket_which:
        base.metadata["bucket_grid"] = bucket_grid_info(
            market,
            settings,
            horizon=spec.expiry_date,
            bucketed_delta_pillars=bucketed_delta_pillars,
        )

    base.metadata["greek_convention"] = {
        "delta": "dNPV/dSpot",
        "delta_cash": "dNPV/dSpot * spot",
        "delta_n": "dNPV/dSpot / spot",
        "gamma": "d2NPV/dSpot2",
        "gamma_cash": "d2NPV/dSpot2 * spot^2 / 100 (NPV change per (1% spot move)^2)",
        "vega": "dNPV/dVol, reported per 1 vol point"
        if settings.report_vega_per_vol_point
        else "dNPV/dVol",
        "volga": "d2NPV/dVol2, reported per (1 vol point)^2"
        if settings.report_vega_per_vol_point
        else "d2NPV/dVol2",
        "theta": "valuation date + theta_days, same trade re-priced, per day",
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


__all__ = ["calculate_greeks", "calculate_greeks_spec", "convert_bucketed_delta"]
