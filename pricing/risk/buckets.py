"""Bucketed (per-pillar) Greeks, shared by the vanilla and the exotic risk runs.

The buckets follow edslib's ``TimeBucketedGreek`` convention, which is also what
the vanilla pricer reports:

================  ============================  =================  ========================
Greek             bumps                         stencil            reporting unit
================  ============================  =================  ========================
``bucketed_vega`` one vol pillar at a time      central/backward   per 1 vol point
``bucketed_rhoq`` one borrow pillar at a time   central            per 1% borrow
``bucketed_rho``  one rate pillar at a time     central            per 1% rate
``bucketed_delta````bucketed_rhoq`` distributed over the spot ``delta_cash``
================  ============================  =================  ========================

The vega buckets are the surface expiries, the rho / rhoQ buckets the curve
pillars.  A flat curve has no pillar of its own, so it is first rebuilt on
edslib's default tenor grid and bumped one tenor at a time afterwards.

Everything works off a ``value(market) -> float`` callable, so the same code
serves the analytic vanilla pricer and the exotic engines (which pass their own
anchored, common-random-number valuation).

Cost: every bucket is a *pair* of valuations, and on the exotic side a per-pillar
vol bump invalidates the cached Dupire table (a 6-pillar surface costs 12 table
rebuilds).  Buckets are therefore opt-in - ``--greeks bucketed_vega`` - and stay
out of ``all``.

The **auto** curve grid is trade-aware (``bucket_grid``): a pillar past the trade's
last payment is dropped (keeping the one bracketing it), and beyond
``RiskSettings.bucket_group_after`` the pillars merge into one bucket per year of
tenor, bumped together.  A 15-pillar borrow curve financing a 2Y snowball costs a
handful of bump pairs instead of thirty valuations, and because these curves add a
per-pillar bump additively the coarse buckets still sum to the parallel Greek.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...core.daycount import DateLike, shift_tenor, to_date, to_datetime
from ...core.market import MarketState
from ..results import RiskSettings
from .bumps import curve_bump, require_surface, spot_bump
from .diff import BUCKET_NAMES, ValueFn, parse_greeks

#: edslib's ``TimeBucketedGreek._DEFAULT_TIME_BUCKETED_TENORS``: the grid a flat
#: curve is rebuilt on (and bucketed over) when it carries no pillars itself.
DEFAULT_TIME_BUCKETED_TENORS = ("1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")

#: Tenor beyond which the auto grid coarsens to one bucket per year of tenor
#: (``RiskSettings.bucket_group_after``; ``None`` keeps a bucket per pillar).
DEFAULT_BUCKET_GROUP_AFTER = "1Y"

_CUMULATIVE_METHODS = {"cumulative", "cumulative_backward"}


# ------------------------------------------------------------------- bucket grids
def vol_pillars(market: MarketState, pillars: Optional[Iterable] = None) -> List:
    """Vega buckets: the surface expiries unless the caller pins its own."""
    if pillars is not None:
        return list(pillars)
    return list(require_surface(market).expiry_dates)


def default_bucket_grid(market: MarketState) -> List:
    """edslib's tenor grid, shifted to the valuation date (flat curves only)."""
    return [
        shift_tenor(market.valuation_date, tenor, market.calendar)
        for tenor in DEFAULT_TIME_BUCKETED_TENORS
    ]


def borrow_buckets(market: MarketState, pillars: Optional[Iterable] = None) -> List:
    """rhoQ / bucketed-delta buckets: the borrow pillars, else the tenor grid."""
    if pillars is not None:
        return list(pillars)
    curve = market.borrow_curve
    if curve is None:
        return []
    pillar_dates = list(getattr(curve, "pillar_dates", None) or [])
    return pillar_dates if pillar_dates else default_bucket_grid(market)


def rate_pillars(market: MarketState) -> List:
    """rho buckets: the rate pillars, else the tenor grid."""
    curve = market.rate_curve
    if curve is None:
        return []
    pillar_dates = list(getattr(curve, "pillar_dates", None) or [])
    return pillar_dates if pillar_dates else default_bucket_grid(market)


# ------------------------------------------------------------------- bucket grids
def bucket_grid(
    pillars: Sequence[Any],
    *,
    valuation: Optional[DateLike] = None,
    horizon: Optional[DateLike] = None,
    group_after: Optional[str] = DEFAULT_BUCKET_GROUP_AFTER,
    calendar=None,
) -> List[Tuple[str, List[Any]]]:
    """Assign curve pillars to buckets - the trade-aware coarse grid.

    Three rules, each about a bump pair that is *not* worth paying for:

    1. a pillar **past** ``horizon`` (the trade's last payment) is dropped - a
       locally interpolated curve says nothing about dates before its own pillar,
       so the trade cannot see it.  The pillar **bracketing** the horizon is kept
       (the segment after it is the first that carries no weight), which is what
       makes the drop exact rather than approximate for ``linear_zero`` and
       flat-forward curves; a non-local cubic curve wants ``group_after=None``;
    2. up to ``group_after`` (default ``1Y``) every pillar keeps a bucket of its
       own - the near end is where the curve has its details and where the
       sensitivity lives;
    3. beyond it, one bucket per **year of tenor**, every pillar of the band bumped
       together, so a 15-pillar curve costs a handful of pairs.

    Without ``valuation`` the year bands have no origin and every pillar keeps its
    own bucket - which is also how an explicitly pinned grid is treated.  A group's
    label is its last pillar (for a band, the band's end).
    """
    ordered = sorted({to_datetime(pillar) for pillar in pillars})
    if not ordered:
        return []
    if group_after is None or valuation is None:
        return [(_label(pillar), [pillar]) for pillar in ordered]

    anchor = to_datetime(valuation)
    # every comparison is by *date*: a pillar may carry the valuation's time of
    # day (curves anchor on a timestamp) while a tenor roll lands at midnight
    cut = to_date(shift_tenor(anchor, str(group_after), calendar))
    kept = _trim_to_horizon(ordered, horizon)
    groups: List[Tuple[str, List[Any]]] = [
        (_label(pillar), [pillar]) for pillar in kept if to_date(pillar) <= cut
    ]
    tail = [pillar for pillar in kept if to_date(pillar) > cut]
    for band, members in _year_bands(tail, anchor, calendar):
        groups.append((_label(band), members))
    return groups


def bucket_grid_info(
    market: MarketState,
    settings: RiskSettings,
    *,
    horizon: Optional[DateLike] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """What the auto bucket grid does: pillars in, buckets out, what was dropped.

    Pure bookkeeping for the report - no valuations - so a risk run can say
    "6 buckets from 8 pillars, 2 dropped beyond 2026-10-05" instead of leaving the
    reader to wonder where the rest of the curve went.
    """
    explicit = bucketed_delta_pillars is not None
    pillars = borrow_buckets(market, bucketed_delta_pillars)
    groups = _auto_groups(pillars, market, settings, horizon, explicit=explicit)
    assigned = sum(len(members) for _, members in groups)
    return {
        "pillars": len(pillars),
        "buckets": len(groups),
        "dropped": max(len(pillars) - assigned, 0),
        "group_after": (
            None
            if explicit
            else getattr(settings, "bucket_group_after", DEFAULT_BUCKET_GROUP_AFTER)
        ),
        "horizon": None if horizon is None else _label(horizon),
        "explicit": explicit,
    }


def _auto_groups(
    pillars: Sequence[Any],
    market: MarketState,
    settings: RiskSettings,
    horizon: Optional[DateLike],
    *,
    explicit: bool,
) -> List[Tuple[str, List[Any]]]:
    """The buckets a run bumps: the caller's grid as given, else the coarse one."""
    if explicit:
        return [(_label(pillar), [pillar]) for pillar in pillars]
    return bucket_grid(
        pillars,
        valuation=getattr(market, "valuation_date", None),
        horizon=horizon,
        group_after=getattr(settings, "bucket_group_after", DEFAULT_BUCKET_GROUP_AFTER),
        calendar=getattr(market, "calendar", None),
    )


def _trim_to_horizon(dates: Sequence[Any], horizon: Optional[DateLike]) -> List[Any]:
    """Pillars the trade can see: up to ``horizon``, plus the one bracketing it."""
    if horizon is None:
        return list(dates)
    limit = to_date(horizon)
    kept = [day for day in dates if to_date(day) <= limit]
    after = [day for day in dates if to_date(day) > limit]
    if after:
        kept.append(after[0])
    return kept


def _year_bands(dates: Sequence[Any], anchor: Any, calendar) -> List[Tuple[Any, List[Any]]]:
    """One bucket per year of tenor: ``(band end, members)``, ascending."""
    bands: Dict[Any, List[Any]] = {}
    for day in dates:
        year = 1
        while year < 100:  # a curve decades long is a data error, not a grid
            edge = shift_tenor(anchor, "{}Y".format(year), calendar)
            if to_date(day) <= to_date(edge):
                break
            year += 1
        bands.setdefault(edge, []).append(day)
    return sorted(bands.items())


def with_bucketed_curves(
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


# -------------------------------------------------------------------- stencils
def vega_bucket(
    value: ValueFn,
    market: MarketState,
    pillar: Any,
    settings: RiskSettings,
) -> float:
    """One per-pillar vega bump (``RiskSettings.bucketed_vega_method`` picks the
    point-by-point or the cumulative-backward surface shift)."""
    step = float(settings.vega_bump)
    surface = require_surface(market)
    if step <= 0.0:
        return 0.0
    if settings.bucketed_vega_method.strip().lower() in _CUMULATIVE_METHODS:
        up = value(market.clone(surface=surface.bump_cumulative_backward(pillar, step)))
        down = value(market.clone(surface=surface.bump_cumulative_backward(pillar, -step)))
    else:
        up = value(market.clone(surface=surface.bump_pillar(pillar, step)))
        down = value(market.clone(surface=surface.bump_pillar(pillar, -step)))
    bucket = (up - down) / (2.0 * step)
    if settings.report_vega_per_vol_point:
        bucket /= 100.0
    return float(bucket)


def curve_group_bucket(
    value: ValueFn,
    market: MarketState,
    pillars: Sequence[Any],
    step: float,
    curve_name: str,
    *,
    per_pct: bool = True,
) -> float:
    """One bump pair for a **group** of curve pillars - a coarse bucket.

    The group is bumped by chaining single-pillar shifts, which is exact for these
    curves (``rates[i] += amount``), so the buckets still add up to the parallel
    Greek across the pillars the grid kept.
    """
    if step <= 0.0 or not pillars:
        return 0.0
    up = value(_bump_pillars(market, curve_name, pillars, step))
    down = value(_bump_pillars(market, curve_name, pillars, -step))
    bucket = (up - down) / (2.0 * step)
    return float(bucket / 100.0 if per_pct else bucket)


def curve_bucket(
    value: ValueFn,
    market: MarketState,
    pillar: Any,
    step: float,
    curve_name: str,
    *,
    per_pct: bool = True,
) -> float:
    """One per-pillar curve bump (``curve_name`` is ``rate`` or ``borrow``)."""
    return curve_group_bucket(value, market, [pillar], step, curve_name, per_pct=per_pct)


def _bump_pillars(
    market: MarketState, curve_name: str, pillars: Sequence[Any], amount: float
) -> MarketState:
    """The same shift on several pillars: chained single-pillar bumps."""
    bumped = market
    for pillar in pillars:
        bumped = curve_bump(bumped, curve_name, pillar, amount)
    return bumped


def convert_bucketed_delta(
    bucketed_rhoq: Dict[str, float],
    delta_cash: Optional[float],
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


# --------------------------------------------------------------------- driver
def bucket_greeks(
    value: ValueFn,
    market: MarketState,
    settings: RiskSettings,
    *,
    delta_cash: Optional[float] = None,
    which: Optional[Iterable[str]] = None,
    bucketed_vega_pillars: Optional[Iterable] = None,
    bucketed_delta_pillars: Optional[Iterable] = None,
    horizon: Optional[DateLike] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the selected bucket bumps and return ``{greek: {label: value}}``.

    ``which`` defaults to the selection in ``RiskSettings.greeks``; ``None``
    there means "parallel Greeks only", so an ordinary risk run keeps paying
    nothing for buckets.  Unrequested buckets come back empty rather than
    ``None`` - the ``PricingResult`` fields are dicts.

    ``horizon`` is the last date the trade can see (its final payment): the auto
    grid stops there and coarsens beyond ``RiskSettings.bucket_group_after``, so
    ``bucketed_delta`` on a long-dated snowball costs a handful of bump pairs
    instead of one valuation pair per curve pillar (see :func:`bucket_grid`).
    ``bucketed_vega`` keeps one bucket per surface expiry - there are few of them,
    and the surface is exactly what those buckets are for.
    """
    wanted = _requested(settings, which)
    result: Dict[str, Dict[str, float]] = {name: {} for name in BUCKET_NAMES}
    if not wanted:
        return result

    if "bucketed_vega" in wanted:
        for pillar in vol_pillars(market, bucketed_vega_pillars):
            result["bucketed_vega"][_label(pillar)] = vega_bucket(
                value, market, pillar, settings
            )

    need_borrow = wanted.intersection({"bucketed_rhoq", "bucketed_delta"})
    borrow = borrow_buckets(market, bucketed_delta_pillars) if need_borrow else []
    rates = rate_pillars(market) if "bucketed_rho" in wanted else []
    bucketed_market = with_bucketed_curves(market, borrow, rates)

    for label, group in _auto_groups(
        borrow, market, settings, horizon, explicit=bucketed_delta_pillars is not None
    ):
        result["bucketed_rhoq"][label] = curve_group_bucket(
            value,
            bucketed_market,
            group,
            float(settings.borrow_bump),
            "borrow",
            per_pct=settings.report_rho_per_pct,
        )

    if "bucketed_delta" in wanted:
        if delta_cash is None:
            delta_cash = spot_cash_delta(value, market, settings)
        result["bucketed_delta"] = convert_bucketed_delta(
            result["bucketed_rhoq"], delta_cash
        )

    for label, group in _auto_groups(rates, market, settings, horizon, explicit=False):
        result["bucketed_rho"][label] = curve_group_bucket(
            value,
            bucketed_market,
            group,
            float(settings.rate_bump),
            "rate",
            per_pct=settings.report_rho_per_pct,
        )
    return result


def spot_cash_delta(
    value: ValueFn, market: MarketState, settings: RiskSettings
) -> Optional[float]:
    """The spot cash delta ``bucketed_delta`` is distributed from.

    Only needed when the caller asks for ``bucketed_delta`` without the spot
    pair (``bump_greeks`` hands its ``delta_cash`` over otherwise, so a run that
    asks for both pays for the pair once).
    """
    step = abs(float(market.spot)) * float(settings.delta_bump_pct)
    if step <= 0.0:
        return None
    up = float(value(spot_bump(market, step)))
    down = float(value(spot_bump(market, -step)))
    return (up - down) / (2.0 * step) * float(market.spot)


def _requested(settings: RiskSettings, which: Optional[Iterable[str]]) -> Set[str]:
    """The bucket names to compute (``RiskSettings.greeks = None`` -> none)."""
    if which is not None:
        selected = set(parse_greeks(which))
    else:
        selection = getattr(settings, "greeks", None)
        selected = set() if selection is None else set(parse_greeks(selection))
    return selected.intersection(BUCKET_NAMES)


def _label(pillar: Any) -> str:
    return to_datetime(pillar).date().isoformat()


__all__ = [
    "BUCKET_NAMES",
    "DEFAULT_BUCKET_GROUP_AFTER",
    "DEFAULT_TIME_BUCKETED_TENORS",
    "borrow_buckets",
    "bucket_greeks",
    "bucket_grid",
    "bucket_grid_info",
    "convert_bucketed_delta",
    "curve_bucket",
    "curve_group_bucket",
    "default_bucket_grid",
    "rate_pillars",
    "spot_cash_delta",
    "vega_bucket",
    "vol_pillars",
    "with_bucketed_curves",
]
