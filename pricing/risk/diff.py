"""Shared bump-and-revalue Greeks for the exotic engines.

Both engines difference the *same* bumps (relative spot, parallel vol, parallel
rate / borrow, valuation date +1D) with the standard risk convention, so a
cross-check compares the pricing methods rather than two different setups.  The
Monte Carlo engine additionally reuses its common random numbers inside
``value``, which keeps the finite differences quiet.

``RiskSettings.greeks`` selects **which** Greeks a risk run computes: the
skipped bumps are neither evaluated nor returned (their result fields stay
``None``), so ``price_autocall --greeks delta,vega`` costs a fraction of a full
run.  ``None`` keeps the library behaviour of computing every parallel Greek;
the bucketed ones (:data:`BUCKET_NAMES`, applied by
:mod:`surface_pricer.pricing.risk.buckets`) stay opt-in because each bucket is a
bump pair.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .bumps import parallel_bump, spot_bump, vol_bump
from ..results import RiskSettings

ValueFn = Callable[[Any], float]

#: Every Greek :func:`bump_greeks` can produce, in report order.  ``delta_cash``
#: and ``delta_n`` ride on the same spot bumps as ``delta``, and ``gamma`` /
#: ``gamma_cash`` reuse them as well - asking for any of the five costs exactly
#: one pair.
GREEK_NAMES: Tuple[str, ...] = (
    "delta",
    "delta_cash",
    "delta_n",
    "gamma",
    "gamma_cash",
    "vega",
    "theta",
    "rho",
    "rhoq",
)

#: Bucketed Greeks (see :mod:`surface_pricer.pricing.risk.buckets`).  They are
#: **not** part of ``all``: each bucket costs a bump pair, and on the exotic side
#: a per-pillar vol bump also rebuilds the cached Dupire table.
BUCKET_NAMES: Tuple[str, ...] = (
    "bucketed_vega",
    "bucketed_rhoq",
    "bucketed_rho",
    "bucketed_delta",
)

#: Second-order Greeks ``bump_greeks`` differences on **both** sides (2026-10; they
#: used to be vanilla-only and the exotic engines refused them).  ``volga`` is the
#: second central difference on the **vega pair's** vol states - so asking for it
#: beside ``vega`` costs no valuation and no Dupire table - and ``vanna`` is the
#: four-state crossed difference, which reuses the same vol states.  Both are part
#: of ``all`` (2026-10); the **buckets** stay opt-in, because each one is a bump
#: pair of its own.
SECOND_ORDER_GREEKS = ("volga", "vanna")

#: What ``all`` (and the library default, ``RiskSettings.greeks=None``) means: every
#: parallel Greek, the second-order ones included, and **no** buckets.
ALL_PARALLEL_GREEKS: Tuple[str, ...] = GREEK_NAMES + SECOND_ORDER_GREEKS

#: Everything :func:`parse_greeks` accepts (``all`` is :data:`ALL_PARALLEL_GREEKS`).
ALL_GREEK_NAMES: Tuple[str, ...] = ALL_PARALLEL_GREEKS + BUCKET_NAMES

#: The Greeks that share the spot bump pair.
SPOT_GREEKS = ("delta", "delta_cash", "delta_n", "gamma", "gamma_cash")

#: Accepted spellings that expand to a set of :data:`GREEK_NAMES`.
_GREEK_ALIASES: Dict[str, Tuple[str, ...]] = {
    "all": ALL_PARALLEL_GREEKS,
    "none": (),
    "npv": (),  # the base valuation is always computed
    "delta_shares": ("delta_n",),
    "buckets": BUCKET_NAMES,
}

#: Bump/reporting convention stamped into ``PricingResult.metadata``.
GREEK_CONVENTION = {
    "delta": "bump-and-revalue dNPV/dSpot",
    "gamma_cash": "d2NPV/dSpot2 * spot^2 / 100 (NPV change per (1% spot move)^2)",
    "vega": "surface parallel bump, reported per 1 vol point",
    "volga": "surface parallel bump, second central difference, reported per (1 vol point)^2",
    "vanna": "spot x vol cross difference (4 states), reported per 1 vol point",
    "rho": "rate curve parallel bump, reported per 1%",
    "rhoq": "borrow curve parallel bump, reported per 1%",
    "theta": "valuation date + theta_days, same trade re-priced, per day",
    "bucketed_vega": "one surface expiry at a time, per 1 vol point",
    "bucketed_rhoq": "one borrow pillar at a time, per 1% borrow",
    "bucketed_rho": "one rate pillar at a time, per 1% rate",
    "bucketed_delta": "delta_cash split over the borrow buckets (sums to delta_cash)",
}


def parse_greeks(selection: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Normalise a Greek selection (``"delta, vega"``, ``["all"]``, ``None``).

    ``None`` and an empty selection mean **no** Greeks - the base NPV is always
    priced - which is the CLI default (``price_autocall --greeks``).  The
    library default is different on purpose: :attr:`RiskSettings.greeks` left at
    ``None`` means *all* Greeks, so existing callers keep their full risk run.

    Raises :class:`ValueError` for an unknown name (the CLI turns that into a
    clean error message).
    """
    if selection is None:
        return ()
    items = (
        selection.replace(",", " ").split()
        if isinstance(selection, str)
        else [str(item) for item in selection]
    )
    selected = []
    for raw in items:
        key = raw.strip().lower()
        if not key:
            continue
        names = _GREEK_ALIASES.get(key)
        if names is None:
            if key not in ALL_GREEK_NAMES:
                raise ValueError(
                    "unknown greek '{}' (choose from {}, all, none, buckets)".format(
                        raw.strip(), ", ".join(ALL_GREEK_NAMES)
                    )
                )
            names = (key,)
        for name in names:
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def selected_greeks(settings: RiskSettings) -> Set[str]:
    """The Greeks this settings object asks for (``None`` -> ``all``).

    ``None`` means exactly what ``all`` spells (the parallel Greeks including
    ``volga`` / ``vanna``, and no buckets), on **both** sides - so the vanilla
    pricer and the exotic engines answer the same request with the same set.
    Shared with the vanilla pricer, which runs its own stencils but must honour the
    same selection convention.
    """
    selection = getattr(settings, "greeks", None)
    if selection is None:
        return set(ALL_PARALLEL_GREEKS)
    return set(parse_greeks(selection))


def bump_greeks(
    value: ValueFn,
    market: Any,
    settings: RiskSettings,
    base: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Bump-and-revalue Greeks on top of ``value(market) -> npv``.

    ``base`` avoids one valuation when the caller already priced the market
    (the MC engine needs the base run anyway to keep its normals warm); the
    requested subset comes from ``settings.greeks``.
    """
    npv = float(value(market)) if base is None else float(base)
    wanted = selected_greeks(settings)
    unsupported = wanted.difference(ALL_GREEK_NAMES)
    if unsupported:
        raise ValueError(
            "unknown Greek(s) {}: choose from {}".format(
                ", ".join(sorted(unsupported)), ", ".join(ALL_GREEK_NAMES)
            )
        )
    result: Dict[str, Optional[float]] = {
        "npv": npv,
        "delta": None,
        "delta_cash": None,
        "delta_n": None,
        "gamma": None,
        "gamma_cash": None,
        "vega": None,
        "volga": None,
        "vanna": None,
        "rho": None,
        "rhoq": None,
        "theta": None,
    }

    # One market per bumped **vol amount**, built once: the exotic engines cache their
    # Dupire table on the bumped surface object, so re-bumping the same amount would
    # pay for a second table.  ``volga`` and ``vanna`` therefore reuse the states the
    # vega pair already built (their bump sizes default to the same 0.5 vol point).
    vol_states: Dict[float, Any] = {}

    def vol_state(amount: float) -> Any:
        if amount not in vol_states:
            vol_states[amount] = vol_bump(market, amount)
        return vol_states[amount]

    spot = float(market.spot)
    spot_step = abs(spot) * float(settings.delta_bump_pct)
    if spot_step > 0.0 and wanted.intersection(SPOT_GREEKS):
        up = float(value(spot_bump(market, spot_step)))
        down = float(value(spot_bump(market, -spot_step)))
        delta = (up - down) / (2.0 * spot_step)
        if wanted.intersection(("delta", "delta_cash", "delta_n")):
            result["delta"] = delta
            result["delta_n"] = delta / spot
            result["delta_cash"] = delta * spot
        if wanted.intersection(("gamma", "gamma_cash")):
            # the family rides together, exactly like delta / delta_cash / delta_n:
            # asking for gamma hands back its cash form too, and vice versa
            gamma = (up - 2.0 * npv + down) / (spot_step ** 2)
            result["gamma"] = gamma
            result["gamma_cash"] = gamma * spot ** 2 / 100.0

    vol_step = float(settings.vega_bump)
    if vol_step > 0.0 and wanted.intersection(("vega", "volga")):
        up = float(value(vol_state(vol_step)))
        down = float(value(vol_state(-vol_step)))
        if "vega" in wanted:
            vega = (up - down) / (2.0 * vol_step)
            result["vega"] = vega / 100.0 if settings.report_vega_per_vol_point else vega
        if "volga" in wanted:
            step = float(settings.volga_bump)
            if abs(step - vol_step) < 1.0e-15:
                volga_up, volga_down = up, down
            else:
                volga_up = float(value(vol_state(step)))
                volga_down = float(value(vol_state(-step)))
            volga = (volga_up - 2.0 * npv + volga_down) / (step ** 2)
            result["volga"] = (
                volga / 10000.0 if settings.report_vega_per_vol_point else volga
            )

    rate_step = float(settings.rate_bump)
    if rate_step > 0.0 and "rho" in wanted:
        up = float(value(parallel_bump(market, "rate", rate_step)))
        down = float(value(parallel_bump(market, "rate", -rate_step)))
        rho = (up - down) / (2.0 * rate_step)
        result["rho"] = rho / 100.0 if settings.report_rho_per_pct else rho

    borrow_step = float(settings.borrow_bump)
    if borrow_step > 0.0 and "rhoq" in wanted:
        up = float(value(parallel_bump(market, "borrow", borrow_step)))
        down = float(value(parallel_bump(market, "borrow", -borrow_step)))
        rhoq = (up - down) / (2.0 * borrow_step)
        result["rhoq"] = rhoq / 100.0 if settings.report_rho_per_pct else rhoq

    if "theta" in wanted:
        days = max(int(settings.theta_days), 1)
        theta_market = market.clone(
            valuation_date=market.valuation_date + timedelta(days=days)
        )
        result["theta"] = (float(value(theta_market)) - npv) / days

    if "vanna" in wanted and spot_step > 0.0:
        vanna_step = float(settings.vanna_vol_bump)
        if vanna_step > 0.0:
            # the crossed spot x vol states, on the **same** vol states volga/vega
            # use - so the extra cost is four valuations and no extra Dupire table
            up_up = float(value(spot_bump(vol_state(vanna_step), spot_step)))
            up_down = float(value(spot_bump(vol_state(-vanna_step), spot_step)))
            down_up = float(value(spot_bump(vol_state(vanna_step), -spot_step)))
            down_down = float(value(spot_bump(vol_state(-vanna_step), -spot_step)))
            vanna = (up_up - up_down - down_up + down_down) / (
                4.0 * spot_step * vanna_step
            )
            result["vanna"] = (
                vanna / 100.0 if settings.report_vega_per_vol_point else vanna
            )
    return result


def apply_greeks(result: Any, greeks: Mapping[str, Optional[float]]) -> Any:
    """Copy a :func:`bump_greeks` mapping onto a :class:`PricingResult`.

    Centralised on purpose: an engine that copies field by field silently drops a
    Greek the moment one is added (``gamma_cash`` went missing on exactly one of
    the two engines that way).  The second-order Greeks ride along - both engines
    report ``volga`` / ``vanna`` when the selection asked for them.
    """
    for name in GREEK_NAMES + SECOND_ORDER_GREEKS:
        if name in greeks:
            setattr(result, name, greeks[name])
    return result


__all__ = [
    "ALL_GREEK_NAMES",
    "ALL_PARALLEL_GREEKS",
    "BUCKET_NAMES",
    "GREEK_CONVENTION",
    "GREEK_NAMES",
    "SECOND_ORDER_GREEKS",
    "SPOT_GREEKS",
    "ValueFn",
    "apply_greeks",
    "bump_greeks",
    "parse_greeks",
    "selected_greeks",
]
