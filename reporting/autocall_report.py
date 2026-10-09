"""Terminal / JSON rendering for an autocallable quote.

Shows three things side by side that a snowball always needs: the resolved terms
(``AutocallSchedule.to_dict()`` - exactly what the engines price and what a JSON
payload may replace), the shift rule with its source, and the **effective**
(absolute, post-shift) barriers with the per-observation coupons.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..pricing.exotics.autocall import AutocallSchedule
from ..pricing.results import PricingResult
from .quote_report import bucket_grid_text


def _number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return "{:,.{digits}f}".format(value, digits=digits)


def _coupon_text(rates) -> str:
    """One rate, or the whole step-up schedule when the observations differ."""
    values = [float(rate) for rate in rates]
    if not values:
        return "0.0000%"
    if max(values) - min(values) <= 1.0e-12:
        return "{:.4%}".format(values[0])
    return "{} (per observation)".format(
        ", ".join("{:.4%}".format(value) for value in values)
    )


#: The Greek rows the text report and the JSON print, in report order - one table,
#: so a Greek the engines fill cannot be quietly dropped by the renderer (``volga``
#: / ``vanna`` were, until 2026-10; the same reason ``apply_greeks`` loops over a
#: list on the engine side).  Labels match the vanilla report's.
_GREEK_ROWS = (
    ("delta", "delta (dNPV/dSpot)"),
    ("delta_cash", "delta cash (delta x spot)"),
    ("delta_n", "delta shares (delta / spot)"),
    ("gamma", "gamma (d2NPV/dSpot2)"),
    ("gamma_cash", "gamma cash (per (1% spot move)^2)"),
    ("vega", "vega (per 1 vol pt)"),
    ("volga", "volga (per (1 vol pt)^2)"),
    ("vanna", "vanna (per 1 vol pt)"),
    ("theta", "theta (per day)"),
    ("rho", "rho (per 1% rate)"),
    ("rhoq", "rhoQ (per 1% borrow)"),
)

#: The bucketed rows (a table for the same reason).
_BUCKET_ROWS = (
    ("bucketed_vega", "bucketed vega (per vol pt)"),
    ("bucketed_delta", "bucketed delta (cash)"),
    ("bucketed_rhoq", "bucketed rhoQ (per 1%)"),
    ("bucketed_rho", "bucketed rho (per 1%)"),
)


def format_autocall(
    schedule: AutocallSchedule,
    result: PricingResult,
    market: Any = None,
    *,
    with_greeks: bool = True,
) -> str:
    """Human-readable quote: resolved contract, effective terms, NPV, Greeks."""
    lines = []
    lines.append(
        "contract   : {} | {} | notional={:,.0f}".format(
            schedule.underlying, schedule.product_type, schedule.notional
        )
    )
    lines.append(
        "terms      : start={} expiry={} | {} observation(s)".format(
            schedule.start_date.date().isoformat() if schedule.start_date else "?",
            schedule.expiry_date.date().isoformat(),
            len(schedule.observation_dates),
        )
    )
    lines.append(
        "coupon     : ko={} | rebate={} (annual, {:.4%} of notional at expiry) | "
        "protection={:.2%} | accrual={} from {}".format(
            _coupon_text(schedule.coupon_rates),
            _coupon_text([schedule.rebate_rate or 0.0]),
            schedule.rebate_ratio,
            schedule.protected_principal,
            schedule.day_count,
            schedule.start_date.date().isoformat() if schedule.start_date else "?",
        )
    )
    lines.append(
        "ki         : {} | {} monitoring date(s) | strike={:,.2f} ({:.2%} of the anchor)"
        " | gearing={:.2f}".format(
            schedule.ki_frequency,
            len(schedule.ki_dates),
            schedule.ki_strike * schedule.spot0,
            schedule.ki_strike,
            schedule.ki_gearing,
        )
    )
    lines.append(
        "shift      : ko={} | ki={} | anchor={}".format(
            schedule.ko_shift.describe(),
            schedule.ki_shift.describe(),
            schedule.anchored_on,
        )
    )
    lines.append(
        "today      : basis={} (same-day knock-in / knock-out determination)".format(
            getattr(schedule, "trigger_basis", "contractual")
        )
    )
    lines.append("effective  :")
    # the raw (pre-shift) level is shown only when it differs: for a payload-fed
    # schedule the payload carries the effective level only
    shifted = any(
        abs(effective - raw) > 1.0e-9
        for effective, raw in zip(schedule.ko_levels, schedule.ko_levels_raw)
    )
    for index, day in enumerate(schedule.observation_dates):
        lines.append(
            "    {}  ko={:>10,.2f}{}  ki={:>10,.2f}  coupon={:.4%}".format(
                day.date().isoformat(),
                schedule.ko_levels[index],
                ""
                if not shifted
                else " (raw {:>10,.2f})".format(schedule.ko_levels_raw[index]),
                schedule.ki_levels[index],
                schedule.coupon_rates[index],
            )
        )
    lines.append(
        "market     : spot={:,.4f} spot0={:,.4f}" "{}".format(
            float(getattr(market, "spot", float("nan"))),
            schedule.spot0,
            ""
            if market is None
            else " | forward(expiry)={:,.4f} df={:.6f}".format(
                float(market.forward(schedule.expiry_date)),
                float(market.discount_factor(schedule.expiry_date)),
            ),
        )
    )
    if schedule.is_settled:
        lines.append(
            "status     : knocked out on {} | cash={:,.2f}".format(
                schedule.knocked_out_payment_date.date().isoformat()
                if schedule.knocked_out_payment_date
                else "?",
                float(schedule.knocked_out_cash or 0.0),
            )
        )
    elif getattr(schedule, "knocked_in_before", False):
        knocked_in = getattr(schedule, "knocked_in_date", None)
        lines.append(
            "status     : knocked in on {} | loss leg settles at expiry".format(
                knocked_in.date().isoformat() if knocked_in else "?"
            )
        )
    lines.append("npv        : {:,.2f}".format(result.npv))

    if with_greeks:
        lines.append("greeks     :")
        for name, label in _GREEK_ROWS:
            value = getattr(result, name, None)
            if value is not None:
                lines.append("  {:22s} {:>18,.8g}".format(label, value))
        for name, label in _BUCKET_ROWS:
            values = getattr(result, name, None) or {}
            if not values:
                continue
            lines.append("  {} :".format(label))
            for bucket_label, bucket in values.items():
                lines.append("    {:20s} {:>18,.8g}".format(bucket_label, bucket))

    grid = (result.metadata or {}).get("bucket_grid")
    if grid and grid.get("buckets"):
        lines.append("  bucket grid: {}".format(bucket_grid_text(grid)))

    metadata = result.metadata or {}
    summary = "method     : {}".format(metadata.get("method", "?"))
    if with_greeks:
        # The discretisation behind the numbers above: worth reading on a **risk**
        # run (that is where the grid is a choice you may want to compare), pure
        # noise on an NPV-only quote - ``--greeks none`` prints the engine and stops.
        if metadata.get("method") == "monte_carlo":
            summary += " | paths={} | steps={} | seed={} | std_error={:,.2f}".format(
                metadata.get("paths"),
                metadata.get("steps"),
                metadata.get("seed"),
                float(metadata.get("std_error", 0.0)),
            )
        elif metadata.get("method") == "pde":
            summary += " | nodes={} | steps={} | theta={}".format(
                metadata.get("nodes"), metadata.get("steps"), metadata.get("theta")
            )
    lines.append(summary)
    if with_greeks and metadata.get("grid_delta") is not None:
        # a delta by another route - a cross-check *for* ``--greeks delta``, not a
        # number to print when no Greek was asked for
        lines.append("             grid delta (cross-check) = {:,.4f}".format(
            float(metadata["grid_delta"])
        ))
    return "\n".join(lines)


def autocall_to_dict(
    schedule: AutocallSchedule,
    result: PricingResult,
) -> Dict[str, Any]:
    """Machine-readable quote (stable keys, JSON friendly).

    ``contract`` is :meth:`AutocallSchedule.to_dict`, i.e. the resolved terms, so
    the payload can be edited and fed straight back to the engines through
    ``AutocallSchedule.from_dict``.
    """
    metadata = dict(result.metadata or {})
    payload: Dict[str, Any] = {
        "contract": schedule.to_dict(),
        "effective": {
            "spot0": float(schedule.spot0),
            "anchored_on": schedule.anchored_on,
            "observation_dates": [
                day.date().isoformat() for day in schedule.observation_dates
            ],
            "ko_levels": [float(value) for value in schedule.ko_levels],
            "ki_levels": [float(value) for value in schedule.ki_levels],
            "coupon_rates": [float(value) for value in schedule.coupon_rates],
            "rebate_rate": float(schedule.rebate_rate or 0.0),
            "rebate_ratio": float(schedule.rebate_ratio),
            "payment_dates": [
                day.date().isoformat() for day in schedule.payment_dates
            ],
            # the same machine-readable rule the payload carries, so the mirror and
            # the contract never drift apart
            "shift": {
                "ko": schedule.ko_shift.to_dict(),
                "ki": schedule.ki_shift.to_dict(),
                "elapsed": int(schedule.shift_elapsed),
                "notes": list(schedule.notes),
            },
            "trigger_basis": getattr(schedule, "trigger_basis", "contractual"),
            "knocked_in_date": (
                None
                if getattr(schedule, "knocked_in_date", None) is None
                else schedule.knocked_in_date.date().isoformat()
            ),
            "knocked_out": bool(schedule.is_settled),
        },
        "npv": float(result.npv),
        "greeks": {name: getattr(result, name, None) for name, _ in _GREEK_ROWS},
        "bucketed": {
            name: dict(getattr(result, name, {}) or {}) for name, _ in _BUCKET_ROWS
        },
        "method": metadata,
    }
    return payload


__all__ = ["autocall_to_dict", "format_autocall"]
