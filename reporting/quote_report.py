"""Text / dict rendering of a single-option quote (NPV plus Greeks).

The contract block is the **resolved** payload
(:meth:`VanillaSpec.to_dict <surface_pricer.pricing.vanilla.spec.VanillaSpec.to_dict>`)
- absolute strike, absolute expiry, the market mapping the price used - i.e. what
the pricer actually saw, not the raw term sheet.  ``quote_to_dict`` output can
therefore be edited and fed back through ``VanillaSpec.from_dict``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..pricing.results import PricingResult
from ..pricing.vanilla import VanillaSpec

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

_BUCKET_ROWS = (
    ("bucketed_vega", "bucketed vega"),
    ("bucketed_rhoq", "bucketed rhoQ"),
    ("bucketed_delta", "bucketed delta"),
    ("bucketed_rho", "bucketed rho"),
)

#: What a relative strike is a ratio *of* (``VanillaContract.absolute_strike``).
_STRIKE_BASES = {
    "percentage": "spot",
    "fwd_percentage": "forward",
}


def quote_to_dict(
    result: PricingResult,
    *,
    run: Optional[str] = None,
    spec: Optional[VanillaSpec] = None,
) -> Dict[str, Any]:
    """Machine-readable quote (used by ``price-tool --json``)."""
    payload: Dict[str, Any] = {
        "run": run,
        "contract": {} if spec is None else spec.to_dict(),
        "npv": result.npv,
        "forward": result.forward,
        "discount_factor": result.discount_factor,
        "implied_vol": result.implied_vol,
        "strike": result.strike,
        "year_fraction": result.year_fraction,
        "greeks": {name: getattr(result, name, None) for name, _ in _GREEK_ROWS},
        "bucketed": {
            name: dict(getattr(result, name, {}) or {}) for name, _ in _BUCKET_ROWS
        },
        "bucket_grid": result.metadata.get("bucket_grid", {}),
        "greek_convention": result.metadata.get("greek_convention", {}),
    }
    return payload


def format_quote(
    result: PricingResult,
    *,
    run: Optional[str] = None,
    spec: Optional[VanillaSpec] = None,
    request: Optional[Dict[str, Any]] = None,
    indent: str = "  ",
) -> str:
    """Human readable quote block for the command line.

    ``request`` carries what the caller typed but the spec cannot know (a tenor
    instead of a date); it is provenance for the header line only.
    """
    lines = []
    if run:
        lines.append("fit run    : {}".format(run))
    if spec is not None:
        tenor = (request or {}).get("tenor")
        lines.append(
            "contract   : {} | strike={} | expiry={} | notional={}{}".format(
                spec.option_type,
                num(spec.strike, 8),
                spec.expiry_date.date().isoformat(),
                num(spec.notional),
                "" if not tenor else " | tenor={}".format(tenor),
            )
        )
        provenance = _strike_provenance(spec)
        if provenance:
            lines.append("strike     : {}".format(provenance))
    lines.append(
        "market     : forward={} | df={} | implied vol={}".format(
            num(result.forward, 4),
            num(result.discount_factor, 6),
            pct(result.implied_vol),
        )
    )
    lines.append(
        "npv        : {}{}".format(
            num(result.npv, 6),
            "" if spec is None else "  (notional={})".format(num(spec.notional)),
        )
    )
    lines.append("greeks     :")
    for name, label in _GREEK_ROWS:
        value = getattr(result, name, None)
        lines.append("{}{:<34}{}".format(indent, label, num(value, 8)))

    buckets = {
        name: dict(getattr(result, name, {}) or {}) for name, _ in _BUCKET_ROWS
    }
    if any(buckets.values()):
        lines.append("bucketed   :")
        grid = result.metadata.get("bucket_grid")
        if grid and grid.get("buckets"):
            lines.append("{}{}".format(indent, bucket_grid_text(grid)))
        for name, label in _BUCKET_ROWS:
            values = buckets.get(name) or {}
            if not values:
                continue
            lines.append("{}{}:".format(indent, label))
            for key, value in values.items():
                lines.append("{}{}  {:<14}{}".format(indent * 2, key, "", num(value, 8)))
    return "\n".join(lines)


def bucket_grid_text(grid: Dict[str, Any]) -> str:
    """One line about the coarse bucket grid (:func:`..risk.buckets.bucket_grid`)."""
    text = "{} bucket(s) from {} pillar(s)".format(
        grid.get("buckets", 0), grid.get("pillars", 0)
    )
    if grid.get("dropped"):
        text += ", {} dropped beyond {}".format(grid["dropped"], grid.get("horizon"))
    elif grid.get("group_after"):
        text += ", merged beyond {}".format(grid["group_after"])
    if grid.get("explicit"):
        text += " (pinned grid)"
    return text


def _strike_provenance(spec: VanillaSpec) -> str:
    """How a relative strike was resolved (empty for an absolute one).

    A payload's ``strike`` may have been edited by hand after the export; then the
    ``strike_input`` no longer reproduces it, and the line says so instead of
    printing arithmetic that does not add up.
    """
    base = _STRIKE_BASES.get(str(spec.strike_type).strip().lower())
    if base is None:
        return ""
    reference = spec.spot if base == "spot" else spec.forward
    resolved = float(spec.strike_input or 0.0) * reference
    if abs(resolved - spec.strike) > 1e-9 * max(abs(spec.strike), 1.0):
        return "strike_input={} ({}) - payload strike used as written".format(
            num(spec.strike_input, 8), spec.strike_type
        )
    return "{} = {} x {} {}".format(
        num(spec.strike, 8),
        num(spec.strike_input, 8),
        base,
        num(reference, 4),
    )


def num(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return "{:,.{digits}g}".format(float(value), digits=digits)
    except (TypeError, ValueError):
        return str(value)


def pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "{:.4f}%".format(float(value) * 100.0)


__all__ = ["bucket_grid_text", "format_quote", "num", "pct", "quote_to_dict"]
