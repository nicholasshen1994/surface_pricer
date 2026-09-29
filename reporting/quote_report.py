"""Text / dict rendering of a single-option quote (NPV plus Greeks)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..pricing.results import PricingResult

_GREEK_ROWS = (
    ("delta", "delta (dNPV/dSpot)"),
    ("delta_cash", "delta cash (delta x spot)"),
    ("delta_n", "delta shares (delta / spot)"),
    ("gamma", "gamma (d2NPV/dSpot2)"),
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


def quote_to_dict(
    result: PricingResult,
    *,
    run: Optional[str] = None,
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Machine-readable quote (used by ``price_tool --json``)."""
    payload: Dict[str, Any] = {
        "run": run,
        "contract": contract or {},
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
        "greek_convention": result.metadata.get("greek_convention", {}),
    }
    return payload


def format_quote(
    result: PricingResult,
    *,
    run: Optional[str] = None,
    contract: Optional[Dict[str, Any]] = None,
    indent: str = "  ",
) -> str:
    """Human readable quote block for the command line."""
    lines = []
    if run:
        lines.append("fit run    : {}".format(run))
    if contract:
        lines.append(
            "contract   : {} {} | strike={} tenor={} expiry={} | notional={}".format(
                contract.get("option_type", "?"),
                contract.get("strike_type", "absolute"),
                _num(contract.get("strike")),
                contract.get("tenor", "-"),
                contract.get("expiry", "-"),
                _num(contract.get("notional", 1.0)),
            )
        )
    lines.append(
        "market     : forward={} | df={} | implied vol={}".format(
            _num(result.forward, 4),
            _num(result.discount_factor, 6),
            _pct(result.implied_vol),
        )
    )
    lines.append(
        "npv        : {}{}".format(
            _num(result.npv, 6),
            "" if contract is None else "  (notional={})".format(_num(contract.get("notional", 1.0))),
        )
    )
    lines.append("greeks     :")
    for name, label in _GREEK_ROWS:
        value = getattr(result, name, None)
        lines.append("{}{:<34}{}".format(indent, label, _num(value, 8)))

    buckets = {
        name: dict(getattr(result, name, {}) or {}) for name, _ in _BUCKET_ROWS
    }
    if any(buckets.values()):
        lines.append("bucketed   :")
        for name, label in _BUCKET_ROWS:
            values = buckets.get(name) or {}
            if not values:
                continue
            lines.append("{}{}:".format(indent, label))
            for key, value in values.items():
                lines.append("{}{}  {:<14}{}".format(indent * 2, key, "", _num(value, 8)))
    return "\n".join(lines)


def _num(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return "{:,.{digits}g}".format(float(value), digits=digits)
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "{:.4f}%".format(float(value) * 100.0)


__all__ = ["format_quote", "quote_to_dict"]
