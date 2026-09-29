"""CSV / JSON reports for a portfolio valuation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .valuation import PortfolioValuation, TradeValuation

CSV_COLUMNS = (
    "trade_id",
    "underlying",
    "product_type",
    "status",
    "expiry_date",
    "call_put",
    "strike",
    "strike_type",
    "notional",
    "npv",
    "delta",
    "delta_cash",
    "delta_n",
    "gamma",
    "vega",
    "theta",
    "vanna",
    "volga",
    "rho",
    "rhoq",
    "ki_flag",
    "ko_flag",
    "ki_triggered",
    "ko_triggered",
    "bucketed_vega",
    "bucketed_delta",
    "bucketed_rhoq",
    "bucketed_rho",
    "message",
)

JSON_BUCKET_KEYS = ("vega", "delta", "rhoq", "rho")


def trade_to_row(trade: TradeValuation) -> Dict[str, Any]:
    terms = trade.terms
    greeks = trade.greeks()
    buckets = trade.buckets()
    row: Dict[str, Any] = {
        "trade_id": terms.trade_id,
        "underlying": terms.underlying,
        "product_type": terms.product_type,
        "status": trade.status,
        "expiry_date": terms.expiry_date.date().isoformat(),
        "call_put": terms.call_put,
        "strike": terms.strike,
        "strike_type": terms.strike_type,
        "notional": terms.notional,
        "npv": trade.npv,
        "ki_flag": terms.ki_flag,
        "ko_flag": terms.ko_flag,
        "ki_triggered": trade.schedule.ki_triggered,
        "ko_triggered": trade.schedule.ko_triggered,
        "message": trade.message,
    }
    row.update(greeks)
    for field_name in ("bucketed_vega", "bucketed_delta", "bucketed_rhoq", "bucketed_rho"):
        values = buckets.get(field_name, {})
        row[field_name] = json.dumps(values, sort_keys=True) if values else ""
    return row


def trades_to_rows(portfolio: PortfolioValuation) -> List[Dict[str, Any]]:
    return [trade_to_row(trade) for trade in portfolio.trades]


def write_csv(portfolio: PortfolioValuation, path: str) -> Path:
    """Write one row per trade; missing Greeks are left blank."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in trades_to_rows(portfolio):
            writer.writerow({key: _format_cell(row.get(key)) for key in CSV_COLUMNS})
    return target


def portfolio_to_dict(portfolio: PortfolioValuation) -> Dict[str, Any]:
    return {
        "valuation_date": portfolio.valuation_date.isoformat(),
        "counts_by_status": portfolio.counts_by_status(),
        "totals": portfolio.totals,
        "bucketed_totals": portfolio.bucketed_totals(),
        "trades": [
            {
                "terms": trade.terms.to_dict(),
                "status": trade.status,
                "npv": trade.npv,
                "greeks": trade.greeks(),
                "bucketed": trade.buckets(),
                "schedule": {
                    "status": trade.schedule.status,
                    "observation_dates": [
                        value.isoformat() for value in trade.schedule.observation_dates
                    ],
                    "past_observations": [
                        value.isoformat() for value in trade.schedule.past_observations
                    ],
                    "next_observation": (
                        trade.schedule.next_observation.isoformat()
                        if trade.schedule.next_observation
                        else None
                    ),
                    "ki_triggered": trade.schedule.ki_triggered,
                    "ko_triggered": trade.schedule.ko_triggered,
                    "ki_source": trade.schedule.ki_source,
                    "ko_source": trade.schedule.ko_source,
                    "notes": list(trade.schedule.notes),
                },
                "message": trade.message,
            }
            for trade in portfolio.trades
        ],
    }


def write_json(portfolio: PortfolioValuation, path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(portfolio_to_dict(portfolio), indent=2, default=str),
        encoding="utf-8",
    )
    return target


def format_summary(portfolio: PortfolioValuation) -> str:
    """Human readable summary for the command line."""
    lines: List[str] = []
    counts = portfolio.counts_by_status()
    lines.append(
        "portfolio | valuation={} | {} trade(s) [{} active, {} expired, {} not started]".format(
            portfolio.valuation_date.isoformat(),
            len(portfolio.trades),
            counts.get("active", 0),
            counts.get("expired", 0),
            counts.get("not_started", 0),
        )
    )
    totals = portfolio.totals
    lines.append(
        "totals    | npv={:,.2f} | delta={} | vega={} | rhoq={}".format(
            totals.get("npv", 0.0),
            _fmt(totals.get("delta")),
            _fmt(totals.get("vega")),
            _fmt(totals.get("rhoq")),
        )
    )
    lines.append(
        "greeks    | delta_cash={} | delta_n={} | gamma={} | theta={} | vanna={} | volga={} | rho={}".format(
            _fmt(totals.get("delta_cash")),
            _fmt(totals.get("delta_n")),
            _fmt(totals.get("gamma")),
            _fmt(totals.get("theta")),
            _fmt(totals.get("vanna")),
            _fmt(totals.get("volga")),
            _fmt(totals.get("rho")),
        )
    )
    for trade in portfolio.trades:
        lines.append(
            "  {:<14} {:<10} {:<12} npv={}".format(
                trade.trade_id,
                trade.terms.underlying,
                trade.status,
                _fmt(trade.npv, money=True),
            )
        )
        if trade.message:
            lines.append("      {}".format(trade.message))
    return "\n".join(lines)


def _format_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _fmt(value: Optional[float], money: bool = False) -> str:
    if value is None:
        return "-"
    if money:
        return "{:,.2f}".format(float(value))
    return "{:.6g}".format(float(value))


__all__ = [
    "CSV_COLUMNS",
    "format_summary",
    "portfolio_to_dict",
    "trade_to_row",
    "trades_to_rows",
    "write_csv",
    "write_json",
]
