"""Text / dict / CSV rendering of a spot slide (``pricing.risk.slide``).

The text form is the desk table: one line per rung, the base row marked, the
requested Greeks as columns.  :func:`slide_to_dict` is the machine-readable twin
(the ``--json`` payload) and :func:`slide_csv` the spreadsheet one, so a slide can
be looked at, charted or loaded into a sheet without reformatting anything.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..pricing.risk.slide import SlideRow
from .quote_report import num

#: Column title per Greek name - the wording of the quote's Greek rows.
GREEK_LABELS: Mapping[str, str] = {
    "delta": "delta",
    "delta_cash": "delta cash",
    "delta_n": "delta shares",
    "gamma": "gamma",
    "gamma_cash": "gamma cash",
    "vega": "vega",
    "volga": "volga",
    "vanna": "vanna",
    "theta": "theta",
    "rho": "rho",
    "rhoq": "rhoQ",
}

#: The columns before the Greeks, in order.
_FIXED_COLUMNS = ("spot", "bump", "npv")


def slide_to_dict(
    rows: Iterable[SlideRow],
    *,
    run: Optional[str] = None,
    contract: Optional[Mapping[str, Any]] = None,
    base_spot: Optional[float] = None,
    greeks: Sequence[str] = (),
    span: Optional[float] = None,
    step: Optional[float] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Machine-readable slide (used by ``price-json --slide --json``).

    Self-describing like the contract payloads: ``kind`` names the shape, the
    contract block is the resolved payload that was priced, and ``span`` / ``step``
    record the ladder the rows came from (``None`` when the rungs were given
    explicitly).
    """
    return {
        "kind": "spot_slide",
        "run": run,
        "contract": {} if contract is None else dict(contract),
        "base_spot": base_spot,
        "span": span,
        "step": step,
        "greeks": list(greeks),
        "method": method,
        "rows": [row.to_dict() for row in rows],
    }


def slide_csv(rows: Iterable[SlideRow], greeks: Sequence[str]) -> str:
    """The rows as CSV (header plus one line per rung)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*_FIXED_COLUMNS, *greeks])
    for row in rows:
        writer.writerow(
            [row.spot, row.bump, row.npv, *[row.greeks.get(name, "") for name in greeks]]
        )
    return buffer.getvalue()


def format_slide(
    rows: Sequence[SlideRow],
    greeks: Sequence[str] = (),
    *,
    lines: Sequence[str] = (),
) -> str:
    """The desk table: ``lines`` is the caller's header block (run / contract).

    Columns are right-aligned and sized to their content, so a six-figure notional
    and a 0.0001 delta line up; the rung where the spot did not move is marked.
    """
    labels = [*_FIXED_COLUMNS, *[GREEK_LABELS.get(name, name) for name in greeks]]
    cells = [
        [
            num(row.spot, 8),
            "{:+.2%}".format(row.bump),
            num(row.npv, 8),
            *[num(row.greeks.get(name), 8) for name in greeks],
        ]
        for row in rows
    ]
    widths = [
        max([len(labels[column])] + [len(line[column]) for line in cells])
        for column in range(len(labels))
    ]
    table = ["  ".join(label.rjust(widths[index]) for index, label in enumerate(labels))]
    for index, line in enumerate(cells):
        text = "  ".join(
            cell.rjust(widths[column]) for column, cell in enumerate(line)
        )
        table.append(text + ("   <- base" if _is_base(rows[index]) else ""))
    return "\n".join([*lines, "", *table])


def _is_base(row: SlideRow) -> bool:
    return abs(float(row.bump)) < 1e-12


__all__ = ["GREEK_LABELS", "format_slide", "slide_csv", "slide_to_dict"]
