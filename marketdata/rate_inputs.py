"""Interest-rate inputs from the desk CSV export.

The export (``surface_pricer/data/interest_rate.csv``) has one row per business
day and one column per instrument::

    DateTime,FR007.IR,FR007S1M.IR,FR007S3M.IR,...,FR007S10Y.IR,...
    2026/9/28,1.42,1.435,1.435,...,1.545,...

* ``FR007.IR``         - the 7-day repo fixing (edslib ``CNY-FR007-1W``)
* ``FR007S<tenor>.IR`` - CNY FR007 IRS par rate for ``<tenor>``

Rates are percentages in the file and decimals here: edslib divides by 100 once
on ingestion (``apps/populate_benchmarks.py``), the same convention is used here.

The parser keeps the **latest** observation of each column (not strictly the
last row: a stale column keeps its own last valid date and says so in
``notes``).  Columns whose whole history is one constant value are treated as
broken exports and dropped (``dropped``), which is how the ``FR007S2M`` /
``FR007S6Y`` fixtures in the current file are handled.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

FIXING_COLUMN = "FR007.IR"
SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "interest_rate.csv"

_RATE_PATTERN = re.compile(r"^FR007S(?P<tenor>\d+[DWMY])\.IR$", re.IGNORECASE)
_MISSING = {"", "nan", "na", "n/a", "-", "--", "none", "null"}
_DATE_FORMATS = ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d", "%Y.%m.%d")


def tenor_days(tenor: str) -> float:
    """Calendar-day length of a tenor, used for ordering and sanity checks."""
    text = str(tenor).strip().upper()
    if len(text) < 2:
        raise ValueError("invalid tenor {!r}".format(tenor))
    unit = text[-1]
    try:
        value = float(text[:-1])
    except ValueError as error:
        raise ValueError("invalid tenor {!r}".format(tenor)) from error
    if unit == "D":
        return value
    if unit == "W":
        return value * 7.0
    if unit == "M":
        return value * 30.0
    if unit == "Y":
        return value * 365.0
    raise ValueError("unsupported tenor {!r}; expected D, W, M or Y".format(tenor))


def _to_float(text: object) -> Optional[float]:
    raw = str(text if text is not None else "").strip()
    if not raw or raw.lower() in _MISSING:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _to_date(text: object) -> Optional[date]:
    raw = str(text if text is not None else "").strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


@dataclass
class RateInputs:
    """FR007 fixing + IRS par quotes for one valuation date (decimals)."""

    valuation_date: date
    fr007: Optional[float] = None
    ir_swap: Dict[str, float] = field(default_factory=dict)
    fr007_date: Optional[date] = None
    quote_dates: Dict[str, date] = field(default_factory=dict)
    dropped: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def tenors(self) -> List[str]:
        """IRS tenors sorted from short to long."""
        return sorted(self.ir_swap, key=tenor_days)

    def rates(self) -> List[float]:
        return [self.ir_swap[tenor] for tenor in self.tenors]

    def to_dict(self) -> Dict[str, object]:
        return {
            "valuation_date": self.valuation_date.isoformat(),
            "fr007": self.fr007,
            "fr007_date": self.fr007_date.isoformat() if self.fr007_date else None,
            "ir_swap": {tenor: self.ir_swap[tenor] for tenor in self.tenors},
            "quote_dates": {
                tenor: value.isoformat()
                for tenor, value in sorted(self.quote_dates.items(), key=lambda item: tenor_days(item[0]))
            },
            "dropped": dict(self.dropped),
            "notes": list(self.notes),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "RateInputs":
        def _date_or_none(text: object) -> Optional[date]:
            return _to_date(text) if text else None

        return cls(
            valuation_date=_to_date(value["valuation_date"]),
            fr007=_to_float(value.get("fr007")) if value.get("fr007") is not None else None,
            ir_swap={str(k): float(v) for k, v in dict(value.get("ir_swap") or {}).items()},
            fr007_date=_date_or_none(value.get("fr007_date")),
            quote_dates={
                str(k): _to_date(v)
                for k, v in dict(value.get("quote_dates") or {}).items()
                if _to_date(v) is not None
            },
            dropped={str(k): str(v) for k, v in dict(value.get("dropped") or {}).items()},
            notes=[str(item) for item in list(value.get("notes") or [])],
            source=str(value.get("source") or ""),
        )

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RateInputs":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_interest_rate_csv(
    path: Union[str, Path] = SAMPLE_CSV,
    *,
    drop_constant_columns: bool = True,
    min_rows_for_constant_check: int = 3,
) -> RateInputs:
    """Parse the desk export and keep the latest observation of each column."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError("interest-rate file {} is empty".format(source))

    reader = _csv_rows(lines)
    header, rows = reader
    if not rows:
        raise ValueError("interest-rate file {} has no data rows".format(source))

    valuation_date = max(row[0] for row in rows)
    result = RateInputs(valuation_date=valuation_date, source=str(source))

    columns = [name for name in header if name != header[0]]

    for column in columns:
        label = _column_label(column)
        samples = [
            (row_date, values[column])
            for row_date, values in rows
            if values.get(column) is not None
        ]
        if not samples:
            result.dropped[label] = "no data"
            continue
        unique = {round(value, 10) for _, value in samples}
        if (
            drop_constant_columns
            and len(unique) == 1
            and len(samples) >= min_rows_for_constant_check
        ):
            result.dropped[label] = "constant column ({:.4g})".format(samples[-1][1])
            result.notes.append(
                "{} dropped: every observation equals {:.4g} (99% a broken export)"
                .format(label, samples[-1][1])
            )
            continue
        quote_date, raw_value = samples[-1]
        if quote_date < valuation_date:
            result.notes.append(
                "{} keeps {} (latest row is {})".format(label, quote_date.isoformat(), valuation_date.isoformat())
            )
        if column.upper() == FIXING_COLUMN.upper():
            result.fr007 = raw_value / 100.0
            result.fr007_date = quote_date
            continue
        match = _RATE_PATTERN.match(column.strip())
        if match is None:
            result.dropped[label] = "unrecognised column"
            continue
        tenor = match.group("tenor").upper()
        result.ir_swap[tenor] = raw_value / 100.0
        result.quote_dates[tenor] = quote_date

    if not result.ir_swap:
        raise ValueError(
            "no usable FR007S<tenor>.IR columns in {}".format(source)
        )
    return result


def _csv_rows(lines: List[str]):
    """Return ``(header, rows)`` with ``rows = [(date, {column: float})]``."""
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        raise ValueError("interest-rate file has no header row")
    header = [name.strip() for name in header]
    rows = []
    for raw in reader:
        if not raw or not str(raw[0]).strip():
            continue
        row_date = _to_date(raw[0])
        if row_date is None:
            continue
        values: Dict[str, Optional[float]] = {}
        for index, name in enumerate(header[1:], start=1):
            values[name] = _to_float(raw[index]) if index < len(raw) else None
        rows.append((row_date, values))
    rows.sort(key=lambda item: item[0])
    return header, rows


def _column_label(column: str) -> str:
    match = _RATE_PATTERN.match(str(column).strip())
    if match is not None:
        return match.group("tenor").upper()
    if str(column).strip().upper() == FIXING_COLUMN.upper():
        return "FR007"
    return str(column).strip()


__all__ = [
    "FIXING_COLUMN",
    "RateInputs",
    "SAMPLE_CSV",
    "parse_interest_rate_csv",
    "tenor_days",
]
