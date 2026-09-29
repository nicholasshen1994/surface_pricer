"""CNY FR007 interest-rate curve: QuantLib bootstrap in the edslib convention.

Mirrors edslib's ``CNY-FR007`` curve (``market_env/ir_curve_config.py`` and
``market_env/ql_helpers.py``):

* instruments: the FR007 1W repo fixing (index first fixing) plus the FR007
  IRS pillar set from ``utils/ir_marking_instruments.json``
  (``1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 6Y, 7Y, 10Y``);
* swap convention: quarterly fixed leg, ACT/365F, BDC following, calendar
  ``IB`` (= ``ql.China(ql.China.IB)``), floating index ``CNY-FR007-3M``
  (QL ``IborIndex``, settlement 1, ACT/365F), pillar ``LastRelevantDate``,
  zero spread;
* curve: ``ql.PiecewiseLinearZero`` (linear on zero rates) over ACT/360,
  extrapolation enabled.

QuantLib is imported lazily and only here: it is needed to *build* the curve.
The pricing layer consumes the exported pillars through
:class:`surface_pricer.core.curves.PiecewiseRateCurve` and never sees QL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from .curves import PiecewiseRateCurve
from .daycount import DateLike, to_date

CURVE_NAME = "CNY-FR007"
FIXING_INDEX_NAME = "CNY-FR007-1W"
FIXING_INDEX_TENOR = "1W"
FIXING_SETTLEMENT_DAYS = 0
SWAP_INDEX_NAME = "CNY-FR007-3M"
SWAP_INDEX_TENOR = "3M"
SWAP_INDEX_SETTLEMENT_DAYS = 1
CURVE_DAY_COUNTER = "act/360"
FIXED_DAY_COUNTER = "act/365f"
FIXED_FREQUENCY = "quarterly"
FIXED_BDC = "following"
CALENDAR_NAME = "IB"


def _quantlib():
    try:
        import QuantLib as ql
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Building the CNY FR007 curve requires QuantLib (curve generation "
            "only - the pricer reads the exported pillars). Install it with "
            "'pip install QuantLib' in the environment used for curve build."
        ) from error
    return ql


@dataclass
class IRCurvePillars:
    """Bootstrapped zero curve, stored as pillar dates + zero rates."""

    valuation_date: date
    tenors: List[str]
    pillar_dates: List[date]
    pillar_days: List[int]
    zero_rates: List[float]
    day_counter: str = CURVE_DAY_COUNTER
    curve_name: str = CURVE_NAME
    fr007: Optional[float] = None
    par_rates: Dict[str, float] = field(default_factory=dict)
    par_check: Dict[str, float] = field(default_factory=dict)
    source: str = ""
    notes: List[str] = field(default_factory=list)

    def max_par_residual(self) -> float:
        """Largest |rebuilt - quoted| par rate, in rate units (e.g. 1e-6)."""
        gaps = [
            abs(self.par_check[tenor] - self.par_rates[tenor])
            for tenor in self.par_rates
            if tenor in self.par_check
        ]
        return max(gaps) if gaps else 0.0

    def to_piecewise_curve(self) -> PiecewiseRateCurve:
        """Runtime representation (linear on zero rates, same time scale)."""
        return PiecewiseRateCurve(
            anchor=self.valuation_date,
            tenors=[float(days) for days in self.pillar_days],
            rates=self.zero_rates,
            interpolation="linear_zero",
            basis=self.day_counter,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "curve_name": self.curve_name,
            "valuation_date": self.valuation_date.isoformat(),
            "day_counter": self.day_counter,
            "fr007": self.fr007,
            "tenors": list(self.tenors),
            "pillar_dates": [value.isoformat() for value in self.pillar_dates],
            "pillar_days": list(self.pillar_days),
            "zero_rates": list(self.zero_rates),
            "par_rates": dict(self.par_rates),
            "par_check": dict(self.par_check),
            "max_par_residual": self.max_par_residual(),
            "source": self.source,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "IRCurvePillars":
        return cls(
            valuation_date=to_date(str(value["valuation_date"])),
            tenors=[str(item) for item in list(value["tenors"])],
            pillar_dates=[to_date(str(item)) for item in list(value["pillar_dates"])],
            pillar_days=[int(item) for item in list(value["pillar_days"])],
            zero_rates=[float(item) for item in list(value["zero_rates"])],
            day_counter=str(value.get("day_counter", CURVE_DAY_COUNTER)),
            curve_name=str(value.get("curve_name", CURVE_NAME)),
            fr007=float(value["fr007"]) if value.get("fr007") is not None else None,
            par_rates={str(k): float(v) for k, v in dict(value.get("par_rates") or {}).items()},
            par_check={str(k): float(v) for k, v in dict(value.get("par_check") or {}).items()},
            source=str(value.get("source", "")),
            notes=[str(item) for item in list(value.get("notes") or [])],
        )

    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "IRCurvePillars":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_fr007_curve(
    valuation_date: DateLike,
    ir_swap: Mapping[str, float],
    *,
    fr007: Optional[float] = None,
    source: str = "",
) -> IRCurvePillars:
    """Bootstrap the CNY FR007 curve from IRS par rates (decimals).

    ``ir_swap`` maps tenor (``"1M"`` .. ``"10Y"``) to the par rate; ``fr007`` is
    the 1W repo fixing recorded on the ``CNY-FR007-1W`` index, mirroring
    edslib's ``index_first_fixing`` handling.
    """
    if not ir_swap:
        raise ValueError("ir_swap must contain at least one tenor")

    ql = _quantlib()
    val_date = to_date(valuation_date)
    ql_date = ql.Date(val_date.day, val_date.month, val_date.year)
    ql.Settings.instance().evaluationDate = ql_date

    calendar = ql.China(ql.China.IB)
    curve_dcc = ql.Actual360()
    fixed_dcc = ql.Actual365Fixed(ql.Actual365Fixed.Standard)

    notes: List[str] = []
    if fr007 is not None:
        fixing_index = ql.IborIndex(
            FIXING_INDEX_NAME,
            ql.Period(FIXING_INDEX_TENOR),
            FIXING_SETTLEMENT_DAYS,
            ql.CNYCurrency(),
            calendar,
            ql.Following,
            False,
            fixed_dcc,
        )
        fixing_index.addFixing(ql_date, float(fr007), True)

    swap_index = ql.IborIndex(
        SWAP_INDEX_NAME,
        ql.Period(SWAP_INDEX_TENOR),
        SWAP_INDEX_SETTLEMENT_DAYS,
        ql.CNYCurrency(),
        calendar,
        ql.Following,
        False,
        fixed_dcc,
    )

    ordered = sorted(ir_swap.items(), key=lambda item: _tenor_sort_key(item[0]))
    helpers = []
    for tenor, rate in ordered:
        helper = ql.SwapRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(float(rate))),
            ql.Period(str(tenor)),
            calendar,
            ql.Quarterly,
            ql.Following,
            fixed_dcc,
            swap_index,
            ql.QuoteHandle(ql.SimpleQuote(0.0)),
            ql.Period(0, ql.Days),
            ql.YieldTermStructureHandle(),
            swap_index.fixingDays(),
            ql.Pillar.LastRelevantDate,
            ql.Date(),
            False,
        )
        helpers.append((str(tenor), helper))

    curve = ql.PiecewiseLinearZero(ql_date, [item[1] for item in helpers], curve_dcc)
    curve.enableExtrapolation()

    tenors: List[str] = []
    pillar_dates: List[date] = []
    pillar_days: List[int] = []
    zero_rates: List[float] = []
    par_rates: Dict[str, float] = {}
    par_check: Dict[str, float] = {}
    for tenor, helper in helpers:
        pillar = helper.pillarDate()
        pillar_py = date(pillar.year(), pillar.month(), pillar.dayOfMonth())
        tenors.append(tenor)
        pillar_dates.append(pillar_py)
        pillar_days.append((pillar_py - val_date).days)
        zero_rates.append(float(curve.zeroRate(pillar, curve_dcc, ql.Continuous).rate()))
        par_rates[tenor] = float(ir_swap[tenor])
        par_check[tenor] = float(helper.impliedQuote())

    pillars = IRCurvePillars(
        valuation_date=val_date,
        tenors=tenors,
        pillar_dates=pillar_dates,
        pillar_days=pillar_days,
        zero_rates=zero_rates,
        fr007=None if fr007 is None else float(fr007),
        par_rates=par_rates,
        par_check=par_check,
        source=source,
        notes=notes,
    )
    residual = pillars.max_par_residual()
    if residual > 1.0e-6:
        pillars.notes.append(
            "par-rate rebuild residual {:.2e} - check the pillar set".format(residual)
        )
    return pillars


def _tenor_sort_key(tenor: str) -> float:
    text = str(tenor).strip().upper()
    try:
        value = float(text[:-1])
    except ValueError as error:
        raise ValueError("invalid tenor {!r}".format(tenor)) from error
    unit = text[-1]
    factor = {"D": 1.0, "W": 7.0, "M": 30.0, "Y": 365.0}.get(unit)
    if factor is None:
        raise ValueError("invalid tenor {!r}".format(tenor))
    return value * factor


__all__ = [
    "CALENDAR_NAME",
    "CURVE_NAME",
    "IRCurvePillars",
    "build_fr007_curve",
]
