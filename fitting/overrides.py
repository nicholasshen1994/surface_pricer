"""Manual pillar overrides and synthetic tenor extension for the standalone fit.

This module carries the "hand adjust the surface, then refit" workflow that
edslib implements through the ``ML.EDS.RISK.MID`` hand-marked context (see
:mod:`apps.vol_fitting.marked_surface_vanilla_generator` and
:mod:`apps.vol_fitting.fit_vol_regulator`).

edslib injects hand values *softly*: the marked surface is turned into synthetic
vanilla quotes with a bid/ask width of ``spread=0.05`` and fed back into the
standard calibration.  The standalone pricer takes the opposite, *hard* route:
pinned values are enforced exactly (the corresponding optimizer dimensions are
removed) and the longer-dated pillars are filled by interpolation of the fitted
surface.  Only edslib's synthetic-tenor rule is reproduced verbatim, so both
tools extend the curve over exactly the same dates.

All smile/ATM values handled here are **surface-layer storage values**, i.e. the
numbers reported by :meth:`EDSSabrSurface.to_dict` (already multiplied by
``max(0.3, sqrt(tau))``).  The conversion to the slice layer happens in
:mod:`surface_pricer.fitting.engine`.

The module deliberately does not import :mod:`surface_pricer.fitting.engine` so the
dependency graph stays acyclic (``fit_settings`` -> ``overrides`` is a
type-checking-only import).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.daycount import BusinessCalendar, DateLike, add_tenor, to_date, to_datetime
from ..marketdata.listed_contracts import cffex_option_expiry_date
from .settings import SMILE_PARAMETER_NAMES
from .surface import EDSSabrSurface, SMILE_NAME_TO_FIELD

# Surface array names, in the same order as ``SMILE_PARAMETER_NAMES``.
SURFACE_FIELD_NAMES: Tuple[str, ...] = (
    "skews",
    "convs",
    "left_skews_1",
    "left_skews_2",
    "right_skews_1",
    "right_skews_2",
)

# ``{"skew": "skews", "conv": "convs", ...}`` - single name -> surface array.
# Owned by :mod:`surface_pricer.fitting.surface` so the two mappings cannot drift apart.
SMILE_FIELD_TO_SURFACE: Dict[str, str] = dict(SMILE_NAME_TO_FIELD)

SMILE_ATTR_NAMES: Tuple[str, ...] = tuple(SMILE_PARAMETER_NAMES)

# Tenor parsing itself lives in ``core.daycount`` (``add_tenor``) so the
# pricing layer can reuse it without importing this module.
_TENOR_PATTERN = re.compile(r"^\s*(\d+)\s*([DWMY])\s*$", re.IGNORECASE)


def _looks_like_tenor(spec: str) -> bool:
    return _TENOR_PATTERN.match(str(spec)) is not None


def _next_business_day(value: date, calendar: Optional[BusinessCalendar]) -> date:
    if calendar is None:
        return value
    shifted = value
    while not calendar.is_business_day(shifted):
        shifted += timedelta(days=1)
    return shifted


def resolve_expiry(
    spec: str,
    valuation_date: DateLike,
    calendar: Optional[BusinessCalendar] = None,
) -> datetime:
    """Resolve ``"2027-06-18"`` or ``"18M"`` into a business-day datetime."""
    text = str(spec).strip()
    if not text:
        raise ValueError("expiry specification must not be empty")
    if _looks_like_tenor(text):
        target = add_tenor(valuation_date, text)
    else:
        try:
            target = to_date(text)
        except ValueError as error:
            raise ValueError(
                "cannot parse expiry {!r}; use an ISO date (2027-06-18) or a tenor (18M)".format(spec)
            ) from error
    return to_datetime(_next_business_day(target, calendar))


# ------------------------------------------------------------ synthetic pillars
def _nth_weekday_of_month(
    year: int,
    month: int,
    weekday: int,
    week: int,
    calendar: Optional[BusinessCalendar] = None,
) -> Optional[date]:
    """``week``-th ``weekday`` of a month, rolled forward to a business day.

    ``weekday`` follows :meth:`datetime.date.weekday` (0=Mon ... 4=Fri).  The
    default CFFEX combination (third Friday) reuses
    :func:`surface_pricer.index_option_contracts.cffex_option_expiry_date` so
    both call sites cannot drift apart.
    """
    if weekday == 4 and week == 3:
        return to_date(cffex_option_expiry_date(year, month, calendar=calendar))
    first = date(int(year), int(month), 1)
    target = first + timedelta(days=(int(weekday) - first.weekday()) % 7 + 7 * (int(week) - 1))
    if target.month != int(month):
        return None
    return _next_business_day(target, calendar)


def synthetic_tenor_dates(
    valuation_date: DateLike,
    max_pillar_date: DateLike,
    calendar: Optional[BusinessCalendar] = None,
    config: Optional["OverrideConfig"] = None,
) -> List[date]:
    """Reproduce edslib's synthetic-tenor rule.

    Ported from ``SyntheticConfig`` + ``UnifiedVolBorrowRegulator``
    (``apps/vol_fitting/fit_vol_regulator.py``): generate the third Friday of
    every June and December, roll it forward to a business day, keep going until
    ``valuation_date + synthetic_end_tenor`` (``"3Y"`` by default) and only keep
    dates strictly beyond the current last pillar.
    """
    config = config or OverrideConfig()
    horizon = add_tenor(valuation_date, config.synthetic_end_tenor)
    last_year = horizon.year
    first_year = to_date(valuation_date).year
    limit = to_date(max_pillar_date)
    dates: List[date] = []
    for year in range(first_year, last_year + 1):
        for month in config.synthetic_months:
            candidate = _nth_weekday_of_month(
                year, month, config.synthetic_weekday, config.synthetic_week, calendar
            )
            if candidate is None or candidate <= limit:
                continue
            dates.append(candidate)
    return sorted(set(dates))


# ------------------------------------------------------------------- overrides
@dataclass
class PillarOverride:
    """Hand values for one expiry, expressed as surface-layer storage values.

    ``None`` means "do not touch this field": the remaining fields of the same
    expiry are still optimised.  An override that pins every smile field makes
    the whole expiry fixed.
    """

    expiry: str
    atm_vol: Optional[float] = None
    skew: Optional[float] = None
    conv: Optional[float] = None
    left_skew_1: Optional[float] = None
    left_skew_2: Optional[float] = None
    right_skew_1: Optional[float] = None
    right_skew_2: Optional[float] = None

    def smile_values(self) -> Dict[str, float]:
        """Pinned smile fields only, keyed by the single-form parameter names."""
        values: Dict[str, float] = {}
        for name in SMILE_ATTR_NAMES:
            value = getattr(self, name)
            if value is not None:
                values[name] = float(value)
        return values

    @property
    def is_empty(self) -> bool:
        return self.atm_vol is None and not self.smile_values()

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"expiry": self.expiry}
        if self.atm_vol is not None:
            payload["atm_vol"] = float(self.atm_vol)
        payload.update(self.smile_values())
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "PillarOverride":
        if not isinstance(payload, Mapping):
            raise ValueError(
                "each override must be a JSON object, got {!r}".format(type(payload).__name__)
            )
        allowed = {item.name for item in dataclass_fields(PillarOverride)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown override field(s): {}".format(", ".join(unknown)))
        if "expiry" not in payload:
            raise ValueError("override requires an 'expiry' field")
        expiry = str(payload["expiry"]).strip()
        values: Dict[str, Any] = {"expiry": expiry}
        for name in ("atm_vol",) + SMILE_ATTR_NAMES:
            if name in payload and payload[name] is not None:
                try:
                    values[name] = float(payload[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "override '{}' of '{}' must be numeric".format(name, expiry)
                    ) from error
        return PillarOverride(**values)


@dataclass
class OverrideConfig:
    """Hand overrides plus the synthetic tenor extension switches."""

    overrides: Tuple[PillarOverride, ...] = ()
    extend_synthetic_tenors: bool = False
    synthetic_months: Tuple[int, ...] = (6, 12)
    synthetic_week: int = 3
    synthetic_weekday: int = 4
    synthetic_end_tenor: str = "3Y"
    scaling_floor: Optional[float] = None

    def __post_init__(self):
        self.overrides = tuple(
            item if isinstance(item, PillarOverride) else PillarOverride.from_dict(item)
            for item in self.overrides
        )
        months: List[int] = []
        for month in self.synthetic_months:
            value = int(month)
            if not 1 <= value <= 12:
                raise ValueError("synthetic_months must be within 1..12, got {}".format(value))
            if value not in months:
                months.append(value)
        self.synthetic_months = tuple(sorted(months))
        if not self.synthetic_months:
            raise ValueError("synthetic_months must not be empty")
        self.synthetic_week = int(self.synthetic_week)
        if self.synthetic_week < 1:
            raise ValueError("synthetic_week must be >= 1, got {}".format(self.synthetic_week))
        self.synthetic_weekday = int(self.synthetic_weekday)
        if not 0 <= self.synthetic_weekday <= 6:
            raise ValueError(
                "synthetic_weekday must be within 0..6 (0=Monday), got {}".format(
                    self.synthetic_weekday
                )
            )
        self.synthetic_end_tenor = str(self.synthetic_end_tenor).strip()
        # raises on a malformed tenor, so a bad config fails fast
        add_tenor(date(2000, 1, 1), self.synthetic_end_tenor)
        if self.scaling_floor is not None:
            self.scaling_floor = float(self.scaling_floor)
            if self.scaling_floor <= 0.0:
                raise ValueError("scaling_floor must be positive")
        for item in self.overrides:
            if item.is_empty:
                raise ValueError(
                    "override for {!r} pins no value; remove it or set atm_vol / smile fields".format(
                        item.expiry
                    )
                )
        seen: Dict[str, str] = {}
        for item in self.overrides:
            key = str(item.expiry).strip()
            if key in seen:
                raise ValueError("duplicate override for {!r}".format(key))
            seen[key] = key

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "overrides": [item.to_dict() for item in self.overrides],
            "extend_synthetic_tenors": bool(self.extend_synthetic_tenors),
            "synthetic_months": list(self.synthetic_months),
            "synthetic_week": int(self.synthetic_week),
            "synthetic_weekday": int(self.synthetic_weekday),
            "synthetic_end_tenor": self.synthetic_end_tenor,
        }
        if self.scaling_floor is not None:
            payload["scaling_floor"] = float(self.scaling_floor)
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "OverrideConfig":
        if not isinstance(payload, Mapping):
            raise ValueError(
                "override config must be a JSON object, got {!r}".format(type(payload).__name__)
            )
        allowed = {item.name for item in dataclass_fields(OverrideConfig)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown override config field(s): {}".format(", ".join(unknown)))
        values: Dict[str, Any] = {}
        if "overrides" in payload and payload["overrides"] is not None:
            raw = payload["overrides"]
            if not isinstance(raw, (list, tuple)):
                raise ValueError("'overrides' must be a list of objects")
            values["overrides"] = tuple(PillarOverride.from_dict(item) for item in raw)
        for name in ("extend_synthetic_tenors",):
            if name in payload and payload[name] is not None:
                values[name] = bool(payload[name])
        for name in ("synthetic_months",):
            if name in payload and payload[name] is not None:
                values[name] = tuple(int(item) for item in payload[name])
        for name in ("synthetic_week", "synthetic_weekday"):
            if name in payload and payload[name] is not None:
                values[name] = int(payload[name])
        for name in ("synthetic_end_tenor",):
            if name in payload and payload[name] is not None:
                values[name] = str(payload[name])
        if "scaling_floor" in payload and payload["scaling_floor"] is not None:
            values["scaling_floor"] = float(payload["scaling_floor"])
        return OverrideConfig(**values)

    @staticmethod
    def from_json(path: str) -> "OverrideConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return OverrideConfig.from_dict(payload)

    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=indent, sort_keys=False)
            handle.write("\n")

    # ---------------------------------------------------------------- resolving
    def resolve(
        self,
        valuation_date: DateLike,
        calendar: Optional[BusinessCalendar] = None,
        pillar_dates: Iterable[DateLike] = (),
    ) -> "Resolution":
        """Resolve every override to a date and add the synthetic pillars."""
        existing = _sorted_unique(to_datetime(value) for value in pillar_dates)
        existing_dates = {to_date(value) for value in existing}
        existing_by_date = {to_date(value): value for value in existing}

        resolved: List[ResolvedOverride] = []
        seen: Dict[date, str] = {}
        for item in self.overrides:
            expiry = resolve_expiry(item.expiry, valuation_date, calendar)
            key = to_date(expiry)
            if key in seen:
                raise ValueError(
                    "duplicate override for {} ({!r} and {!r})".format(key, seen[key], item.expiry)
                )
            seen[key] = item.expiry
            matched = key in existing_dates
            if matched:
                # keep the listed pillar timestamp (e.g. 15:00 expiry)
                expiry = existing_by_date[key]
            resolved.append(
                ResolvedOverride(
                    expiry=expiry,
                    raw_expiry=item.expiry,
                    atm_vol=item.atm_vol,
                    smile=item.smile_values(),
                    matched=matched,
                )
            )

        synthetic: List[datetime] = []
        if self.extend_synthetic_tenors:
            fixed_dates = [to_date(item.expiry) for item in resolved]
            anchor = max(existing_dates.union(fixed_dates)) if (existing_dates or fixed_dates) else to_date(valuation_date)
            for candidate in synthetic_tenor_dates(valuation_date, anchor, calendar, self):
                if candidate in existing_dates or candidate in seen:
                    continue
                synthetic.append(to_datetime(candidate))

        pillars = _sorted_unique(
            list(existing)
            + [item.expiry for item in resolved]
            + list(synthetic)
        )
        return Resolution(
            overrides=tuple(resolved),
            synthetic_dates=tuple(synthetic),
            pillar_dates=tuple(pillars),
            existing_pillar_dates=tuple(existing),
        )


@dataclass
class ResolvedOverride:
    """One override after tenor resolution, still in surface-layer values."""

    expiry: datetime
    raw_expiry: str
    atm_vol: Optional[float]
    smile: Dict[str, float]
    matched: bool
    synthetic: bool = False

    @property
    def fields(self) -> Tuple[str, ...]:
        names: Tuple[str, ...] = tuple(self.smile)
        if self.atm_vol is not None:
            return ("atm_vol",) + names
        return names


@dataclass
class Resolution:
    """Outcome of :meth:`OverrideConfig.resolve`."""

    overrides: Tuple[ResolvedOverride, ...] = ()
    synthetic_dates: Tuple[datetime, ...] = ()
    pillar_dates: Tuple[datetime, ...] = ()
    existing_pillar_dates: Tuple[datetime, ...] = ()

    @property
    def pinned_dates(self) -> Tuple[datetime, ...]:
        return tuple(item.expiry for item in self.overrides if item.matched)

    @property
    def extra_pillar_dates(self) -> Tuple[datetime, ...]:
        return tuple(item.expiry for item in self.overrides if not item.matched)

    def new_pillars(self) -> Tuple[datetime, ...]:
        """Pillars that do not exist in the fitted surface yet."""
        known = {to_date(value) for value in self.existing_pillar_dates}
        return tuple(value for value in self.pillar_dates if to_date(value) not in known)

    @property
    def has_work(self) -> bool:
        return bool(self.overrides or self.synthetic_dates)


# ----------------------------------------------------------- surface extension
def _sorted_unique(values: Iterable[datetime]) -> List[datetime]:
    unique: Dict[date, datetime] = {}
    for value in values:
        unique.setdefault(to_date(value), to_datetime(value))
    return [unique[key] for key in sorted(unique)]


def extend_and_apply(
    surface: EDSSabrSurface,
    resolution: Resolution,
) -> Tuple[EDSSabrSurface, List[ResolvedOverride], Tuple[datetime, ...]]:
    """Extend a fitted surface to the requested pillars and pin hand values.

    Returns ``(extended_surface, applied_overrides, new_pillars)`` where
    ``new_pillars`` are the expiries added on top of the fitted ones.  Existing
    pillars keep their fitted values because :meth:`EDSSabrSurface.rebuild`
    interpolates exactly on its own pivot dates.
    """
    existing = [to_datetime(value) for value in np.asarray(surface.expiry_dates)]
    known = {to_date(value) for value in existing}
    targets = list(existing)
    new_pillars: List[datetime] = []
    for value in resolution.pillar_dates:
        if to_date(value) in known:
            continue
        known.add(to_date(value))
        targets.append(to_datetime(value))
        new_pillars.append(to_datetime(value))
    targets = _sorted_unique(targets)

    if not resolution.overrides and not new_pillars:
        return surface, [], ()

    extended = surface.rebuild(targets)

    applied: List[ResolvedOverride] = []
    for item in resolution.overrides:
        if extended.pillar_index(item.expiry) is None:
            continue
        extended.set_pillar(item.expiry, atm_vol=item.atm_vol, smile=item.smile)
        applied.append(item)

    return extended, applied, tuple(new_pillars)


# ------------------------------------------------------------- layer conversion
def parameter_scale(tau: float, scaling_floor: float) -> float:
    """``max(scaling_floor, sqrt(tau))``, the surface <-> slice-layer scaling."""
    return max(float(scaling_floor), float(np.sqrt(max(float(tau), 1.0e-12))))


def surface_to_slice_values(
    smile_values: Mapping[str, float],
    tau: float,
    scaling_floor: float,
) -> Dict[str, float]:
    """Convert pinned surface-layer smile values into slice-layer values."""
    scale = parameter_scale(tau, scaling_floor)
    return {name: float(value) / scale for name, value in smile_values.items()}


def validate_slice_values(
    smile_values: Mapping[str, float],
    bounds: Sequence[Sequence[float]],
    expiry: Optional[DateLike] = None,
) -> None:
    """Raise a helpful :class:`ValueError` when a pinned value breaks bounds."""
    label = "" if expiry is None else " for expiry {}".format(to_date(expiry))
    for index, name in enumerate(SMILE_ATTR_NAMES):
        if name not in smile_values:
            continue
        lower, upper = float(bounds[index][0]), float(bounds[index][1])
        value = float(smile_values[name])
        if value < lower or value > upper:
            raise ValueError(
                "pinned {}={:.6g}{} is outside the allowed range [{:.6g}, {:.6g}] "
                "(slice-layer value; surface value = slice value * max(floor, sqrt(tau)))".format(
                    name, value, label, lower, upper
                )
            )


# ------------------------------------------------------------------ CLI helper
def parse_pin_spec(text: str) -> PillarOverride:
    """Parse ``"2027-06-18:atm_vol=0.215,skew=0.04"`` into a :class:`PillarOverride`."""
    expiry, separator, body = str(text).partition(":")
    if not separator:
        raise ValueError(
            "pin specification {!r} must be '<expiry>:<field>=<value>[,<field>=<value>...]'".format(text)
        )
    values: Dict[str, Any] = {"expiry": expiry.strip()}
    allowed = set(SMILE_ATTR_NAMES) | {"atm_vol"}
    for chunk in body.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, equals, raw = chunk.partition("=")
        if not equals:
            raise ValueError("pin entry {!r} must look like '<field>=<value>'".format(chunk))
        name = name.strip().lower()
        if name not in allowed:
            raise ValueError(
                "unknown pin field {!r}; expected one of {}".format(
                    name, ", ".join(["atm_vol"] + list(SMILE_ATTR_NAMES))
                )
            )
        try:
            values[name] = float(raw)
        except ValueError as error:
            raise ValueError("pin entry {!r} is not numeric".format(chunk)) from error
    return PillarOverride.from_dict(values)


__all__ = [
    "OverrideConfig",
    "PillarOverride",
    "Resolution",
    "ResolvedOverride",
    "SMILE_ATTR_NAMES",
    "SMILE_FIELD_TO_SURFACE",
    "SURFACE_FIELD_NAMES",
    "add_tenor",
    "extend_and_apply",
    "parameter_scale",
    "parse_pin_spec",
    "resolve_expiry",
    "surface_to_slice_values",
    "synthetic_tenor_dates",
    "validate_slice_values",
]
