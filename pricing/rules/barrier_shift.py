"""Barrier-shift pricing rules for autocallable products.

``shift`` is part of the **pricing rule set** - not a risk pre-process.  The
standard lives in ``surface_pricer/config/barrier_shift.json`` (see its
``description`` field for the format); every contract may override the rule or
the value and the CLI sits on top:

    cli  >  contract override  >  config effective-date block  >  config defaults  >  none

All values are expressed in **barrier units** (``1.0`` = 100% of the initial
spot, the same ratio space the barriers live in, matching edslib):
``relative`` scales the original level by ``1 + value`` while ``additive``
adds ``value`` to it.  ``stepwise`` shifts accrue ``value / periods`` per
observation so the last observation carries the full ``value``, clamped to
``[floor, cap]`` on the *accumulated* shift; ``step_offset`` leaves that many
leading observations unshifted.

The expansion into per-observation **absolute** levels happens through
:func:`expand_shift`, called from two places that must agree: ``build_schedule``
(raw terms -> schedule) and ``AutocallSchedule.from_dict`` (payload -> schedule,
where the barriers are the term-sheet levels and the rule travels in the
payload).  Both hand the engines the same already-shifted absolute terms, so the
MC and PDE engines never see a relative / shift concept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ...core.daycount import DateLike, to_date

#: Environment variable that replaces the packaged standard (same effect as
#: passing ``--shift-config``); an explicit path argument wins over both.
CONFIG_ENV_VAR = "SURFACE_PRICER_SHIFT_CONFIG"

#: Packaged standard, shipped with the code (``surface_pricer/config``).
PACKAGE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "barrier_shift.json"

MODE_NONE = "none"
MODE_ADDITIVE = "additive"
MODE_RELATIVE = "relative"
_MODES = (MODE_NONE, MODE_ADDITIVE, MODE_RELATIVE)

_SIDES = ("ko_shift", "ki_shift")


def _ticker_code(value: Any) -> str:
    """``"000852.SH"`` / ``"000852"`` -> ``"000852"`` (upper case, no venue suffix)."""
    text = str(value or "").strip().upper()
    head, dot, _ = text.partition(".")
    return head if dot else text

#: One year of coupon on a plain snowball, in *calendar* years between the
#: start date and the last observation - edslib's
#: ``unit_gap * max(cumsum(1 / n_periods))`` for a periodic coupon.
_GAP_DAY_BASIS = 365.0


class ShiftConfigError(ValueError):
    """Raised for a missing / malformed barrier-shift configuration."""


@dataclass(frozen=True)
class BarrierShiftSpec:
    """One side (KO or KI) of a resolved barrier shift, in barrier units."""

    mode: str = MODE_NONE
    value: float = 0.0
    stepwise: bool = False
    step_offset: int = 0
    cap: Optional[float] = None
    floor: Optional[float] = None
    source: str = "none"

    def __post_init__(self):
        mode = str(self.mode or MODE_NONE).strip().lower()
        if mode not in _MODES:
            raise ValueError(
                "shift mode must be one of {}, got {!r}".format(", ".join(_MODES), self.mode)
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "step_offset", max(int(self.step_offset), 0))
        if self.cap is not None:
            object.__setattr__(self, "cap", float(self.cap))
        if self.floor is not None:
            object.__setattr__(self, "floor", float(self.floor))
        if mode == MODE_NONE:
            object.__setattr__(self, "value", 0.0)

    @property
    def active(self) -> bool:
        return self.mode != MODE_NONE and self.value != 0.0

    def describe(self) -> str:
        if not self.active:
            return "none"
        return "{} {value:+.4%} ({})".format(self.mode, self.source, value=self.value)

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable rule - what a payload hands over instead of the levels.

        Only the fields the rule actually uses are written, so a plain "no shift"
        is ``{"mode": "none"}`` and a flat −0.5% is two numbers.  ``source`` records
        where the rule came from (config block / contract override / CLI / ``cli``),
        which is what makes a payload self-explanatory.
        """
        payload: Dict[str, Any] = {"mode": self.mode}
        if self.mode != MODE_NONE:
            payload["value"] = float(self.value)
        if self.stepwise:
            payload["stepwise"] = True
        if self.step_offset:
            payload["step_offset"] = int(self.step_offset)
        if self.cap is not None:
            payload["cap"] = float(self.cap)
        if self.floor is not None:
            payload["floor"] = float(self.floor)
        payload["source"] = self.source
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BarrierShiftSpec":
        """Rebuild a spec from :meth:`to_dict` (or a bare ``"none"`` / number)."""
        if payload is None:
            return cls()
        if isinstance(payload, str):
            mode = payload.strip().lower()
            if mode in ("", MODE_NONE):
                return cls()
            if mode not in _MODES:
                raise ValueError(
                    "shift must be an object such as "
                    '{"mode": "relative", "value": -0.005} - got the description '
                    "string {!r} (a payload written before the rule became "
                    "machine-readable; re-export it)".format(payload)
                )
            return cls(mode=mode, source="payload")
        if isinstance(payload, (int, float)):
            return cls(mode=MODE_RELATIVE, value=float(payload), source="payload")
        return cls(
            mode=str(payload.get("mode", MODE_NONE)),
            value=float(payload.get("value", 0.0) or 0.0),
            stepwise=bool(payload.get("stepwise", False)),
            step_offset=int(payload.get("step_offset", 0) or 0),
            cap=_optional_float(payload.get("cap")),
            floor=_optional_float(payload.get("floor")),
            source=str(payload.get("source", "payload")),
        )


@dataclass(frozen=True)
class ShiftConfig:
    """Parsed ``barrier_shift.json``: defaults + effective-date blocks."""

    path: str
    templates: Tuple[str, ...] = ()
    underlying_classes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    effective_dates: Tuple[Tuple[date, Mapping[str, Mapping[str, Any]]], ...] = ()

    def applies_to(self, product_type: str) -> bool:
        """Whether the product type participates in shifting at all."""
        if not self.templates:
            return True
        return str(product_type or "").strip().lower() in set(self.templates)

    def block_for(self, start_date: Optional[DateLike]) -> Tuple[Mapping[str, Any], str]:
        """Rule block for a contract starting on ``start_date`` plus a label.

        Picks the latest effective-date block not later than the start date and
        merges it field-by-field over ``defaults`` (per side); contracts before
        the first block, or without a start date, get ``defaults``.
        """
        if not self.effective_dates:
            return self.defaults, "defaults"
        if start_date is None:
            return self.defaults, "defaults"
        when = to_date(start_date)
        chosen: Optional[Tuple[date, Mapping[str, Mapping[str, Any]]]] = None
        for from_date, block in self.effective_dates:
            if from_date <= when:
                chosen = (from_date, block)
            else:
                break
        if chosen is None:
            return self.defaults, "defaults"
        from_date, block = chosen
        merged: Dict[str, Mapping[str, Any]] = {}
        for side in _SIDES:
            merged[side] = {**dict(self.defaults.get(side, {})), **dict(block.get(side, {}))}
        return merged, "config:{}".format(from_date.isoformat())

    def underlying_class_of(self, underlying: Optional[str]) -> str:
        """``index_like`` when the ticker is in the configured list, else ``other``.

        Compared on the **bare code**: the venue suffix is not part of the class, so
        ``000852.SH`` (the cash index), ``000852`` and an ``MO`` / ETF ticker all
        resolve to whatever list their code appears in.  That is what lets a
        snowball be traded on the cash index while its vol surface is fitted from
        the futures' (or the ETF's) options - the two only have to agree on the
        index they reference.
        """
        key = _ticker_code(underlying)
        for class_name, members in self.underlying_classes.items():
            if key and key in {_ticker_code(member) for member in members}:
                return str(class_name)
        return "other"


# --------------------------------------------------------------------- loading
_CONFIG_CACHE: Dict[str, Tuple[float, ShiftConfig]] = {}


def load_shift_config(path: Optional[str] = None) -> ShiftConfig:
    """Read the packaged standard, or ``path`` / ``$SURFACE_PRICER_SHIFT_CONFIG``.

    Results are cached per absolute path and file mtime; call
    :func:`clear_shift_config_cache` after editing a file in-process.
    """
    import os

    target = path or os.environ.get(CONFIG_ENV_VAR) or str(PACKAGE_CONFIG_PATH)
    source = Path(target)
    if not source.is_file():
        raise ShiftConfigError(
            "barrier-shift config not found: {} (pass --shift-config or fix {})".format(
                source, CONFIG_ENV_VAR
            )
        )
    key = str(source.resolve())
    mtime = source.stat().st_mtime
    cached = _CONFIG_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ShiftConfigError("{} is not valid JSON: {}".format(source, error)) from error
    config = _parse_config(payload, key)
    _CONFIG_CACHE[key] = (mtime, config)
    return config


def clear_shift_config_cache() -> None:
    """Drop the config cache (tests and in-process edits)."""
    _CONFIG_CACHE.clear()


def _parse_config(payload: Any, path: str) -> ShiftConfig:
    if not isinstance(payload, Mapping):
        raise ShiftConfigError("{} must contain a JSON object".format(path))
    templates = payload.get("templates", [])
    if templates is not None and (
        not isinstance(templates, Sequence) or isinstance(templates, (str, bytes))
    ):
        raise ShiftConfigError("{}: 'templates' must be a list".format(path))
    raw_classes = payload.get("underlying_classes") or {}
    if not isinstance(raw_classes, Mapping):
        raise ShiftConfigError("{}: 'underlying_classes' must be an object".format(path))
    underlying_classes = {
        str(name): tuple(str(member) for member in (members or []))
        for name, members in raw_classes.items()
    }

    defaults = _parse_sides(payload.get("defaults"), path, "defaults")

    blocks: List[Tuple[date, Mapping[str, Mapping[str, Any]]]] = []
    seen: set = set()
    for index, block in enumerate(payload.get("effective_dates") or []):
        if not isinstance(block, Mapping) or "from" not in block:
            raise ShiftConfigError(
                "{}: effective_dates[{}] needs a 'from' date".format(path, index)
            )
        from_date = to_date(block["from"])
        if from_date in seen:
            raise ShiftConfigError(
                "{}: duplicate effective date {}".format(path, from_date.isoformat())
            )
        seen.add(from_date)
        blocks.append((from_date, _parse_sides(block, path, "effective_dates[{}]".format(index))))
    blocks.sort(key=lambda item: item[0])

    return ShiftConfig(
        path=path,
        templates=tuple(str(name).strip().lower() for name in (templates or [])),
        underlying_classes=underlying_classes,
        defaults=defaults,
        effective_dates=tuple(blocks),
    )


def _parse_sides(payload: Any, path: str, where: str) -> Mapping[str, Mapping[str, Any]]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ShiftConfigError("{}: {} must be an object".format(path, where))
    parsed: Dict[str, Mapping[str, Any]] = {}
    for side in _SIDES:
        block = payload.get(side)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise ShiftConfigError("{}: {} must be an object".format(path, where))
        parsed[side] = _validate_side(dict(block), path, where)
    return parsed


def _validate_side(block: Dict[str, Any], path: str, where: str) -> Dict[str, Any]:
    rule = str(block.get("rule") or "fixed").strip().lower()
    if rule not in {"fixed", "coupon_fraction", "gap_tiers", "underlying_class"}:
        raise ShiftConfigError(
            "{}: {} rule must be fixed / coupon_fraction / gap_tiers / "
            "underlying_class, got {!r}".format(path, where, rule)
        )
    block["rule"] = rule
    if rule == "underlying_class":
        for class_name in ("index_like", "other"):
            branch = block.get(class_name)
            if branch is None:
                continue
            if not isinstance(branch, Mapping):
                raise ShiftConfigError(
                    "{}: {} {} must be an object".format(path, where, class_name)
                )
            block[class_name] = _validate_side(dict(branch), path, where)
    else:
        mode = str(block.get("mode") or MODE_RELATIVE).strip().lower()
        if mode not in (MODE_ADDITIVE, MODE_RELATIVE):
            raise ShiftConfigError(
                "{}: {} mode must be additive or relative, got {!r}".format(path, where, mode)
            )
        block["mode"] = mode
        if rule == "gap_tiers":
            tiers = block.get("tiers")
            if not isinstance(tiers, Sequence) or isinstance(tiers, (str, bytes)) or not tiers:
                raise ShiftConfigError(
                    "{}: {} gap_tiers needs a non-empty 'tiers' list".format(path, where)
                )
            for tier in tiers:
                if not isinstance(tier, Mapping) or "value" not in tier:
                    raise ShiftConfigError(
                        "{}: {} every gap tier needs a 'value'".format(path, where)
                    )
        elif rule == "coupon_fraction":
            if "fraction" not in block:
                raise ShiftConfigError(
                    "{}: {} coupon_fraction needs 'fraction'".format(path, where)
                )
        else:  # fixed
            if "value" not in block:
                raise ShiftConfigError("{}: {} fixed needs 'value'".format(path, where))
    return block


# ------------------------------------------------------------------- resolving
def base_coupon(contract: Any) -> float:
    """The contract's *base* (first) coupon rate as a plain float.

    A term sheet may spell the coupon as a step-up list (one rate per
    observation, mirroring ``ko_levels``); the shift rule sizes itself on the
    coupon the first observations are quoted at, and duck-typed contracts may
    still hand over a bare float.
    """
    value = getattr(contract, "annual_coupon", 0.0) or 0.0
    if isinstance(value, (int, float)):
        return float(value)
    for item in value:
        return float(item)
    return 0.0  # pragma: no cover - an empty list has no base coupon


def contract_max_gap(contract: Any) -> float:
    """Largest cumulative coupon over the life, in currency (edslib's gap).

    ``|annual_coupon| * notional * years`` with ``years`` the calendar-year
    span from the start date to the last observation (or the expiry).  For a
    plain snowball this is one year of coupon, mirroring edslib's
    ``unit_gap * max(cumsum(1 / n_periods))``.
    """
    coupon = abs(base_coupon(contract))
    notional = abs(float(getattr(contract, "notional", 1.0) or 1.0))
    start = getattr(contract, "start_date", None)
    if start is None:
        return 0.0
    observations = tuple(getattr(contract, "observation_dates", ()) or ())
    end = observations[-1] if observations else getattr(contract, "expiry_date", None)
    if end is None:
        return 0.0
    years = max((to_date(end) - to_date(start)).days, 0) / _GAP_DAY_BASIS
    return coupon * notional * years


def resolve_shift(
    contract: Any,
    config: ShiftConfig,
    *,
    override: Optional[Mapping[str, Any]] = None,
    cli: Optional[Mapping[str, Any]] = None,
    underlying_class: Optional[str] = None,
) -> Tuple[BarrierShiftSpec, BarrierShiftSpec]:
    """Resolve the KO and KI shift specs for ``contract``.

    ``contract`` only needs the attributes ``product_type``, ``start_date``,
    ``annual_coupon`` (a rate or a per-observation list), ``notional``,
    ``observation_dates`` (and ``underlying`` for the class-dependent rule) -
    the autocall contract or anything duck-typed likes this.  ``override`` (contract level, persistable) and
    ``cli`` (this run only) accept either a plain number - which replaces the
    *value* and keeps the rest of the rule - or a mapping with
    ``mode`` / ``value`` / ``stepwise`` / ``step_offset`` / ``cap`` / ``floor``
    (and ``rule`` / ``fraction`` / ``tiers`` / ``index_like`` / ``other`` to
    replace the rule itself).

    Returns ``(ko_spec, ki_spec)``, both inactive when the product type does
    not participate or no rule applies.
    """
    product_type = str(getattr(contract, "product_type", "") or "autocallable")
    if not config.applies_to(product_type):
        return BarrierShiftSpec(), BarrierShiftSpec()

    block, label = config.block_for(getattr(contract, "start_date", None))
    klass = underlying_class or config.underlying_class_of(getattr(contract, "underlying", None))
    ko = _spec_from_block(block.get("ko_shift"), contract, klass, label, where="ko_shift")
    ki = _spec_from_block(block.get("ki_shift"), contract, klass, label, where="ki_shift")

    for level, source in ((override, "contract-override"), (cli, "cli")):
        if not level:
            continue
        ko = _apply_override(ko, level, ("ko", "ko_shift"), source)
        ki = _apply_override(ki, level, ("ki", "ki_shift"), source)
    return ko, ki


def _spec_from_block(
    block: Optional[Mapping[str, Any]],
    contract: Any,
    underlying_class: str,
    label: str,
    *,
    where: str,
) -> BarrierShiftSpec:
    if not block:
        return BarrierShiftSpec()
    rule = str(block.get("rule") or "fixed").strip().lower()
    source = "{}:{}".format(label, where)
    if rule == "underlying_class":
        branch = block.get(str(underlying_class)) or block.get("other")
        if not branch:
            return BarrierShiftSpec()
        return _spec_from_block(branch, contract, underlying_class, label, where=where)

    if rule == "fixed":
        value = float(block.get("value", 0.0))
    elif rule == "coupon_fraction":
        value = float(block.get("fraction", 0.0)) * base_coupon(contract)
    elif rule == "gap_tiers":
        value = _pick_gap_tier(block.get("tiers") or (), contract_max_gap(contract))
    else:  # pragma: no cover - validated at load time
        raise ShiftConfigError("unsupported rule {!r}".format(rule))

    return BarrierShiftSpec(
        mode=str(block.get("mode") or MODE_RELATIVE),
        value=value,
        stepwise=bool(block.get("stepwise", False)),
        step_offset=int(block.get("step_offset", 0) or 0),
        cap=_optional_float(block.get("cap")),
        floor=_optional_float(block.get("floor")),
        source=source,
    )


def _pick_gap_tier(tiers: Iterable[Mapping[str, Any]], gap: float) -> float:
    for tier in tiers:
        max_gap = _optional_float(tier.get("max_gap"))
        if max_gap is None or gap < max_gap:
            return float(tier.get("value", 0.0))
    return 0.0


def _apply_override(
    spec: BarrierShiftSpec,
    level: Mapping[str, Any],
    keys: Tuple[str, ...],
    source: str,
) -> BarrierShiftSpec:
    value = None
    for key in keys:
        if key in level:
            value = level[key]
            break
    if value is None:
        return spec
    if isinstance(value, Mapping):
        merged = {**{f: getattr(spec, f) for f in ("mode", "value", "stepwise", "step_offset", "cap", "floor")}}
        merged.update(dict(value))
        return BarrierShiftSpec(
            mode=str(merged.get("mode") or spec.mode),
            value=float(merged.get("value", spec.value)),
            stepwise=bool(merged.get("stepwise", spec.stepwise)),
            step_offset=int(merged.get("step_offset", spec.step_offset) or 0),
            cap=_optional_float(merged.get("cap")),
            floor=_optional_float(merged.get("floor")),
            source=source,
        )
    return BarrierShiftSpec(
        mode=spec.mode,
        value=float(value),
        stepwise=spec.stepwise,
        step_offset=spec.step_offset,
        cap=spec.cap,
        floor=spec.floor,
        source=source,
    )


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


# ------------------------------------------------------------------ expansion
def expand_shift(
    spec: BarrierShiftSpec,
    levels: Sequence[float],
    *,
    start_index: int = 0,
    elapsed: int = 0,
) -> Tuple[float, ...]:
    """Expand a spec into per-observation levels (same units as ``levels``).

    ``stepwise`` specs accrue ``value / periods`` per observation (``periods``
    counted after ``step_offset``) so the last one carries the full ``value``;
    flat specs apply the same ``value`` to every observation from
    ``step_offset`` on.  The applied shift is clamped to ``[floor, cap]``
    (on the accumulated amount for stepwise specs).  ``relative`` scales the
    original level, ``additive`` offsets it.  Observations before
    ``start_index`` (already fixed in the past) keep their original level while
    the accrual schedule itself stays anchored to the contract start, exactly
    like edslib's ``on_or_before(ds_date).merge(shifted.after(ds_date))``.

    ``elapsed`` says how many observations of the accrual schedule lie *before*
    ``levels`` - a payload only carries the future ones, so a stepwise rule has to
    know where in the accrual it resumes: the divisor stays the contractual period
    count and the already-accrued part is seeded, which is what makes a mid-life
    payload reproduce the same levels as the full timeline.
    """
    out = [float(level) for level in levels]
    count = len(out)
    if not spec.active or count == 0:
        return tuple(out)

    elapsed = max(int(elapsed), 0)
    offset = min(int(spec.step_offset), elapsed + count)
    periods = max(elapsed + count - offset, 1)
    increment = spec.value / periods if spec.stepwise else spec.value

    # the periods that accrued before ``elapsed`` (stepwise only; a flat shift
    # applies the same value everywhere).
    accrued_before = max(elapsed - offset, 0)
    accumulated = increment * accrued_before if spec.stepwise else 0.0

    for index in range(count):
        position = elapsed + index
        if position >= offset:
            accumulated = accumulated + increment if spec.stepwise else increment
        shift = accumulated
        if spec.cap is not None:
            shift = min(shift, spec.cap)
        if spec.floor is not None:
            shift = max(shift, spec.floor)
        if index < start_index:
            continue  # past observations keep the original level
        if spec.mode == MODE_RELATIVE:
            out[index] = out[index] * (1.0 + shift)
        else:
            out[index] = out[index] + shift
    return tuple(out)


__all__ = [
    "BarrierShiftSpec",
    "CONFIG_ENV_VAR",
    "MODE_ADDITIVE",
    "MODE_NONE",
    "MODE_RELATIVE",
    "PACKAGE_CONFIG_PATH",
    "ShiftConfig",
    "ShiftConfigError",
    "clear_shift_config_cache",
    "contract_max_gap",
    "expand_shift",
    "load_shift_config",
    "resolve_shift",
]
