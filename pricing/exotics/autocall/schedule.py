"""Effective (post-shift) terms: the single place a barrier shift is applied."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ....core.daycount import resolve_basis, to_date, to_datetime, year_fraction
from ....core.market import MarketState
from ...rules.barrier_shift import (
    BarrierShiftSpec,
    ShiftConfig,
    expand_shift,
    load_shift_config,
    resolve_shift,
)
from .cashflows import accrual_between, ko_cash_flow, rebate_ratio
from .contract import (
    AutocallContract,
    monitoring_grid,
    resolve_boundary,
    resolve_ki_frequency,
    triggers,
)

#: The one payload monitoring rule that carries its own grid: with
#: ``knock_in.frequency = "custom"`` the dates and levels are written out, while
#: every other rule is rebuilt from the calendar and the observation dates.
CUSTOM_FREQUENCY = "custom"

def resolve_monitoring_frequency(value: Any, *, explicit: bool = False) -> str:
    """Normalise a payload's ``knock_in.frequency`` - four names, one spelling each.

    ``custom`` is the one value that carries its own grid (``knock_in.dates`` /
    ``levels``); the other three are rules the reader rebuilds from the calendar
    and the observation dates.  ``explicit`` says the grid *is* in the payload,
    which only ``custom`` allows: a payload that lists dates under a rule name is
    refused (it used to be quietly read as ``custom``), and the old synonyms
    (``explicit`` / ``manual`` / ``list``) are gone.
    """
    text = str(value or "daily").strip().lower()
    if text == CUSTOM_FREQUENCY:
        return CUSTOM_FREQUENCY
    resolved = resolve_ki_frequency(text)
    if explicit:
        raise ValueError(
            "knock_in.frequency {!r} carries its own dates/levels: spell it "
            "'custom', or drop the grid".format(value)
        )
    return resolved


#: The two ways a same-day fixing may be read.  ``contractual`` tests the raw
#: term-sheet level (the legal determination, what the ledger replay uses);
#: ``effective`` tests the post-shift level the engines price against - the
#: intraday reading, where the shift says the desk has not triggered yet.
TRIGGER_CONTRACTUAL = "contractual"
TRIGGER_EFFECTIVE = "effective"
TRIGGER_BASES = (TRIGGER_CONTRACTUAL, TRIGGER_EFFECTIVE)


def resolve_trigger_basis(value: Any = None) -> str:
    """Normalise the ``trigger_basis`` valuation input (``contractual`` default).

    A *valuation* input, not a term: nothing about the contract changes, only the
    barrier today's fixing is tested against.  Intraday runs pass ``"effective"``
    (the greeks stay on the shifted line the engines price), EOD runs and every
    ledger write-back pass ``"contractual"``.  Anything else is refused rather
    than guessed.
    """
    text = str(value or TRIGGER_CONTRACTUAL).strip().lower().replace("-", "_")
    if text not in TRIGGER_BASES:
        raise ValueError(
            "unknown trigger basis {!r}: use 'contractual' (EOD, raw terms) or "
            "'effective' (intraday, post-shift)".format(value)
        )
    return text


def period_level(
    levels: Sequence[float], observations: Sequence[datetime], when: datetime
) -> float:
    """Barrier of the observation period ``when`` falls into (the last one if past).

    A stepwise shift changes the barrier from one observation period to the next, so
    every monitoring date has to be read through the period it belongs to.
    """
    for index, day in enumerate(observations):
        if day >= when:
            return float(levels[index])
    return float(levels[-1]) if levels else 0.0


@dataclass(frozen=True)
class AutocallSchedule:
    """Effective (post-shift) terms - the only input the engines see.

    All levels are **absolute prices**, never ratios; ``*_raw`` keeps the
    unshifted levels for reporting.  ``knocked_out`` marks a contract that
    already knocked out on a past observation (then NPV reduces to one
    discounted cash flow and the engines do not simulate anything).
    """

    underlying: str
    spot0: float
    anchored_on: str
    valuation_date: datetime
    expiry_date: datetime
    observation_dates: Tuple[datetime, ...]
    payment_dates: Tuple[datetime, ...]
    ko_levels_raw: Tuple[float, ...]
    ko_levels: Tuple[float, ...]
    ki_level_raw: float
    ki_levels: Tuple[float, ...]
    #: The grid the engines test the knock-in on: every business day by default
    #: (edslib's ``ki_dates``), the knock-out observations in the simplified
    #: convention.  ``ki_levels`` stays the per-period view used by the report.
    ki_dates: Tuple[datetime, ...]
    ki_monitor_levels: Tuple[float, ...]
    ki_frequency: str
    vol_times: Tuple[float, ...]
    discount_factors: Tuple[float, ...]
    expiry_vol_time: float
    expiry_discount_factor: float
    expiry_payment_date: datetime
    ko_shift: BarrierShiftSpec
    ki_shift: BarrierShiftSpec
    #: Barrier-touch convention per side (``inclusive`` / ``exclusive``): whether
    #: landing exactly on the level triggers.  Contractual - it is what the ledger
    #: replay and the same-day determination read, and it travels in the payload.
    ko_boundary: str = "inclusive"
    ki_boundary: str = "inclusive"
    #: Derived from :attr:`knocked_in_date` against the valuation date - the state
    #: the engines switch on.  ``knocked_in_date`` is the ledger record; the flag
    #: is re-derived whenever the valuation date moves (see :meth:`rebased`).
    knocked_in_before: bool = False
    knocked_in_date: Optional[datetime] = None
    knocked_out: bool = False
    knocked_out_index: Optional[int] = None
    knocked_out_cash: Optional[float] = None
    knocked_out_payment_date: Optional[datetime] = None
    knocked_out_discount_factor: Optional[float] = None
    notes: Tuple[str, ...] = ()
    # ---- cash-flow inputs: everything the engines multiply out ---------------
    #: Product tag, carried through so a reporter needs nothing else.
    product_type: str = "autocallable"
    #: Notional the cash flows scale with.
    notional: float = 1.0
    #: Accrual origin: the contractual start date (**the inception, not the
    #: valuation date** - a trade valued mid-life still accrues its coupon over
    #: the whole period, and a valuation-date bump must not move any cash flow).
    start_date: Optional[datetime] = None
    #: Day count of that accrual: ``act/365f`` (default) / ``act/360`` / ``act/act``.
    day_count: str = "act/365f"
    #: How many observations of the accrual schedule lie **before** this view of it.
    #: A payload carries the term-sheet levels plus the shift *rule*, and a stepwise
    #: rule divides its value over the contractual period count - so a payload
    #: written mid-life has to say how many periods are already behind it for the
    #: reader to reproduce the same levels (``rebased`` bumps it as dates drop).
    shift_elapsed: int = 0
    #: Annualised knock-out coupon, one per (future) observation.
    coupon_rates: Tuple[float, ...] = ()
    #: Total payoff ratio at expiry when neither knock-out nor knock-in fires
    #: (``1 + rebate_rate x accrual(start, expiry)``) - the engines' input.
    rebate_ratio: float = 1.0
    #: The **annual** rate behind :attr:`rebate_ratio`, kept so a payload can spell
    #: the rebate the same way as ``coupon_rates`` (rate, not total).  ``None`` only
    #: for schedules nobody built from terms or a payload (then the writer derives it).
    rebate_rate: Optional[float] = None
    #: Loss-leg terms: put strike (ratio of the anchor), gearing and floor.
    ki_strike: float = 1.0
    ki_gearing: float = 1.0
    protected_principal: float = 0.0
    #: Settlement lag, kept so a payload can rebuild ``payment_dates``.
    settlement_days: int = 0
    #: Which barrier today's fixing was read against (``contractual`` /
    #: ``effective``): a valuation input that rides along with the resolved view
    #: for reporting and audit.  It is **not** part of the contract, so
    #: :meth:`to_dict` leaves it out and a round-trip re-derives it from the
    #: caller's flag.
    trigger_basis: str = TRIGGER_CONTRACTUAL

    @property
    def n_observations(self) -> int:
        return len(self.observation_dates)

    @property
    def is_settled(self) -> bool:
        """True when the payoff is a single known cash flow (past knock-out)."""
        return bool(self.knocked_out)

    def rebased(self, market: MarketState) -> "AutocallSchedule":
        """The same resolved terms priced on another market state.

        The market side moves (valuation date, vol times, discount factors); the
        contractual data stays put - except that anything the new valuation date
        has left behind is dropped, exactly as :func:`build_schedule` does when
        it resolves the future view of a contract: the observations at or before
        it, and the knock-in monitoring dates too (with the monitoring level of
        the period each remaining date falls into).

        A bumped market in a Greek run is precisely this, which is what lets a
        JSON-fed contract be risked without a term-sheet object behind it.
        """
        valuation = to_datetime(market.valuation_date)
        changes: Dict[str, Any] = {
            "valuation_date": valuation,
            # the state is a function of (ledger date, valuation date): moving the
            # valuation back before the knock-in switches it off again
            "knocked_in_before": (
                self.knocked_in_date is not None
                and self.knocked_in_date.date() <= valuation.date()
            ),
            "expiry_vol_time": market.year_fraction(self.expiry_date),
            "expiry_discount_factor": market.discount_factor(self.expiry_payment_date),
            "knocked_out_discount_factor": (
                None
                if self.knocked_out_payment_date is None
                else float(market.discount_factor(self.knocked_out_payment_date))
            ),
        }
        if self.is_settled:
            return replace(self, **changes)

        keep = [
            index
            for index, day in enumerate(self.observation_dates)
            if day > valuation
        ]
        if len(keep) != len(self.observation_dates):
            changes.update(
                observation_dates=tuple(self.observation_dates[index] for index in keep),
                payment_dates=tuple(self.payment_dates[index] for index in keep),
                ko_levels=tuple(self.ko_levels[index] for index in keep),
                ko_levels_raw=tuple(self.ko_levels_raw[index] for index in keep),
                ki_levels=tuple(self.ki_levels[index] for index in keep),
                coupon_rates=tuple(self.coupon_rates[index] for index in keep),
                # the dropped periods are now *behind* the future view: a stepwise
                # shift resumes from there instead of restarting
                shift_elapsed=self.shift_elapsed
                + (len(self.observation_dates) - len(keep)),
            )
        changes["vol_times"] = tuple(
            market.year_fraction(day)
            for day in changes.get("observation_dates", self.observation_dates)
        )
        changes["discount_factors"] = tuple(
            market.discount_factor(day)
            for day in changes.get("payment_dates", self.payment_dates)
        )

        monitoring = [
            index for index, day in enumerate(self.ki_dates) if day > valuation
        ]
        if len(monitoring) != len(self.ki_dates):
            # the surviving dates keep the level they were resolved with - a custom
            # grid's levels are data, not something to re-derive from the periods
            changes["ki_dates"] = tuple(self.ki_dates[index] for index in monitoring)
            changes["ki_monitor_levels"] = tuple(
                self.ki_monitor_levels[index] for index in monitoring
            )
        return replace(self, **changes)

    # ----------------------------------------------------------- JSON layer
    def to_dict(self) -> Dict[str, Any]:
        """The resolved contract as a hand-editable payload.

        The barriers are the **term-sheet (pre-shift)** levels: the shift is a
        pricing *rule*, so it travels as a rule (``shift``) and the engine applies it
        - a reader can see "103% of spot, shifted −0.5% in four steps" instead of two
        versions of every level.  Everything else is resolved as before: absolute
        levels, one coupon rate per observation, the knock-in monitoring grid and the
        loss-leg terms.

        :meth:`from_dict` reads it back and applies the shift, so a quote can be
        exported to JSON, edited by hand and priced.
        """
        shift: Dict[str, Any] = {
            "ko": self.ko_shift.to_dict(),
            "ki": self.ki_shift.to_dict(),
        }
        if self.shift_elapsed:
            # a stepwise shift resumes where the full timeline left off; a payload
            # written at inception has nothing to say here (and says nothing)
            shift["elapsed"] = int(self.shift_elapsed)
        shift["notes"] = list(self.notes)
        return {
            "kind": "autocall_schedule",
            "version": 1,
            "underlying": self.underlying,
            "product_type": self.product_type,
            "spot0": float(self.spot0),
            "anchored_on": self.anchored_on,
            "start_date": None if self.start_date is None else _stamp(self.start_date),
            "day_count": self.day_count,
            "valuation_date": _stamp(self.valuation_date),
            "expiry_date": _stamp(self.expiry_date),
            "notional": float(self.notional),
            "settlement_days": int(self.settlement_days),
            "observations": [
                {
                    "date": day.date().isoformat(),
                    # pre-shift: the engine applies ``shift`` on read
                    "ko": float(self.ko_levels_raw[index]),
                    "coupon_rate": float(self.coupon_rates[index]),
                }
                for index, day in enumerate(self.observation_dates)
            ],
            # The barrier-touch conventions: contractual data, so they are written
            # out rather than implied ("inclusive" = touching the level triggers).
            "knock_out": {"boundary": self.ko_boundary},
            "knock_in": self._knock_in_dict(),
            "rebate": self._rebate_rate(),
            # the ledger record, not the derived flag: a reader re-derives the
            # state against *its* valuation date, which is what makes back-dating
            # a valuation work without touching the payload
            "knocked_in_date": (
                None
                if self.knocked_in_date is None
                else self.knocked_in_date.date().isoformat()
            ),
            # the settlement date is stored, the observation date is derived from
            # it: a settled trade's knock-out observation is no longer in the list
            # of future observations, so it cannot be indexed any more
            "knocked_out_at": (
                None
                if self.knocked_out_payment_date is None
                else (
                    self.knocked_out_payment_date
                    - timedelta(days=int(self.settlement_days))
                )
                .date()
                .isoformat()
            ),
            "knocked_out_cash": (
                None
                if self.knocked_out_cash is None
                else float(self.knocked_out_cash)
            ),
            "shift": shift,
        }

    def _accrual_to_expiry(self) -> float:
        """Accrual from the start date to expiry, in years (0 when unknown)."""
        if self.start_date is None or self.expiry_date is None:
            return 0.0
        return accrual_between(self.start_date, self.expiry_date, basis=self.day_count)

    def _rebate_rate(self) -> float:
        """The rebate's **annual rate** - the payload's unit, like ``coupon_rate``.

        The knock-out coupon is spelled as an annual rate and so is the rebate; the
        total the leg pays at expiry is derived from it
        (``1 + rate x accrual(start, expiry)``).  A schedule that only knows the ratio
        (nobody built it from terms or a payload) has the rate derived back.
        """
        if self.rebate_rate is not None:
            return float(self.rebate_rate)
        accrual = self._accrual_to_expiry()
        if accrual <= 0.0:
            return 0.0
        return (float(self.rebate_ratio) - 1.0) / accrual

    def _knock_in_dict(self) -> Dict[str, Any]:
        """The knock-in block: the pre-shift barrier, and a grid only when it is custom.

        ``level`` is the **term-sheet** barrier in absolute terms (the shift in
        ``shift.ki`` turns it into the level each period actually uses), which is why
        a stepwise knock-in shift no longer needs a per-period array here.  A
        daily-monitored snowball has ~250 monitoring dates and they are not
        contractual data either - ``frequency`` names the rule and the reader rebuilds
        the dates from the calendar.  ``custom`` is the one value that comes with its
        own explicit ``dates`` / ``levels`` (and those are used as written).
        """
        block: Dict[str, Any] = {
            "frequency": self.ki_frequency,
            "boundary": self.ki_boundary,
            "strike": float(self.ki_strike) * float(self.spot0),
            "gearing": float(self.ki_gearing),
            "protected_principal": float(self.protected_principal),
            "level": float(self.ki_level_raw),
        }
        if self.ki_frequency == CUSTOM_FREQUENCY:
            block["dates"] = [day.date().isoformat() for day in self.ki_dates]
            block["levels"] = [float(value) for value in self.ki_monitor_levels]
        return block

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        market: MarketState,
        *,
        trigger_basis: str = TRIGGER_CONTRACTUAL,
    ) -> "AutocallSchedule":
        """Rebuild a schedule from :meth:`to_dict` output.

        Only the *contractual* data is read from the payload; the market-derived
        pieces (vol times, discount factors, payment dates) come from ``market``,
        so a payload written for one day is priced on the curves of another.

        The barriers in the payload are the **term-sheet** ones and the shift is a
        rule, so the shift is applied *here* - the same ``expand_shift`` the term
        sheet path uses, with ``elapsed`` telling a stepwise rule how many periods of
        the accrual are already behind the payload.  Editing the rule edits the
        levels, which is the point of spelling it out.

        ``trigger_basis`` only steers the **same-day** determination below
        (:func:`resolve_today`): ``contractual`` reads the raw level, ``effective``
        the post-shift one, so an intraday quote keeps the greeks of the state the
        engines actually price.
        """
        payload = _unwrap(payload)
        trigger_basis = resolve_trigger_basis(trigger_basis)
        valuation = to_datetime(market.valuation_date)
        if "knocked_in_before" in payload:
            raise ValueError(
                "the payload carries the retired key 'knocked_in_before': use "
                "'knocked_in_date' instead - the date the knock-in happened (null "
                "when it has not).  The state is then derived against the valuation "
                "date, so a valuation can be moved back before it."
            )
        # A knock-in *after* the valuation date is not an error: it is exactly what
        # a valuation moved back before the knock-in looks like - the state below is
        # derived by comparing the two dates.
        knocked_in_date = _payload_day(payload.get("knocked_in_date"))
        expiry = to_datetime(payload["expiry_date"])
        if valuation >= expiry:
            raise ValueError(
                "contract already expired at {}: expiry is {}".format(
                    valuation.date(), expiry.date()
                )
            )

        items = list(payload.get("observations") or ())
        if not items:
            raise ValueError("observations must not be empty")
        observations = tuple(to_datetime(item["date"]) for item in items)
        knocked_out_at = payload.get("knocked_out_at")
        # A settled trade keeps its knock-out observation in the list even though
        # the valuation date has since moved past it; a live contract may only
        # carry future observations, exactly like the ones ``build_schedule``
        # resolves from raw terms.
        # An observation printed *on* the valuation date is a fixing, not a term:
        # it is kept here and decided by ``resolve_today`` below.
        if knocked_out_at is None and any(
            day.date() < valuation.date() for day in observations
        ):
            raise ValueError("every observation must be today or in the future")
        # the payload carries the *term-sheet* levels; the shift in ``shift`` is
        # applied below, exactly like ``build_schedule`` does for raw terms
        ko_levels_raw = tuple(float(item["ko"]) for item in items)
        coupon_rates = tuple(float(item.get("coupon_rate", 0.0)) for item in items)

        # A payload that carries a ``shift`` block is read **as written** - that
        # block is the record of the rule the levels above went through.  A payload
        # **without** one gets the *default* rule (the packaged config, or whatever
        # ``--shift-config`` pointed at), exactly like ``build_schedule`` does for
        # raw terms, so both entry points expand the same barrier (2026-10).  A
        # payload that wants no shift at all says so: ``{"ko": {"mode": "none"}}``.
        shift_block = payload.get("shift")
        if shift_block is None:
            ko_shift, ki_shift = resolve_shift(
                _shift_subject(payload, coupon_rates, observations),
                load_shift_config(),
            )
            shift_elapsed = 0
        else:
            ko_shift = BarrierShiftSpec.from_dict(shift_block.get("ko"))
            ki_shift = BarrierShiftSpec.from_dict(shift_block.get("ki"))
            shift_elapsed = max(int(shift_block.get("elapsed", 0) or 0), 0)
        shift = dict(shift_block or {})  # read below for its ``notes``

        knock_in = dict(payload.get("knock_in") or {})
        spot0 = float(payload["spot0"])
        # A shift *value* lives in barrier-**ratio** space (1.0 = 100% of the
        # anchor), the same space ``build_schedule`` expands in; the payload's
        # levels are absolute prices, so the rule is applied to the ratios - an
        # "additive -0.03" moves a 65% knock-in to 62% of spot 0, not by three
        # cents - and the result is scaled back to prices here.
        anchor = spot0 if spot0 > 0.0 else 1.0
        ko_levels = tuple(
            ratio * anchor
            for ratio in expand_shift(
                ko_shift,
                tuple(level / anchor for level in ko_levels_raw),
                elapsed=shift_elapsed,
            )
        )
        # The monitoring grid: ``custom`` carries its own dates and levels, every
        # other frequency is a *rule* the reader rebuilds from the calendar and the
        # observation dates - which is what keeps a daily-monitored payload short.
        grid_dates = tuple(to_datetime(value) for value in (knock_in.get("dates") or ()))
        grid_levels = tuple(float(value) for value in (knock_in.get("levels") or ()))
        frequency = resolve_monitoring_frequency(
            knock_in.get("frequency"), explicit=bool(grid_dates)
        )
        if frequency == CUSTOM_FREQUENCY:
            if not grid_dates:
                raise ValueError(
                    "knock_in.frequency 'custom' needs its own dates and levels"
                )
            if len(grid_levels) != len(grid_dates):
                raise ValueError(
                    "knock_in levels must match its dates "
                    "({} levels, {} dates)".format(len(grid_levels), len(grid_dates))
                )
            # the same window rule as every other frequency: a custom grid is a
            # future view, so anything the valuation date has left behind is dropped
            kept = [
                (day, level)
                for day, level in zip(grid_dates, grid_levels)
                if day > valuation
            ]
            ki_dates = tuple(day for day, _ in kept)
            grid_levels = tuple(level for _, level in kept)
        else:
            ki_dates = monitoring_grid(
                frequency, valuation, expiry, market.calendar, observations
            )
            grid_levels = ()

        # The term-sheet barrier, in absolute terms, for every observation period.
        if knock_in.get("level") is not None:
            ki_level_raw = float(knock_in["level"])
        elif grid_levels:
            ki_level_raw = grid_levels[-1]
        else:
            raise ValueError(
                "knock_in must state the barrier as 'level' (absolute), or as "
                "'dates' / 'levels' next to frequency 'custom'"
            )
        if ki_level_raw <= 0.0:
            raise ValueError("knock_in level must be positive: got {!r}".format(ki_level_raw))

        if frequency == CUSTOM_FREQUENCY:
            # a custom grid is used as written: its levels are the levels, so the
            # knock-in shift does not apply to them
            ki_levels = (ki_level_raw,) * len(observations)
            monitor_levels = grid_levels
        else:
            ki_levels = tuple(
                ratio * anchor
                for ratio in expand_shift(
                    ki_shift,
                    (ki_level_raw / anchor,) * len(observations),
                    elapsed=shift_elapsed,
                )
            )
            monitor_levels = tuple(
                period_level(ki_levels, observations, day) for day in ki_dates
            )

        # The loss-leg strike is absolute in the payload (a ratio of the anchor
        # internally, which is the space the cash-flow rule works in).
        strike = float(knock_in.get("strike", spot0))
        ki_strike = strike / spot0 if spot0 else 1.0
        if not 0.05 <= ki_strike <= 3.0:
            raise ValueError(
                "knock_in.strike is the loss-leg strike as an absolute price, not a "
                "ratio of the anchor: got {!r} (spot0 {} = {:.4%} of the anchor)".format(
                    knock_in.get("strike"), spot0, ki_strike
                )
            )

        settlement_days = int(payload.get("settlement_days", 0))
        payment_dates = tuple(day + timedelta(days=settlement_days) for day in observations)
        expiry_payment = expiry + timedelta(days=settlement_days)
        # The accrual origin is contractual: a payload that omits it would
        # otherwise accrue only from the valuation date, i.e. pay too little
        # coupon on every observation.
        if not payload.get("start_date"):
            raise ValueError(
                "the payload has no start_date: it is the coupon accrual origin "
                "and must not default to the valuation date"
            )
        start_date = to_datetime(payload["start_date"])
        day_count = resolve_basis(payload.get("day_count", "act/365f"))
        # ``rebate`` is an annual rate, the same unit as ``coupon_rate``; the ratio
        # the leg pays at expiry follows from it and the accrual.
        rebate_rate = float(payload.get("rebate", 0.0))
        if rebate_rate < 0.0:
            raise ValueError("rebate must not be negative")
        if rebate_rate >= 1.0:
            raise ValueError(
                "rebate is the annual rate of the no-KO / no-KI leg (same unit as "
                "coupon_rate), so it must be below 1.0: got {!r} - a value above 1 "
                "looks like the old 'total ratio' spelling".format(payload.get("rebate"))
            )
        rebate_ratio_value = rebate_ratio(
            rebate_rate, start_date, expiry, basis=day_count
        )
        # Barrier-touch conventions: contractual (default inclusive = touching the
        # level triggers), stated per side in the payload.
        knock_out = dict(payload.get("knock_out") or {})
        ko_boundary = resolve_boundary(knock_out.get("boundary"))
        ki_boundary = resolve_boundary(knock_in.get("boundary"))

        _check_knocked_in_date(
            knocked_in_date,
            frequency=frequency,
            observations=observations,
            market=market,
            expiry=expiry,
            start=start_date,
        )
        schedule = cls(
            underlying=str(payload.get("underlying", "")),
            product_type=str(payload.get("product_type", "autocallable")),
            spot0=float(payload["spot0"]),
            anchored_on=str(payload.get("anchored_on", "given")),
            valuation_date=valuation,
            expiry_date=expiry,
            observation_dates=observations,
            payment_dates=payment_dates,
            ko_levels_raw=ko_levels_raw,
            ko_levels=ko_levels,
            ki_level_raw=ki_level_raw,
            ki_levels=ki_levels,
            ki_dates=ki_dates,
            ki_monitor_levels=monitor_levels,
            ki_frequency=frequency,
            ko_boundary=ko_boundary,
            ki_boundary=ki_boundary,
            shift_elapsed=shift_elapsed,
            vol_times=tuple(market.year_fraction(day) for day in observations),
            discount_factors=tuple(market.discount_factor(day) for day in payment_dates),
            expiry_vol_time=market.year_fraction(expiry),
            expiry_discount_factor=market.discount_factor(expiry_payment),
            expiry_payment_date=expiry_payment,
            ko_shift=ko_shift,
            ki_shift=ki_shift,
            knocked_in_before=(
                knocked_in_date is not None
                and knocked_in_date.date() <= valuation.date()
            ),
            knocked_in_date=knocked_in_date,
            notes=tuple(shift.get("notes") or ()),
            notional=float(payload.get("notional", 1.0)),
            start_date=start_date,
            day_count=day_count,
            coupon_rates=coupon_rates,
            rebate_ratio=rebate_ratio_value,
            rebate_rate=rebate_rate,
            ki_strike=ki_strike,
            ki_gearing=float(knock_in.get("gearing", 1.0)),
            protected_principal=float(knock_in.get("protected_principal", 0.0)),
            settlement_days=settlement_days,
            trigger_basis=trigger_basis,
        )

        if knocked_out_at is None:
            # a same-day observation / monitoring date is a fixing, not a term
            schedule = resolve_today(schedule, market, basis=trigger_basis)
            if schedule.is_settled:
                return schedule

        if knocked_out_at:
            when = to_datetime(knocked_out_at)
            if when in observations:
                index: Optional[int] = observations.index(when)
                cash = float(
                    payload.get("knocked_out_cash")
                    if payload.get("knocked_out_cash") is not None
                    else schedule.notional
                    * (
                        1.0
                        + coupon_rates[index]
                        * accrual_between(start_date, when, basis=day_count)
                    )
                )
                payment = payment_dates[index]
            else:
                # a settled trade: the knock-out happened before the valuation
                # date, so it is no longer one of the (future) observations - the
                # payload has to state the cash it settled for
                index = None
                cash = payload.get("knocked_out_cash")
                if cash is None:
                    raise ValueError(
                        "knocked_out_at {} is not one of the observation dates, "
                        "so the payload must also carry knocked_out_cash".format(
                            when.date().isoformat()
                        )
                    )
                payment = when + timedelta(days=settlement_days)
            return _with_knock_out(
                schedule, index, float(cash), payment, market
            )
        return schedule


# ------------------------------------------------------------------ building
def build_schedule(
    contract: AutocallContract,
    market: MarketState,
    *,
    ko_shift: Optional[Any] = None,
    ki_shift: Optional[Any] = None,
    shift_config: Optional[ShiftConfig] = None,
    contractual: bool = False,
    trigger_basis: str = TRIGGER_CONTRACTUAL,
) -> AutocallSchedule:
    """Expand raw terms into the effective schedule (the shift's only home).

    ``ko_shift`` / ``ki_shift`` are this-run overrides (CLI level); the
    contract's persisted override sits below them and above the config file;
    ``contractual=True`` disables shifting entirely (price the raw terms).
    ``trigger_basis`` picks which barrier a **same-day** fixing is read against
    (see :func:`resolve_today`) and rides on the schedule for audit; it changes
    no term and no level.
    """
    valuation = to_datetime(market.valuation_date)
    if valuation >= contract.expiry_date:
        raise ValueError(
            "contract already expired at {}: expiry is {}".format(
                valuation.date(), contract.expiry_date.date()
            )
        )

    spot0, anchored_on = resolve_spot0(contract, market)

    ko_spec, ki_spec = _resolve_specs(
        contract,
        ko_shift=ko_shift,
        ki_shift=ki_shift,
        shift_config=shift_config,
        contractual=contractual,
    )

    observations = contract.observation_dates
    ko_levels_raw = tuple(level * spot0 for level in contract.ko_levels)
    ki_levels_raw = tuple(contract.ki_level * spot0 for _ in observations)

    # The historic part is ledger state and arrives as flags (see
    # ``autocall.history``); only its consistency with the terms is checked here
    knocked_in_date, past_index, knocked_out_at = _history_state(
        contract, market, observations
    )
    first_future = past_index + 1
    # the ledger stores the date; the state the engines read is derived from it
    knocked_in_before = (
        knocked_in_date is not None and knocked_in_date.date() <= valuation.date()
    )

    # The shift lives in **barrier-ratio space** (1.0 = 100% of the start spot),
    # exactly like the barriers themselves and like edslib: "additive -0.03"
    # moves a 75% knock-in to 72%, it does not move it by 3% of spot.  Absolute
    # levels are only the final step.
    ko_ratios = expand_shift(ko_spec, contract.ko_levels, start_index=first_future)
    ki_ratios = expand_shift(
        ki_spec,
        (contract.ki_level,) * len(observations),
        start_index=first_future,
    )
    ko_levels = tuple(ratio * spot0 for ratio in ko_ratios)
    ki_levels = tuple(ratio * spot0 for ratio in ki_ratios)

    # Knock-in monitoring grid.  Every monitoring date carries the level of the
    # observation period it falls into, so a stepwise shift keeps accruing
    # exactly as it does on the observation grid.
    ki_monitoring = contract.ki_monitoring_dates(
        market.calendar, valuation, contract.expiry_date
    )
    ki_monitor_levels = tuple(
        period_level(ki_levels, observations, when) for when in ki_monitoring
    )

    future = tuple(range(first_future, len(observations)))
    future_dates = tuple(observations[index] for index in future)
    # Resolved cash-flow inputs: the engines receive rates and ratios, never a
    # term-sheet rule.  The knock-out coupon is per observation (a step-up / -down
    # schedule is just a list); the rebate defaults to the last observation's
    # coupon, accrued to expiry.
    coupon_rates = tuple(float(contract.annual_coupon[index]) for index in future)
    rebate_rate = (
        float(contract.rebate)
        if contract.rebate is not None
        else float(contract.annual_coupon[-1])
    )
    rebate_ratio_value = rebate_ratio(
        rebate_rate, contract.start_date, contract.expiry_date, basis=contract.day_count
    )
    payment_dates = tuple(
        day + timedelta(days=contract.settlement_days) for day in future_dates
    )
    expiry_payment = contract.expiry_date + timedelta(days=contract.settlement_days)

    notes = [
        "anchor={}".format(anchored_on),
        "ko_shift={}".format(ko_spec.describe()),
        "ki_shift={}".format(ki_spec.describe()),
        "ki_frequency={} ({} monitoring dates)".format(
            contract.ki_frequency, len(ki_monitoring)
        ),
    ]
    if knocked_in_before:
        notes.append("knocked in on {}".format(knocked_in_date.date().isoformat()))

    schedule = AutocallSchedule(
        underlying=contract.underlying,
        product_type=contract.product_type,
        spot0=spot0,
        anchored_on=anchored_on,
        valuation_date=valuation,
        expiry_date=contract.expiry_date,
        observation_dates=future_dates,
        payment_dates=payment_dates,
        ko_levels_raw=tuple(ko_levels_raw[index] for index in future),
        ko_levels=tuple(ko_levels[index] for index in future),
        ki_level_raw=float(ki_levels_raw[0]) if ki_levels_raw else 0.0,
        ki_levels=tuple(ki_levels[index] for index in future),
        ki_dates=ki_monitoring,
        ki_monitor_levels=ki_monitor_levels,
        ki_frequency=contract.ki_frequency,
        ko_boundary=contract.ko_boundary,
        ki_boundary=contract.ki_boundary,
        vol_times=tuple(market.year_fraction(day) for day in future_dates),
        discount_factors=tuple(market.discount_factor(day) for day in payment_dates),
        expiry_vol_time=market.year_fraction(contract.expiry_date),
        expiry_discount_factor=market.discount_factor(expiry_payment),
        expiry_payment_date=expiry_payment,
        ko_shift=ko_spec,
        ki_shift=ki_spec,
        knocked_in_before=knocked_in_before,
        knocked_in_date=knocked_in_date,
        notes=tuple(notes),
        notional=float(contract.notional),
        start_date=contract.start_date,
        day_count=contract.day_count,
        # a seasoned contract: the past observations are behind this view, so a
        # stepwise shift keeps accruing where the term sheet is (they are dropped
        # right above, which is exactly what ``shift_elapsed`` accounts for)
        shift_elapsed=int(first_future),
        coupon_rates=coupon_rates,
        rebate_ratio=rebate_ratio_value,
        rebate_rate=rebate_rate,
        ki_strike=float(contract.ki_strike),
        ki_gearing=float(contract.ki_gearing),
        protected_principal=float(contract.protected_principal),
        settlement_days=int(contract.settlement_days),
        trigger_basis=resolve_trigger_basis(trigger_basis),
    )

    # an observation / monitoring date that falls *on* the valuation date is a
    # fixing: settle it (or mark the ledger) before the engines see anything
    schedule = resolve_today(schedule, market, basis=trigger_basis)
    if schedule.is_settled:
        return schedule

    if knocked_out_at is not None:
        # the rate is indexed on the *contract* observations, the cash on the
        # resolved notional - the past knock-out settles at its own coupon
        cash = schedule.notional * (
            1.0
            + float(contract.annual_coupon[knocked_out_at])
            * accrual_between(contract.start_date, observations[knocked_out_at])
        )
        pay_date = observations[knocked_out_at] + timedelta(days=contract.settlement_days)
        return _with_knock_out(schedule, knocked_out_at, cash, pay_date, market)

    return schedule


def _same_day_ki_level(
    schedule: AutocallSchedule,
    market: MarketState,
    basis: str = TRIGGER_CONTRACTUAL,
) -> Optional[float]:
    """The level a knock-in is tested against **today**, or ``None`` when today is
    not a monitoring date.

    Mirrors :func:`monitoring_grid`: a daily grid walks the calendar's business days
    and always includes an observation date, ``observation_dates`` only its own
    dates, ``expiry`` only maturity, and a ``custom`` grid is taken as written.  A
    rule-based grid tests the raw term-sheet barrier under ``contractual`` and the
    period's post-shift level under ``effective``; a custom grid is taken as written
    either way (its levels never carry a shift).
    """
    valuation = to_datetime(market.valuation_date)
    if schedule.ki_frequency == CUSTOM_FREQUENCY:
        for day, level in zip(schedule.ki_dates, schedule.ki_monitor_levels):
            if day.date() == valuation.date():
                return float(level)
        return None
    if schedule.ki_frequency == "expiry":
        if valuation.date() < schedule.expiry_date.date():
            return None
    else:
        # date granularity: a same-day observation is "today" whatever the time
        on_observation = any(
            day.date() == valuation.date() for day in schedule.observation_dates
        )
        if not on_observation:
            if schedule.ki_frequency == "observation_dates":
                return None
            calendar = getattr(market, "calendar", None)
            if calendar is not None and not calendar.is_business_day(valuation.date()):
                return None
    if basis == TRIGGER_EFFECTIVE:
        # the monitoring grid starts *after* today, so today's level is read
        # through the observation period the valuation date falls into
        return period_level(schedule.ki_levels, schedule.observation_dates, valuation)
    return float(schedule.ki_level_raw)


def resolve_today(
    schedule: AutocallSchedule,
    market: MarketState,
    *,
    basis: str = TRIGGER_CONTRACTUAL,
) -> AutocallSchedule:
    """Apply the events that fall **on the valuation date** to the ledger.

    The valuation date is a fixing date too: a daily knock-in grid monitors today,
    and an observation printed on today is observed at the spot the caller supplied
    (``--spot``, else the fit run's).  Both sides decide under their own
    ``boundary`` convention - so a spot that merely *touches* an inclusive barrier
    settles the trade (at that observation's accrued coupon) while an exclusive one
    needs the spot strictly through.

    ``basis`` picks the level.  ``contractual`` (default) reads the raw term-sheet
    barrier - the legal determination, what the ledger replay and every EOD run
    use.  ``effective`` reads the post-shift one, the line the engines price
    against: a spot already through the raw barrier but not yet through the shifted
    one stays *not knocked in*, which is the intraday reading - the desk keeps the
    greeks of the state the model is in until the close, when the EOD run on
    ``contractual`` makes the determination legal.

    What was observed today and did **not** trigger is behind this view afterwards,
    so it is pruned by :meth:`rebased` - which is also what keeps the engines from
    re-testing a fixing with their (smoothed) indicators.
    """
    basis = resolve_trigger_basis(basis)
    if schedule.is_settled:
        return schedule
    valuation = to_datetime(market.valuation_date)
    spot = float(market.spot)

    observed = False
    for index, day in enumerate(schedule.observation_dates):
        # date granularity: a fixing is "printed on" the valuation *date*, so a
        # valuation timestamp later in the day must not push it into the past
        if day.date() != valuation.date():
            continue
        observed = True
        # a same-day observation wins over the knock-in on a tied date, exactly
        # like the history replay
        level = (
            schedule.ko_levels[index]
            if basis == TRIGGER_EFFECTIVE
            else schedule.ko_levels_raw[index]
        )
        if triggers(spot, level, schedule.ko_boundary, above=True):
            return _with_knock_out(
                schedule,
                index,
                ko_cash_flow(schedule, index),
                schedule.payment_dates[index],
                market,
            )
        break

    ki_level = _same_day_ki_level(schedule, market, basis)
    if ki_level is not None:
        observed = True
        if not schedule.knocked_in_before and triggers(
            spot, ki_level, schedule.ki_boundary, above=False
        ):
            # the same-day determination is the ledger record for today
            schedule = replace(
                schedule, knocked_in_before=True, knocked_in_date=valuation
            )

    return schedule.rebased(market) if observed else schedule


def resolve_spot0(contract: AutocallContract, market: MarketState) -> Tuple[float, str]:
    if contract.anchor == "valuation_spot":
        return float(market.spot), "valuation_spot"
    if contract.start_spot is not None:
        return float(contract.start_spot), "start_spot"
    if to_date(contract.start_date) == to_date(market.valuation_date):
        return float(market.spot), "start_spot"
    raise ValueError(
        "start spot is required: the valuation date differs from the start date "
        "and AutocallContract.start_spot is not set"
    )


def _resolve_specs(
    contract: AutocallContract,
    *,
    ko_shift: Optional[Any],
    ki_shift: Optional[Any],
    shift_config: Optional[ShiftConfig],
    contractual: bool,
) -> Tuple[BarrierShiftSpec, BarrierShiftSpec]:
    if contractual:
        return BarrierShiftSpec(), BarrierShiftSpec()
    config = shift_config or load_shift_config()
    cli = {}
    if ko_shift is not None:
        cli["ko"] = ko_shift
    if ki_shift is not None:
        cli["ki"] = ki_shift
    return resolve_shift(
        contract,
        config,
        override=contract.shift_override,
        cli=cli or None,
    )


@dataclass
class _ShiftSubject:
    """A stand-in for :class:`AutocallContract` in :func:`resolve_shift`.

    The house shift rule is sized off the **trade** (coupon, notional, the accrual
    window, and the underlying class for the index / single-name rule), and
    ``resolve_shift`` only ever reads attributes - so a payload without a ``shift``
    block can be given the default rule without rebuilding a contract.
    """

    product_type: str
    underlying: str
    start_date: Optional[datetime]
    annual_coupon: Tuple[float, ...]
    notional: float
    observation_dates: Tuple[datetime, ...]

    @property
    def coupon_rate(self) -> float:
        """The base coupon the rule scales off: the first observation's rate."""
        return float(self.annual_coupon[0]) if self.annual_coupon else 0.0


def _shift_subject(
    payload: Mapping[str, Any], coupon_rates, observations
) -> _ShiftSubject:
    """Build the :class:`_ShiftSubject` a shift rule needs from the payload."""
    return _ShiftSubject(
        product_type=str(payload.get("product_type") or "autocallable"),
        underlying=str(payload.get("underlying") or ""),
        start_date=_payload_day(payload.get("start_date")),
        annual_coupon=tuple(float(rate) for rate in coupon_rates),
        notional=float(payload.get("notional", 1.0) or 1.0),
        observation_dates=tuple(observations),
    )


def _payload_day(value: Any) -> Optional[datetime]:
    """A payload date field: ``None`` / ISO string / date / datetime -> datetime."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return to_datetime(value)


def _check_knocked_in_date(
    when: Optional[datetime],
    *,
    frequency: str,
    observations,
    market: MarketState,
    expiry: Optional[datetime] = None,
    start: Optional[datetime] = None,
) -> None:
    """Structural checks on a ledger knock-in date.

    The *fixing* that triggered it is the caller's evidence - the engine has no
    price for that day - so only the shape is verified: the date must fall inside
    the contract's life and on a date the knock-in grid can actually produce.
    ``daily`` monitors business days (plus the observation dates, which are
    monitored whether or not they are open), ``expiry`` only the maturity,
    ``observation_dates`` the observations, and a ``custom`` grid is data the
    payload owns - a date no longer listed there (a knocked-in trade keeps only its
    future monitoring dates) is accepted.

    A date **after the valuation date** is fine: that is a valuation moved back
    before the knock-in, and the state is derived from the comparison.
    """
    if when is None:
        return
    if start is not None and when.date() <= start.date():
        raise ValueError(
            "knocked_in_date {} must be after the start date {}".format(
                when.date().isoformat(), start.date().isoformat()
            )
        )
    if expiry is not None and when.date() > expiry.date():
        raise ValueError(
            "knocked_in_date {} must not be past the expiry date {}".format(
                when.date().isoformat(), expiry.date().isoformat()
            )
        )
    if frequency == "expiry":
        if expiry is None or when.date() != expiry.date():
            raise ValueError(
                "knocked_in_date {} must be the expiry date {}: the european "
                "knock-in is only tested at maturity".format(
                    when.date().isoformat(),
                    "?" if expiry is None else expiry.date().isoformat(),
                )
            )
        return
    if frequency == "daily":
        on_observation = any(day.date() == when.date() for day in observations)
        calendar = getattr(market, "calendar", None)
        if (
            not on_observation
            and calendar is not None
            and not calendar.is_business_day(when.date())
        ):
            raise ValueError(
                "knocked_in_date {} is not a monitoring date: the daily knock-in "
                "grid follows the business days (and the observation dates)".format(
                    when.date().isoformat()
                )
            )
        return
    if frequency == "observation_dates" and observations:
        first, last = observations[0].date(), observations[-1].date()
        if first <= when.date() <= last and not any(
            day.date() == when.date() for day in observations
        ):
            raise ValueError(
                "knocked_in_date {} is not one of the observation dates".format(
                    when.date().isoformat()
                )
            )


def _history_state(
    contract: AutocallContract,
    market: MarketState,
    observations: Tuple[datetime, ...],
) -> Tuple[Optional[datetime], int, Optional[int]]:
    """Validate the ledger dates: (knocked-in date, last past index, KO index).

    The dates themselves come from :func:`autocall.history.replay_history`; a
    knock-in date the monitoring grid cannot produce, a knock-in after the
    valuation date, or a knock-out that is not a past observation is a bookkeeping
    error, not something to price around.
    """
    valuation = to_datetime(market.valuation_date)
    # date granularity: an observation printed *on* the valuation date is a fixing
    # for ``resolve_today``, not something the ledger has already left behind
    past = [
        index for index, day in enumerate(observations) if day.date() < valuation.date()
    ]
    last_past = past[-1] if past else -1
    knocked_in_date = _payload_day(contract.knocked_in_date)
    _check_knocked_in_date(
        knocked_in_date,
        frequency=contract.ki_frequency,
        observations=observations,
        market=market,
        expiry=contract.expiry_date,
        start=contract.start_date,
    )
    if contract.knocked_out_at is None:
        return knocked_in_date, last_past, None

    knocked_out = to_datetime(contract.knocked_out_at)
    if knocked_out not in observations:
        raise ValueError(
            "knocked_out_at {} is not one of the observation dates".format(
                knocked_out.date().isoformat()
            )
        )
    index = observations.index(knocked_out)
    if index not in past:
        raise ValueError(
            "knocked_out_at {} must be a past observation (valuation date {})".format(
                knocked_out.date().isoformat(), valuation.date().isoformat()
            )
        )
    return knocked_in_date, last_past, index


def _stamp(value: datetime) -> str:
    """ISO stamp of a schedule date (dates keep midnight, timestamps keep time)."""
    return value.isoformat()


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a bare schedule payload or a whole quote (``{"contract": ..}``).

    The ``price_autocall --json`` output is a quote *around* the resolved
    contract, so a quote exported from the CLI can be edited and fed straight
    back without stripping the wrapper.
    """
    if "observations" not in payload and isinstance(payload.get("contract"), Mapping):
        return payload["contract"]
    return payload


def _with_knock_out(
    schedule: AutocallSchedule,
    index: Optional[int],
    cash: float,
    payment_date: datetime,
    market: MarketState,
) -> AutocallSchedule:
    return replace(
        schedule,
        knocked_out=True,
        knocked_out_index=None if index is None else int(index),
        knocked_out_cash=float(cash),
        knocked_out_payment_date=payment_date,
        knocked_out_discount_factor=float(market.discount_factor(payment_date)),
    )


__all__ = ["AutocallSchedule", "build_schedule", "resolve_spot0"]
