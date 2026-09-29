"""EDS SABR surface container and interpolation rules."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np

from ..core.daycount import BusinessCalendar, DateLike, to_date, to_datetime, year_fraction
from .eds_slice import EDSSabrSlice, EDSSliceParameters


# Single-form smile parameter name -> surface array name (see ``_SMILE_FIELDS``).
SMILE_NAME_TO_FIELD: Dict[str, str] = {
    "skew": "skews",
    "conv": "convs",
    "left_skew_1": "left_skews_1",
    "left_skew_2": "left_skews_2",
    "right_skew_1": "right_skews_1",
    "right_skew_2": "right_skews_2",
}


_SMILE_FIELDS = (
    "skews",
    "convs",
    "left_skews_1",
    "right_skews_1",
    "left_skews_2",
    "right_skews_2",
)


@dataclass
class EDSSabrSurface:
    init_date: DateLike
    init_spot: float
    expiry_dates: Iterable[DateLike]
    atm_vols: Iterable[float]
    skews: Optional[Iterable[float]] = None
    convs: Optional[Iterable[float]] = None
    left_skews_1: Optional[Iterable[float]] = None
    right_skews_1: Optional[Iterable[float]] = None
    left_skews_2: Optional[Iterable[float]] = None
    right_skews_2: Optional[Iterable[float]] = None
    stickiness_ratio: float = 1.0
    calendar: Optional[BusinessCalendar] = None
    trading_days_per_year: Optional[float] = None
    holiday_weight: float = 0.0
    interpolation_method: str = "direct"

    def __post_init__(self):
        self.init_date = to_datetime(self.init_date)
        self.init_spot = float(self.init_spot)
        self.expiry_dates = np.asarray(
            [to_datetime(value) for value in self.expiry_dates], dtype=object
        )
        order = np.argsort(self.expiry_dates)
        self.expiry_dates = self.expiry_dates[order]
        self.atm_vols = np.asarray(self.atm_vols, dtype=float)[order]
        count = len(self.expiry_dates)
        for field_name in _SMILE_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                value = np.zeros(count, dtype=float)
            value = np.asarray(value, dtype=float)
            if len(value) != count:
                raise ValueError("{} must have one value per expiry".format(field_name))
            setattr(self, field_name, value[order])
        if len(self.atm_vols) != count:
            raise ValueError("atm_vols must have one value per expiry")
        if not 0.0 <= float(self.stickiness_ratio) <= 1.0:
            raise ValueError("stickiness_ratio must be between 0 and 1")
        self.stickiness_ratio = float(self.stickiness_ratio)
        self.interpolation_method = str(self.interpolation_method or "direct").lower()
        if self.interpolation_method not in {"direct", "slice", "total_variance", "variance"}:
            raise ValueError("interpolation_method must be direct or total_variance")
        self.expiry_times = np.asarray(
            [
                year_fraction(
                    self.init_date,
                    expiry,
                    calendar=self.calendar,
                    trading_days_per_year=self.trading_days_per_year,
                    holiday_weight=self.holiday_weight,
                )
                for expiry in self.expiry_dates
            ],
            dtype=float,
        )

    def clone(self, **changes) -> "EDSSabrSurface":
        values = {
            "init_date": self.init_date,
            "init_spot": self.init_spot,
            "expiry_dates": self.expiry_dates.copy(),
            "atm_vols": self.atm_vols.copy(),
            "skews": self.skews.copy(),
            "convs": self.convs.copy(),
            "left_skews_1": self.left_skews_1.copy(),
            "right_skews_1": self.right_skews_1.copy(),
            "left_skews_2": self.left_skews_2.copy(),
            "right_skews_2": self.right_skews_2.copy(),
            "stickiness_ratio": self.stickiness_ratio,
            "calendar": self.calendar,
            "trading_days_per_year": self.trading_days_per_year,
            "holiday_weight": self.holiday_weight,
            "interpolation_method": self.interpolation_method,
        }
        values.update(changes)
        return EDSSabrSurface(**values)

    def with_init_date(self, init_date: DateLike) -> "EDSSabrSurface":
        return self.clone(init_date=init_date)

    def get_vol_time(self, expiry: DateLike, valuation_date: Optional[DateLike] = None) -> float:
        start = self.init_date if valuation_date is None else to_datetime(valuation_date)
        return year_fraction(
            start,
            expiry,
            calendar=self.calendar,
            trading_days_per_year=self.trading_days_per_year,
            holiday_weight=self.holiday_weight,
        )

    @staticmethod
    def _linear(time: float, pillar_times: np.ndarray, values: np.ndarray) -> float:
        if len(values) == 1:
            return float(values[0])
        if time <= pillar_times[0]:
            return float(values[0])
        if time >= pillar_times[-1]:
            return float(values[-1])
        index = int(np.searchsorted(pillar_times, time, side="left"))
        left = index - 1
        weight = (pillar_times[index] - time) / (pillar_times[index] - pillar_times[left])
        return float(values[left] * weight + values[index] * (1.0 - weight))

    def get_atm_vol(self, vol_time: float) -> float:
        if len(self.expiry_times) == 1:
            return float(self.atm_vols[0])
        if vol_time <= 0.0 or vol_time <= self.expiry_times[0]:
            return float(self.atm_vols[0])
        if vol_time < self.expiry_times[-1]:
            index = int(np.searchsorted(self.expiry_times, vol_time, side="left"))
            if abs(self.expiry_times[index] - vol_time) < 1.0e-14:
                return float(self.atm_vols[index])
            left = index - 1
            weight = (self.expiry_times[index] - vol_time) / (
                self.expiry_times[index] - self.expiry_times[left]
            )
            variance = (
                self.atm_vols[left] ** 2 * self.expiry_times[left] * weight
                + self.atm_vols[index] ** 2 * self.expiry_times[index] * (1.0 - weight)
            )
            return max(1.0e-5, float(np.sqrt(max(0.0, variance) / vol_time)))

        variance_left = self.atm_vols[-2] ** 2 * self.expiry_times[-2]
        variance_right = self.atm_vols[-1] ** 2 * self.expiry_times[-1]
        weight = (self.expiry_times[-1] - vol_time) / (
            self.expiry_times[-1] - self.expiry_times[-2]
        )
        variance = variance_left * weight + variance_right * (1.0 - weight)
        return max(1.0e-5, float(np.sqrt(max(0.0, variance) / vol_time)))

    def _get_smile_parameters(self, vol_time: float):
        return {
            field_name: self._linear(vol_time, self.expiry_times, getattr(self, field_name))
            for field_name in _SMILE_FIELDS
        }

    def have_smile(self) -> bool:
        return any(np.any(np.abs(getattr(self, field_name)) > 1.0e-14) for field_name in _SMILE_FIELDS)

    def _reference_strike(self, current_forward: float, initial_forward: Optional[float]) -> float:
        current_forward = float(current_forward)
        if initial_forward is None or not self.have_smile() or self.stickiness_ratio == 0.0:
            initial_forward = current_forward
        return float(initial_forward) ** self.stickiness_ratio * current_forward ** (1.0 - self.stickiness_ratio)

    def get_slice(
        self,
        expiry: DateLike,
        current_forward: float,
        initial_forward: Optional[float] = None,
        valuation_date: Optional[DateLike] = None,
    ) -> EDSSabrSlice:
        vol_time = self.get_vol_time(expiry, valuation_date=valuation_date)
        atm_vol = self.get_atm_vol(vol_time)
        parameters = self._get_smile_parameters(vol_time)
        current_forward = float(current_forward)
        ref_strike = self._reference_strike(current_forward, initial_forward)

        if vol_time <= 0.0:
            scale = 1.0
            parameters = {name: 0.0 for name in _SMILE_FIELDS}
        else:
            scale = max(0.3, np.sqrt(vol_time))
            parameters = {name: value / scale for name, value in parameters.items()}

        return EDSSabrSlice(
            ref_strike=ref_strike,
            forward=current_forward,
            vol_atmf=atm_vol,
            tau=max(vol_time, 1.0e-12),
            skew=parameters["skews"],
            conv=parameters["convs"],
            left_skew_1=parameters["left_skews_1"],
            right_skew_1=parameters["right_skews_1"],
            left_skew_2=parameters["left_skews_2"],
            right_skew_2=parameters["right_skews_2"],
        )

    def implied_vol(
        self,
        expiry: DateLike,
        strikes,
        current_forward: float,
        initial_forward: Optional[float] = None,
        valuation_date: Optional[DateLike] = None,
        interpolation_method: Optional[str] = None,
        forward_resolver: Optional[Callable[[DateLike], float]] = None,
        initial_forward_resolver: Optional[Callable[[DateLike], float]] = None,
    ):
        method = str(interpolation_method or self.interpolation_method).lower()
        if method in {"total_variance", "variance"}:
            return self._implied_vol_total_variance(
                expiry,
                strikes,
                current_forward=current_forward,
                initial_forward=initial_forward,
                valuation_date=valuation_date,
                forward_resolver=forward_resolver,
                initial_forward_resolver=initial_forward_resolver,
            )

        slice_ = self.get_slice(
            expiry,
            current_forward=current_forward,
            initial_forward=initial_forward,
            valuation_date=valuation_date,
        )
        return slice_.get_implied_vol(strikes)

    def _resolve_forward(
        self,
        expiry: DateLike,
        fallback: float,
        resolver: Optional[Callable[[DateLike], float]],
    ) -> float:
        if resolver is None:
            return float(fallback)
        return float(resolver(expiry))

    def _resolve_initial_forward(
        self,
        expiry: DateLike,
        current_forward: float,
        fallback: Optional[float],
        resolver: Optional[Callable[[DateLike], float]],
    ) -> float:
        if not self.have_smile() or self.stickiness_ratio == 0.0:
            return float(current_forward)
        if resolver is not None:
            return float(resolver(expiry))
        if fallback is not None:
            return float(fallback)
        return float(current_forward)

    def _slice_for_total_variance(
        self,
        expiry: DateLike,
        fallback_forward: float,
        fallback_initial_forward: Optional[float],
        valuation_date: Optional[DateLike],
        forward_resolver: Optional[Callable[[DateLike], float]],
        initial_forward_resolver: Optional[Callable[[DateLike], float]],
    ) -> EDSSabrSlice:
        forward = self._resolve_forward(expiry, fallback_forward, forward_resolver)
        initial_forward = self._resolve_initial_forward(
            expiry,
            forward,
            fallback_initial_forward,
            initial_forward_resolver,
        )
        return self.get_slice(
            expiry,
            current_forward=forward,
            initial_forward=initial_forward,
            valuation_date=valuation_date,
        )

    def _implied_vol_total_variance(
        self,
        expiry: DateLike,
        strikes,
        current_forward: float,
        initial_forward: Optional[float],
        valuation_date: Optional[DateLike],
        forward_resolver: Optional[Callable[[DateLike], float]],
        initial_forward_resolver: Optional[Callable[[DateLike], float]],
    ):
        expiry_dt = to_datetime(expiry)
        strike_array = np.atleast_1d(np.asarray(strikes, dtype=float))
        vol_time = self.get_vol_time(expiry_dt, valuation_date=valuation_date)
        if vol_time <= 0.0:
            return self.implied_vol(
                expiry_dt,
                strike_array,
                current_forward=current_forward,
                initial_forward=initial_forward,
                valuation_date=valuation_date,
                interpolation_method="direct",
            )

        query_initial_forward = self._resolve_initial_forward(
            expiry_dt,
            current_forward,
            initial_forward,
            initial_forward_resolver,
        )
        query_ref_strike = self._reference_strike(current_forward, query_initial_forward)
        moneyness = strike_array / query_ref_strike
        pillar_times = np.asarray(
            [self.get_vol_time(date, valuation_date=valuation_date) for date in self.expiry_dates],
            dtype=float,
        )
        exact = np.where(self.expiry_dates == expiry_dt)[0]
        if len(exact):
            slice_ = self._slice_for_total_variance(
                expiry_dt,
                current_forward,
                initial_forward,
                valuation_date,
                forward_resolver,
                initial_forward_resolver,
            )
            return slice_.get_implied_vol(strike_array)

        index = int(np.searchsorted(self.expiry_dates, expiry_dt, side="left"))
        if index == 0:
            post_slice = self._slice_for_total_variance(
                self.expiry_dates[0],
                current_forward,
                initial_forward,
                valuation_date,
                forward_resolver,
                initial_forward_resolver,
            )
            post_vols = post_slice.get_implied_vol(post_slice.ref_strike * moneyness)
            post_variance = post_vols ** 2 * pillar_times[0]
            variance = post_variance * vol_time / max(pillar_times[0], 1.0e-12)
        elif index >= len(self.expiry_dates):
            if len(self.expiry_dates) == 1:
                pre_slice = self._slice_for_total_variance(
                    self.expiry_dates[0],
                    current_forward,
                    initial_forward,
                    valuation_date,
                    forward_resolver,
                    initial_forward_resolver,
                )
                return pre_slice.get_implied_vol(pre_slice.ref_strike * moneyness)
            pre_pre_slice = self._slice_for_total_variance(
                self.expiry_dates[-2],
                current_forward,
                initial_forward,
                valuation_date,
                forward_resolver,
                initial_forward_resolver,
            )
            pre_slice = self._slice_for_total_variance(
                self.expiry_dates[-1],
                current_forward,
                initial_forward,
                valuation_date,
                forward_resolver,
                initial_forward_resolver,
            )
            pre_pre_vols = pre_pre_slice.get_implied_vol(pre_pre_slice.ref_strike * moneyness)
            pre_vols = pre_slice.get_implied_vol(pre_slice.ref_strike * moneyness)
            pre_pre_variance = pre_pre_vols ** 2 * pillar_times[-2]
            pre_variance = pre_vols ** 2 * pillar_times[-1]
            slope = (pre_variance - pre_pre_variance) / (pillar_times[-1] - pillar_times[-2])
            variance = pre_variance + slope * (vol_time - pillar_times[-1])
        else:
            pre_slice = self._slice_for_total_variance(
                self.expiry_dates[index - 1],
                current_forward,
                initial_forward,
                valuation_date,
                forward_resolver,
                initial_forward_resolver,
            )
            post_slice = self._slice_for_total_variance(
                self.expiry_dates[index],
                current_forward,
                initial_forward,
                valuation_date,
                forward_resolver,
                initial_forward_resolver,
            )
            pre_vols = pre_slice.get_implied_vol(pre_slice.ref_strike * moneyness)
            post_vols = post_slice.get_implied_vol(post_slice.ref_strike * moneyness)
            pre_variance = pre_vols ** 2 * pillar_times[index - 1]
            post_variance = post_vols ** 2 * pillar_times[index]
            weight = (vol_time - pillar_times[index - 1]) / (pillar_times[index] - pillar_times[index - 1])
            variance = pre_variance * (1.0 - weight) + post_variance * weight
        return np.sqrt(np.maximum(variance, 0.0) / max(vol_time, 1.0e-12))

    def bump_parallel(self, amount: float) -> "EDSSabrSurface":
        bumped = self.clone()
        bumped.atm_vols = np.maximum(1.0e-5, bumped.atm_vols + amount)
        return bumped

    def bump_pillar(self, pillar, amount: float) -> "EDSSabrSurface":
        bumped = self.clone()
        if isinstance(pillar, int):
            index = int(pillar)
        else:
            target = to_datetime(pillar)
            index = int(np.argmin([abs(to_datetime(item) - target) for item in self.expiry_dates]))
        bumped.atm_vols[index] = max(1.0e-5, bumped.atm_vols[index] + amount)
        return bumped

    def bump_cumulative_backward(self, pillar, amount: float) -> "EDSSabrSurface":
        bumped = self.clone()
        if isinstance(pillar, int):
            index = int(pillar)
        else:
            target = to_datetime(pillar)
            index = int(np.searchsorted(bumped.expiry_dates, target, side="left"))
        bumped.atm_vols[index:] = np.maximum(1.0e-5, bumped.atm_vols[index:] + amount)
        return bumped

    def pillar_index(self, expiry: DateLike) -> Optional[int]:
        """Position of ``expiry`` among the pillars, or ``None`` when absent."""
        target = to_date(expiry)
        for index, value in enumerate(self.expiry_dates):
            if to_date(value) == target:
                return index
        return None

    def set_pillar(
        self,
        expiry: DateLike,
        *,
        atm_vol: Optional[float] = None,
        smile: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Overwrite one existing pillar in place.

        ``smile`` keys use the single-form names (``skew`` / ``left_skew_1`` ...).
        The pillar date must already be present; call :meth:`rebuild` first to
        add newer expiries.
        """
        index = self.pillar_index(expiry)
        if index is None:
            raise ValueError(
                "expiry {} is not a pillar of this surface".format(to_date(expiry))
            )
        if atm_vol is not None:
            self.atm_vols[index] = float(atm_vol)
        for name, value in (smile or {}).items():
            field_name = SMILE_NAME_TO_FIELD.get(str(name))
            if field_name is None:
                raise ValueError("unknown smile parameter {!r}".format(name))
            getattr(self, field_name)[index] = float(value)

    def rebuild(self, pillar_dates: Iterable[DateLike]) -> "EDSSabrSurface":
        dates = np.asarray([to_datetime(value) for value in pillar_dates], dtype=object)
        values = {
            field_name: np.asarray(
                [self._linear(self.get_vol_time(date), self.expiry_times, getattr(self, field_name)) for date in dates]
            )
            for field_name in _SMILE_FIELDS
        }
        atm = np.asarray([self.get_atm_vol(self.get_vol_time(date)) for date in dates])
        return EDSSabrSurface(
            init_date=self.init_date,
            init_spot=self.init_spot,
            expiry_dates=dates,
            atm_vols=atm,
            stickiness_ratio=self.stickiness_ratio,
            calendar=self.calendar,
            trading_days_per_year=self.trading_days_per_year,
            holiday_weight=self.holiday_weight,
            interpolation_method=self.interpolation_method,
            **values,
        )

    def to_dict(self) -> dict:
        result = {
            "type": "eds_sabr",
            "init_date": self.init_date.isoformat(sep=" "),
            "init_spot": self.init_spot,
            "expiry_dates": [to_datetime(value).isoformat(sep=" ") for value in self.expiry_dates],
            "stickiness_ratio": self.stickiness_ratio,
            "atm_vols": self.atm_vols.tolist(),
            "skews": self.skews.tolist(),
            "convs": self.convs.tolist(),
            "left_skews_1": self.left_skews_1.tolist(),
            "right_skews_1": self.right_skews_1.tolist(),
            "left_skews_2": self.left_skews_2.tolist(),
            "right_skews_2": self.right_skews_2.tolist(),
            "interpolation_method": self.interpolation_method,
        }
        if self.calendar is not None:
            result["calendar"] = self.calendar.name
        if self.trading_days_per_year is not None:
            result["trading_days_per_year"] = self.trading_days_per_year
        if self.holiday_weight:
            result["holiday_weight"] = self.holiday_weight
        return result
