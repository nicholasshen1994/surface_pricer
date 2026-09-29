"""JSON adapters for standalone market and contract inputs."""

import json
from typing import Optional

from ..core.curves import ConstantRateCurve, PiecewiseRateCurve
from ..core.daycount import BusinessCalendar, DateHelperBusinessCalendar, to_datetime
from ..core.market import MarketState
from ..fitting.surface import EDSSabrSurface
from ..marketdata.offline_quotes import OptionQuote, QuoteSlice
from ..pricing.contracts import VanillaContract


def curve_from_dict(value, anchor=None, calendar=None, trading_days_per_year=None):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return ConstantRateCurve(float(value), anchor=anchor)
    curve_type = str(value.get("type", "")).lower()
    if curve_type in {"constant_rate_curve", "constant_borrow_rate", "constant"}:
        return ConstantRateCurve(float(value.get("value", value.get("rate", 0.0))), anchor=anchor)
    return PiecewiseRateCurve(
        anchor=value.get("valuation_time", anchor),
        tenors=value.get("tenors", []),
        rates=value.get("rates", []),
        calendar=calendar,
        trading_days_per_year=trading_days_per_year,
        interpolation=value.get("interpolation_method", "linear_zero"),
        basis=value.get("day_counter", "act/365f"),
    )


def calendar_from_name(
    name: Optional[str],
    calendar_file: Optional[str] = None,
) -> Optional[BusinessCalendar]:
    """Rebuild a calendar from a stored name.

    Names known to the bundled ``DateHelper`` (``SHX``) come back with their
    full holiday list, so vol times and tenor rolls rebuilt from a
    ``surface.json`` match the ones used by the fit.  An explicit
    ``calendar_file`` takes precedence; unknown names fall back to a plain
    weekday calendar.
    """
    if not name:
        return None
    if calendar_file:
        return BusinessCalendar.from_json(calendar_file, name)
    try:
        return DateHelperBusinessCalendar(name)
    except KeyError:
        return BusinessCalendar(name=name)


def surface_from_dict(
    value: dict,
    calendar: Optional[BusinessCalendar] = None,
) -> EDSSabrSurface:
    return EDSSabrSurface(
        init_date=value["init_date"],
        init_spot=value["init_spot"],
        expiry_dates=value["expiry_dates"],
        atm_vols=value["atm_vols"],
        skews=value.get("skews"),
        convs=value.get("convs"),
        left_skews_1=value.get("left_skews_1"),
        right_skews_1=value.get("right_skews_1"),
        left_skews_2=value.get("left_skews_2"),
        right_skews_2=value.get("right_skews_2"),
        stickiness_ratio=value.get("stickiness_ratio", 1.0),
        calendar=calendar,
        trading_days_per_year=value.get("trading_days_per_year"),
        holiday_weight=value.get("holiday_weight", 0.0),
        interpolation_method=value.get("interpolation_method", "direct"),
    )


def market_from_dict(
    value: dict,
    underlying: Optional[str] = None,
    calendar_file: Optional[str] = None,
) -> MarketState:
    valuation_date = value.get("valuation_date")
    products = value.get("products", value)
    if underlying is None:
        underlying = next(
            name
            for name, product in products.items()
            if product.get("type") == "equity"
        )
    equity = products[underlying]
    currency_name = equity.get("currency")
    currency = products.get(currency_name, {})
    calendar_name = (
        equity.get("vol_surface", {}).get("calendar")
        if equity.get("vol_surface")
        else equity.get("calendar")
    )
    calendar = calendar_from_name(calendar_name, calendar_file=calendar_file)
    tdpy = (
        equity.get("vol_surface", {}).get("trading_days_per_year")
        if equity.get("vol_surface")
        else None
    )
    surface = (
        surface_from_dict(equity["vol_surface"], calendar=calendar)
        if equity.get("vol_surface")
        and str(equity["vol_surface"].get("type", "")).lower() == "eds_sabr"
        else None
    )
    rate_curve = curve_from_dict(
        currency.get("rate_curve", currency.get("rate", 0.0)),
        anchor=valuation_date,
        calendar=calendar,
        trading_days_per_year=tdpy,
    )
    borrow_curve = curve_from_dict(
        equity.get("borrow_rate"),
        anchor=valuation_date,
        calendar=calendar,
        trading_days_per_year=tdpy,
    )
    return MarketState(
        valuation_date=valuation_date,
        spot=equity["spot"],
        rate_curve=rate_curve,
        borrow_curve=borrow_curve,
        calendar=calendar,
        trading_days_per_year=tdpy,
        holiday_weight=equity.get("vol_surface", {}).get("holiday_weight", 0.0),
        surface=surface,
    )


def contract_from_dict(value: dict) -> VanillaContract:
    return VanillaContract(
        expiry=value.get("expiry_date", value.get("expiry")),
        strike=value["strike"],
        option_type=value.get("call_put", value.get("option_type", "call")),
        notional=value.get("notional", 1.0),
        strike_type=value.get("strike_type", "absolute"),
    )


def quote_slices_from_dict(value: dict, default_spot=None):
    if isinstance(value, dict) and "price_info" not in value and "quotes" not in value:
        return [
            QuoteSlice.from_dict(item, default_spot=default_spot)
            for item in value.values()
        ]
    return [QuoteSlice.from_dict(value, default_spot=default_spot)]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def market_from_surface(
    value: dict,
    *,
    spot: Optional[float] = None,
    rate: float = 0.0,
    borrow: float = 0.0,
    valuation_date=None,
    calendar_file: Optional[str] = None,
    rate_curve=None,
    borrow_curve=None,
) -> MarketState:
    """Build a single-underlying market state around a fitted surface payload.

    This is the entry point for files written by ``surface_pricer fit``
    (``surface.json``): spot, valuation date, calendar, vol-time convention and
    holiday weight all come from the payload unless overridden.  A
    ``rate_curve`` / ``borrow_curve`` (as loaded from ``ir_curve.json`` /
    ``borrow_curve.json``) replaces the flat ``rate`` / ``borrow``.
    """
    calendar = calendar_from_name(value.get("calendar"), calendar_file=calendar_file)
    surface = surface_from_dict(value, calendar=calendar)
    valuation = to_datetime(valuation_date) if valuation_date else surface.init_date
    return MarketState(
        valuation_date=valuation,
        spot=float(spot if spot is not None else surface.init_spot),
        rate_curve=(
            rate_curve
            if rate_curve is not None
            else ConstantRateCurve(float(rate), anchor=valuation)
        ),
        # always present, even at a zero level: rhoQ / bucketed delta need a
        # curve to bump (a zero borrow rate still moves the forward)
        borrow_curve=(
            borrow_curve
            if borrow_curve is not None
            else ConstantRateCurve(float(borrow), anchor=valuation)
        ),
        calendar=calendar,
        trading_days_per_year=surface.trading_days_per_year,
        holiday_weight=surface.holiday_weight,
        surface=surface,
    )
