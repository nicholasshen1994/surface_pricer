"""Hand overrides: config validation, synthetic tenors, hard pinning."""

from datetime import date, datetime

import numpy as np
import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.math.black import black_price
from surface_pricer.fitting.engine import EDSSabrFitter
from surface_pricer.fitting.overrides import (
    OverrideConfig,
    PillarOverride,
    extend_and_apply,
    parse_pin_spec,
    synthetic_tenor_dates,
)
from surface_pricer.fitting.pipeline import fit_surface
from surface_pricer.fitting.settings import FitSettings
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.marketdata.data import OptionQuoteRecord, RawSnapshot

VALUATION = datetime(2026, 9, 28, 14, 55)
NEAR = datetime(2026, 12, 18, 15, 0)
FAR = datetime(2027, 6, 18, 15, 0)
FORWARD = 7500.0
STRIKES = (7000.0, 7300.0, 7500.0, 7700.0, 8000.0)


def _snapshot(expiries=(NEAR, FAR), volatility=0.22):
    calendar = BusinessCalendar(name="TEST")
    rate_curve = ConstantRateCurve(0.0, anchor=VALUATION)
    market = MarketState(
        valuation_date=VALUATION,
        spot=FORWARD,
        rate_curve=rate_curve,
        calendar=calendar,
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )
    records = []
    for expiry in expiries:
        tau = market.year_fraction(expiry)
        for strike in STRIKES:
            for option_type in ("call", "put"):
                price = black_price(FORWARD, strike, tau, volatility, 1.0, option_type)
                records.append(
                    OptionQuoteRecord(
                        underlying="TEST",
                        expiry=expiry,
                        strike=float(strike),
                        option_type=option_type,
                        bid=float(price * 0.998),
                        ask=float(price * 1.002),
                    )
                )
    return RawSnapshot(
        underlying="TEST",
        valuation_datetime=VALUATION,
        spot=FORWARD,
        option_records=records,
        rate_curve=rate_curve,
        calendar=calendar,
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


def _surface():
    return EDSSabrSurface(
        init_date=VALUATION,
        init_spot=FORWARD,
        expiry_dates=[NEAR, FAR],
        atm_vols=[0.22, 0.24],
        skews=[0.5, 0.6],
        convs=[0.4, 0.4],
        left_skews_1=[0.5, 0.5],
        left_skews_2=[0.0, 0.0],
        right_skews_1=[0.3, 0.3],
        right_skews_2=[0.0, 0.0],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


# ------------------------------------------------------------------ config
def test_override_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        OverrideConfig.from_dict({"overrides": [{"expiry": "3Y", "vega": 1.0}]})
    with pytest.raises(ValueError):
        OverrideConfig.from_dict({"overrides": [], "extends": True})
    with pytest.raises(ValueError):
        OverrideConfig.from_dict({"overrides": "3Y"})


def test_override_config_rejects_empty_duplicate_and_bad_rules():
    with pytest.raises(ValueError):
        OverrideConfig(overrides=(PillarOverride(expiry="3Y"),))
    with pytest.raises(ValueError):
        OverrideConfig(
            overrides=(
                PillarOverride(expiry="3Y", atm_vol=0.2),
                PillarOverride(expiry="3Y", skew=0.1),
            )
        )
    with pytest.raises(ValueError):
        OverrideConfig(synthetic_end_tenor="3X")
    with pytest.raises(ValueError):
        OverrideConfig(synthetic_months=(13,))
    with pytest.raises(ValueError):
        OverrideConfig(synthetic_week=0)
    with pytest.raises(ValueError):
        OverrideConfig(scaling_floor=0.0)


def test_override_config_round_trip():
    config = OverrideConfig(
        overrides=(PillarOverride(expiry="2028-06-16", atm_vol=0.27, skew=0.7),),
        extend_synthetic_tenors=True,
    )
    restored = OverrideConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_parse_pin_spec():
    pin = parse_pin_spec("2027-06-18:atm_vol=0.215,skew=0.04")
    assert pin.expiry == "2027-06-18"
    assert pin.atm_vol == pytest.approx(0.215)
    assert pin.smile_values() == {"skew": pytest.approx(0.04)}
    with pytest.raises(ValueError):
        parse_pin_spec("2027-06-18")
    with pytest.raises(ValueError):
        parse_pin_spec("2027-06-18:vega=1")


def test_resolve_matches_existing_and_tenor_pillars():
    config = OverrideConfig(
        overrides=(
            PillarOverride(expiry="2026-12-18", atm_vol=0.23),
            PillarOverride(expiry="18M", skew=0.5),
        )
    )
    resolution = config.resolve(VALUATION, BusinessCalendar(name="TEST"), [NEAR, FAR])
    assert resolution.overrides[0].matched is True
    assert resolution.overrides[1].matched is False
    # 18M from 2026-09-28 is 2028-03-28, a Tuesday, so no calendar roll
    assert resolution.overrides[1].expiry == datetime(2028, 3, 28)
    assert resolution.pinned_dates == (NEAR,)
    assert resolution.extra_pillar_dates == (datetime(2028, 3, 28),)


# ----------------------------------------------------------- synthetic pillars
def test_synthetic_tenor_dates_follow_edslib_rule():
    calendar = BusinessCalendar(name="TEST")
    config = OverrideConfig(extend_synthetic_tenors=True)
    dates = synthetic_tenor_dates(VALUATION, FAR, calendar, config)

    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)
    for value in dates:
        assert value.month in (6, 12)
        assert value.weekday() == 4
        assert 15 <= value.day <= 21
        assert value > FAR.date()
        assert calendar.is_business_day(value)
    assert date(2027, 12, 17) in dates
    assert max(dates).year == 2029  # valuation + 3Y


def test_extend_and_apply_keeps_existing_pillars():
    surface = _surface()
    config = OverrideConfig(
        overrides=(PillarOverride(expiry="2028-06-16", atm_vol=0.27, skew=0.7),),
        extend_synthetic_tenors=True,
    )
    resolution = config.resolve(VALUATION, surface.calendar, surface.expiry_dates)
    extended, applied, new_pillars = extend_and_apply(surface, resolution)

    assert len(applied) == 1
    assert datetime(2028, 6, 16) in new_pillars
    # existing pillars keep the fitted values
    np.testing.assert_allclose(extended.atm_vols[:2], surface.atm_vols)
    np.testing.assert_allclose(extended.skews[:2], surface.skews)
    # the hand values land on the requested pillar, others stay interpolated
    index = extended.pillar_index(datetime(2028, 6, 16))
    assert index is not None
    assert extended.atm_vols[index] == pytest.approx(0.27)
    assert extended.skews[index] == pytest.approx(0.7)
    # extra synthetic tenors beyond the hand pillar are appended too
    assert len(extended.expiry_dates) > len(new_pillars)


def test_extend_and_apply_is_a_noop_without_work():
    surface = _surface()
    resolution = OverrideConfig().resolve(VALUATION, surface.calendar, surface.expiry_dates)
    extended, applied, new_pillars = extend_and_apply(surface, resolution)
    assert extended is surface
    assert applied == []
    assert new_pillars == ()


# ---------------------------------------------------------------- hard pinning
def test_fully_pinned_expiry_skips_optimizer(monkeypatch):
    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(
            overrides=(
                PillarOverride(
                    expiry="2026-12-18",
                    atm_vol=0.25,
                    skew=0.6,
                    conv=0.4,
                    left_skew_1=0.5,
                    left_skew_2=0.1,
                    right_skew_1=0.3,
                    right_skew_2=0.2,
                ),
            )
        ),
    )
    calls = {"count": 0}

    def _boom(self, *args, **kwargs):
        calls["count"] += 1
        raise AssertionError("the optimizer must not run for a pinned expiry")

    monkeypatch.setattr(EDSSabrFitter, "_optimize", _boom)
    result = fit_surface(_snapshot(expiries=(NEAR,)), settings)

    assert calls["count"] == 0
    assert len(result.slices) == 1
    slice_result = result.slices[0]
    assert slice_result.is_override is True
    assert slice_result.override_source == "manual"
    assert slice_result.optimizer_method == "pinned"
    assert result.surface.atm_vols[0] == pytest.approx(0.25)
    assert result.surface.skews[0] == pytest.approx(0.6)
    assert result.surface.convs[0] == pytest.approx(0.4)
    assert result.surface.left_skews_1[0] == pytest.approx(0.5)
    assert result.surface.left_skews_2[0] == pytest.approx(0.1)
    assert result.surface.right_skews_1[0] == pytest.approx(0.3)
    assert result.surface.right_skews_2[0] == pytest.approx(0.2)


def test_partially_pinned_expiry_still_fits_the_rest():
    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(
            overrides=(PillarOverride(expiry="2026-12-18", skew=0.75),)
        ),
    )
    result = fit_surface(_snapshot(expiries=(NEAR,)), settings)

    assert result.slices[0].is_override is True
    assert result.surface.skews[0] == pytest.approx(0.75)
    # the ATM vol still comes from the market, not from the override
    assert result.surface.atm_vols[0] == pytest.approx(0.22, abs=5.0e-3)


def test_pinned_pillar_anchors_its_neighbour():
    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(
            overrides=(
                PillarOverride(
                    expiry="2027-06-18",
                    atm_vol=0.30,
                    skew=0.8,
                    conv=0.5,
                    left_skew_1=0.6,
                    left_skew_2=0.2,
                    right_skew_1=0.4,
                    right_skew_2=0.2,
                ),
            )
        ),
    )
    result = fit_surface(_snapshot(), settings)

    dates = [item.expiry.date().isoformat() for item in result.slices]
    assert dates == ["2026-12-18", "2027-06-18"]
    far_result = result.slices[1]
    assert far_result.is_override is True
    assert far_result.fitted.vol_atmf == pytest.approx(0.30)
    assert far_result.fitted.skew == pytest.approx(0.8 / max(0.3, np.sqrt(far_result.tau)))
    # the unpinned neighbour is still calibrated
    near_result = result.slices[0]
    assert near_result.is_override is False
    assert near_result.rmse < 5.0e-3


def test_pinned_values_are_bounds_checked():
    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(
            overrides=(PillarOverride(expiry="2026-12-18", skew=50.0),)
        ),
    )
    with pytest.raises(ValueError):
        fit_surface(_snapshot(expiries=(NEAR,)), settings)


# --------------------------------------------------------- synthetic extension
def test_synthetic_tenors_are_appended_to_surface_and_slices():
    snapshot = _snapshot()
    plain = fit_surface(snapshot, FitSettings(max_iterations=40))
    extended = fit_surface(
        snapshot,
        FitSettings(
            max_iterations=40,
            override_config=OverrideConfig(extend_synthetic_tenors=True),
        ),
    )

    assert len(extended.surface.expiry_dates) > len(plain.surface.expiry_dates)
    assert len(extended.slices) == len(extended.surface.expiry_dates)
    synthetic = [item for item in extended.slices if item.is_synthetic]
    assert synthetic
    assert len(extended.slices) == len(plain.slices) + len(synthetic)
    last_listed = max(item.expiry for item in plain.slices)
    assert all(item.expiry > last_listed for item in synthetic)
    assert all(item.slice_info.strikes.size == 0 for item in synthetic)
    # listed pillars keep the plain-run values
    np.testing.assert_allclose(extended.surface.atm_vols[: len(plain.slices)], plain.surface.atm_vols)
    # metrics flag the added pillars
    metrics = extended.metrics
    for item in synthetic:
        assert metrics[item.expiry.date().isoformat()]["synthetic"] == 1.0
    assert metrics[plain.slices[0].expiry.date().isoformat()]["synthetic"] == 0.0


def test_metrics_flags_pinned_pillars():
    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(
            overrides=(PillarOverride(expiry="2026-12-18", skew=0.7),)
        ),
    )
    result = fit_surface(_snapshot(expiries=(NEAR,)), settings)
    metrics = result.metrics
    key = NEAR.date().isoformat()
    assert metrics[key]["fixed"] == 1.0
    assert metrics[key]["synthetic"] == 0.0


def test_empty_override_config_matches_default_run():
    snapshot = _snapshot()
    plain = fit_surface(snapshot, FitSettings(max_iterations=40))
    configured = fit_surface(
        snapshot, FitSettings(max_iterations=40, override_config=OverrideConfig())
    )

    assert configured.override_config is not None
    assert len(configured.slices) == len(plain.slices)
    np.testing.assert_array_equal(configured.surface.atm_vols, plain.surface.atm_vols)
    np.testing.assert_array_equal(configured.surface.skews, plain.surface.skews)
    np.testing.assert_array_equal(configured.surface.convs, plain.surface.convs)
    np.testing.assert_array_equal(
        configured.surface.left_skews_1, plain.surface.left_skews_1
    )
    np.testing.assert_array_equal(
        configured.surface.right_skews_2, plain.surface.right_skews_2
    )
    assert (
        configured.surface.expiry_dates.tolist()
        == plain.surface.expiry_dates.tolist()
    )


# ------------------------------------------------------------------- CLI wiring
def test_build_override_config_merges_cli_flags():
    from surface_pricer.fit_surface_snapshot import _build_override_config, _parse_args

    args = _parse_args(
        [
            "--pin",
            "2026-12-18:atm_vol=0.25",
            "--extend-tenors",
            "--synthetic-end-tenor",
            "2Y",
            "--synthetic-months",
            "3,9",
            "--synthetic-week",
            "2",
        ]
    )
    config = _build_override_config(args)

    assert config is not None
    assert len(config.overrides) == 1
    assert config.overrides[0].atm_vol == pytest.approx(0.25)
    assert config.extend_synthetic_tenors is True
    assert config.synthetic_end_tenor == "2Y"
    assert config.synthetic_months == (3, 9)
    assert config.synthetic_week == 2


def test_build_override_config_returns_none_without_requests():
    from surface_pricer.fit_surface_snapshot import _build_override_config, _parse_args

    assert _build_override_config(_parse_args([])) is None
    assert _build_override_config(_parse_args(["--no-extend-tenors"])) is None


def test_build_override_config_rejects_bad_pin():
    from surface_pricer.fit_surface_snapshot import _build_override_config, _parse_args

    with pytest.raises(ValueError):
        _build_override_config(_parse_args(["--pin", "2026-12-18:vega=1"]))


# ---------------------------------------------------------------------- report
def test_report_marks_pinned_pillars():
    from surface_pricer.reporting.fit_report import format_fit_report

    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(
            overrides=(PillarOverride(expiry="2026-12-18", skew=0.7),)
        ),
    )
    report = format_fit_report(fit_surface(_snapshot(expiries=(NEAR,)), settings))

    assert "[MANUAL]" in report
    assert "[SYNTH]" not in report


def test_report_marks_synthetic_pillars_only():
    from surface_pricer.reporting.fit_report import format_fit_report

    settings = FitSettings(
        max_iterations=40,
        override_config=OverrideConfig(extend_synthetic_tenors=True),
    )
    report = format_fit_report(fit_surface(_snapshot(), settings))

    assert "[SYNTH]" in report
    assert "[MANUAL]" not in report


def test_report_has_no_markers_without_overrides():
    from surface_pricer.reporting.fit_report import format_fit_report

    report = format_fit_report(fit_surface(_snapshot(), FitSettings(max_iterations=40)))

    assert "[MANUAL]" not in report
    assert "[SYNTH]" not in report
    assert report.endswith(
        "Parameters are scaled surface values: raw fit params x max(0.3, sqrt(tau))."
    )
