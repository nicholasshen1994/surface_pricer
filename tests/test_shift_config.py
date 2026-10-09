"""Barrier-shift pricing rules: config loading, tier resolution, overrides and
the expansion into per-observation shifted levels."""

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Tuple

import pytest

from surface_pricer.pricing.rules import (
    MODE_ADDITIVE,
    MODE_NONE,
    MODE_RELATIVE,
    BarrierShiftSpec,
    ShiftConfigError,
    clear_shift_config_cache,
    contract_max_gap,
    expand_shift,
    load_shift_config,
    resolve_shift,
)


@dataclass
class _Contract:
    """Minimal duck-typed autocall contract for the resolver."""

    annual_coupon: float = 0.20
    notional: float = 1.0e6
    start_date: Any = date(2026, 1, 1)
    observation_dates: Tuple[Any, ...] = (
        date(2026, 3, 31),
        date(2026, 6, 30),
        date(2026, 9, 30),
        date(2026, 12, 31),
    )
    expiry_date: Any = date(2026, 12, 31)
    underlying: str = "MO"
    product_type: str = "autocallable"


# ------------------------------------------------------------------ the standard
def test_packaged_config_matches_the_edslib_standard():
    config = load_shift_config()

    assert "autocallable" in config.templates
    assert "snowball" in config.templates

    ko = config.defaults["ko_shift"]
    assert ko["rule"] == "coupon_fraction"
    assert ko["fraction"] == pytest.approx(-0.1)  # edslib: -annual_coupon / 10
    assert ko["mode"] == MODE_RELATIVE
    assert ko["stepwise"] is True

    ki = config.defaults["ki_shift"]
    assert ki["rule"] == "underlying_class"
    assert ki["index_like"]["mode"] == MODE_RELATIVE
    assert ki["index_like"]["value"] == pytest.approx(-0.0125)
    assert ki["other"]["mode"] == MODE_ADDITIVE
    assert ki["other"]["value"] == pytest.approx(-0.03)

    # the effective-date block mirrors indexo_barrier_shift_config.json
    assert [block[0].isoformat() for block in config.effective_dates] == ["2026-04-01"]
    tiers = config.effective_dates[0][1]["ko_shift"]["tiers"]
    assert [tier.get("max_gap") for tier in tiers][:2] == [5000000, 10000000]
    assert tiers[-1]["value"] == pytest.approx(-0.025)
    assert config.effective_dates[0][1]["ko_shift"]["floor"] == pytest.approx(-0.05)


def test_effective_date_block_selection():
    config = load_shift_config()

    before, before_label = config.block_for(date(2026, 1, 1))
    assert before_label == "defaults"
    assert before["ko_shift"]["rule"] == "coupon_fraction"

    exactly, exactly_label = config.block_for(date(2026, 4, 1))
    assert exactly_label == "config:2026-04-01"
    assert exactly["ko_shift"]["rule"] == "gap_tiers"
    # the block merges over defaults: the KI side is inherited untouched
    assert exactly["ki_shift"]["rule"] == "underlying_class"

    after, after_label = config.block_for(date(2026, 5, 1))
    assert after_label == "config:2026-04-01"
    assert after["ko_shift"]["rule"] == "gap_tiers"


def test_config_load_errors(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(ShiftConfigError):
        load_shift_config(str(missing))

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ShiftConfigError):
        load_shift_config(str(broken))

    bad_rule = tmp_path / "bad_rule.json"
    bad_rule.write_text(
        json.dumps({"defaults": {"ko_shift": {"rule": "whatever", "value": 1.0}}}),
        encoding="utf-8",
    )
    with pytest.raises(ShiftConfigError):
        load_shift_config(str(bad_rule))

    bad_mode = tmp_path / "bad_mode.json"
    bad_mode.write_text(
        json.dumps(
            {"defaults": {"ko_shift": {"rule": "fixed", "value": -0.01, "mode": "sideways"}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ShiftConfigError):
        load_shift_config(str(bad_mode))

    no_from = tmp_path / "no_from.json"
    no_from.write_text(json.dumps({"effective_dates": [{"ko_shift": {}}]}), encoding="utf-8")
    with pytest.raises(ShiftConfigError):
        load_shift_config(str(no_from))

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {
                "effective_dates": [
                    {"from": "2026-01-01"},
                    {"from": "2026-01-01"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ShiftConfigError):
        load_shift_config(str(duplicate))


def test_config_cache_returns_the_same_object_until_cleared(tmp_path):
    path = tmp_path / "shift.json"
    path.write_text(
        json.dumps({"defaults": {"ko_shift": {"rule": "fixed", "value": -0.01}}}),
        encoding="utf-8",
    )

    first = load_shift_config(str(path))
    assert load_shift_config(str(path)) is first

    clear_shift_config_cache()
    again = load_shift_config(str(path))
    assert again is not first
    assert again.defaults["ko_shift"]["value"] == pytest.approx(-0.01)


# ------------------------------------------------------------------- resolving
def test_coupon_fraction_rule_scales_with_the_coupon():
    config = load_shift_config()

    ko, ki = resolve_shift(_Contract(annual_coupon=0.20), config)
    assert ko.mode == MODE_RELATIVE
    assert ko.value == pytest.approx(-0.02)  # -0.1 * 0.20
    assert ko.stepwise is True
    assert ko.source.startswith("defaults")

    assert ki.mode == MODE_RELATIVE
    assert ki.value == pytest.approx(-0.0125)  # MO is index-like
    assert ki.stepwise is False


def test_gap_tiers_follow_the_coupon_gap():
    config = load_shift_config()
    contract = _Contract(annual_coupon=0.20, notional=1.0e8, start_date=date(2026, 5, 1))

    # 2026-05-01 -> 2026-12-31 is 244 days; gap = coupon * notional * years
    gap = contract_max_gap(contract)
    assert gap == pytest.approx(0.20 * 1.0e8 * 244 / 365.0, rel=1e-3)
    assert 1.0e7 <= gap < 2.0e7

    ko, _ = resolve_shift(contract, config)
    assert ko.value == pytest.approx(-0.015)  # the 20M tier
    assert ko.floor == pytest.approx(-0.05)
    assert ko.source.startswith("config:2026-04-01")
    assert ko.stepwise is True


def test_underlying_class_defaults_and_explicit_override():
    config = load_shift_config()

    _, index_like = resolve_shift(_Contract(underlying="MO"), config)
    _, other = resolve_shift(_Contract(underlying="600519.SH"), config)
    _, forced = resolve_shift(_Contract(underlying="MO"), config, underlying_class="other")

    assert index_like.mode == MODE_RELATIVE and index_like.value == pytest.approx(-0.0125)
    assert other.mode == MODE_ADDITIVE and other.value == pytest.approx(-0.03)
    assert forced.mode == MODE_ADDITIVE and forced.value == pytest.approx(-0.03)


def test_the_class_match_uses_the_bare_code():
    """A snowball on the cash index is fitted from the futures'/ETF's options: the
    venue suffix is not part of the class, so both codes take the index-like KI."""
    config = load_shift_config()

    for ticker in ("000852.SH", "000852", "mo", "MO.CFE", "000905", "510500.SH"):
        _, ki = resolve_shift(_Contract(underlying=ticker), config)
        assert ki.mode == MODE_RELATIVE, ticker
        assert ki.value == pytest.approx(-0.0125), ticker

    # an equity code keeps the "other" branch, suffix or not
    for ticker in ("600519.SH", "600519", ""):
        _, ki = resolve_shift(_Contract(underlying=ticker), config)
        assert ki.mode == MODE_ADDITIVE, ticker


def test_override_priority_cli_wins_over_contract_and_config():
    config = load_shift_config()
    contract = _Contract(annual_coupon=0.20)

    # a plain number replaces the value and keeps the resolved rule
    ko, _ = resolve_shift(contract, config, override={"ko": -0.01})
    assert ko.value == pytest.approx(-0.01)
    assert ko.mode == MODE_RELATIVE and ko.stepwise is True
    assert ko.source == "contract-override"

    # a mapping may replace rule fields too
    ko_cli, ki_cli = resolve_shift(
        contract,
        config,
        override={"ko": -0.01},
        cli={"ko": {"value": -0.02, "stepwise": False}, "ki": -0.0},
    )
    assert ko_cli.value == pytest.approx(-0.02)
    assert ko_cli.stepwise is False
    assert ko_cli.source == "cli"
    assert ki_cli.source == "cli"  # an explicit zero disables the side
    assert ki_cli.active is False

    # an inactive side stays untouched when the override omits it
    _, ki = resolve_shift(contract, config, override={"ko": -0.01})
    assert ki.value == pytest.approx(-0.0125)


def test_shift_is_disabled_for_other_product_types():
    config = load_shift_config()
    ko, ki = resolve_shift(_Contract(product_type="vanilla"), config)
    assert ko.mode == MODE_NONE and ki.mode == MODE_NONE
    assert ko.active is False and ki.active is False


def test_contract_max_gap_needs_dates():
    assert contract_max_gap(_Contract()) > 0.0
    assert contract_max_gap(_Contract(start_date=None)) == 0.0
    assert contract_max_gap(_Contract(annual_coupon=0.0)) == 0.0


# ------------------------------------------------------------------- expansion
def test_expand_shift_relative_and_additive_agree_at_unit_level():
    levels = (1.0, 1.0)

    relative = expand_shift(
        BarrierShiftSpec(mode=MODE_RELATIVE, value=-0.0125, stepwise=False), levels
    )
    additive = expand_shift(
        BarrierShiftSpec(mode=MODE_ADDITIVE, value=-0.0125, stepwise=False), levels
    )
    assert relative == pytest.approx((0.9875, 0.9875))
    assert additive == pytest.approx(relative)

    # ... and diverge once the original level is not exactly 1
    off_unit = expand_shift(
        BarrierShiftSpec(mode=MODE_ADDITIVE, value=-0.0125, stepwise=False), (0.80, 0.90)
    )
    assert off_unit == pytest.approx((0.7875, 0.8875))


def test_expand_shift_is_a_noop_when_inactive():
    assert expand_shift(BarrierShiftSpec(), (0.8, 0.9)) == (0.8, 0.9)
    assert expand_shift(BarrierShiftSpec(mode=MODE_RELATIVE, value=0.0), (0.8, 0.9)) == (
        0.8,
        0.9,
    )


def test_expand_shift_stepwise_accrues_to_the_full_value():
    spec = BarrierShiftSpec(mode=MODE_RELATIVE, value=-0.04, stepwise=True)
    expanded = expand_shift(spec, (1.0, 1.0, 1.0, 1.0))
    assert expanded == pytest.approx((0.99, 0.98, 0.97, 0.96))

    # step_offset leaves the leading observation(s) unshifted, the rest still
    # accrues to the same final value
    offset = BarrierShiftSpec(mode=MODE_RELATIVE, value=-0.04, stepwise=True, step_offset=1)
    assert expand_shift(offset, (1.0, 1.0, 1.0, 1.0)) == pytest.approx(
        (1.0, 1.0 - 0.04 / 3, 1.0 - 0.08 / 3, 1.0 - 0.04)
    )

    # a flat (non-stepwise) shift is the same amount every observation
    flat = BarrierShiftSpec(mode=MODE_ADDITIVE, value=-0.03, stepwise=False)
    assert expand_shift(flat, (1.0, 1.0, 1.0, 1.0)) == pytest.approx((0.97, 0.97, 0.97, 0.97))


def test_expand_shift_clamps_the_accumulated_shift():
    spec = BarrierShiftSpec(mode=MODE_ADDITIVE, value=-0.12, stepwise=True, floor=-0.05)
    expanded = expand_shift(spec, (1.0, 1.0, 1.0, 1.0))
    assert expanded == pytest.approx((0.97, 0.95, 0.95, 0.95))

    capped = BarrierShiftSpec(mode=MODE_ADDITIVE, value=0.12, stepwise=True, cap=0.05)
    assert expand_shift(capped, (1.0, 1.0, 1.0, 1.0)) == pytest.approx((1.03, 1.05, 1.05, 1.05))


def test_expand_shift_keeps_past_observations_but_anchors_the_schedule():
    """Past observations keep the original level while the accrual schedule
    stays anchored to the contract start (edslib's merge semantics)."""
    spec = BarrierShiftSpec(mode=MODE_ADDITIVE, value=-0.12, stepwise=True, floor=-0.05)
    expanded = expand_shift(spec, (1.0, 1.0, 1.0, 1.0), start_index=2)

    # first two observations untouched; the third carries the *third* accrual
    # (-0.09 -> clamped to -0.05), not a restarted one
    assert expanded == pytest.approx((1.0, 1.0, 0.95, 0.95))


def test_expand_shift_works_on_barrier_ratio_levels():
    """Levels are ratios of the initial spot (0.85 = 85%), like edslib."""
    levels = (0.85, 0.85, 0.85)
    spec = BarrierShiftSpec(mode=MODE_RELATIVE, value=-0.05, stepwise=True)
    expanded = expand_shift(spec, levels)
    assert expanded[0] == pytest.approx(0.85 * (1 - 0.05 / 3))
    assert expanded[-1] == pytest.approx(0.85 * 0.95)
