"""Tests for the rate / borrow curve build.

``build-ir-curve`` needs QuantLib, which the pricing runtime does not: those
tests are skipped when QL is absent (``pytest.importorskip``).  The borrow and
OU parts are plain numpy and always run.
"""

import math
from datetime import date, timedelta

import pytest
from scipy.integrate import quad

from surface_pricer.core.borrow_curve import (
    build_borrow_curve,
    extend_borrow_tail,
    instantaneous_f0,
    integrate_ou_forward,
    next_quarter_expiry,
    zero_rate_at,
)
from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import DateHelperBusinessCalendar, add_tenor, year_fraction
from surface_pricer.marketdata.rate_inputs import (
    SAMPLE_CSV,
    RateInputs,
    parse_interest_rate_csv,
)

VALUATION = date(2026, 9, 28)
RATE = 0.0142
BORROW = 0.015
SPOT = 7000.0


# ------------------------------------------------------------------ rate input
def test_parse_interest_rate_csv_keeps_latest_and_drops_broken_columns():
    inputs = parse_interest_rate_csv(SAMPLE_CSV)

    assert inputs.valuation_date == VALUATION
    assert inputs.fr007 == pytest.approx(0.0142)
    assert inputs.ir_swap["1M"] == pytest.approx(0.01435)
    assert inputs.ir_swap["10Y"] == pytest.approx(0.016025)
    # constant fixtures and empty columns are reported, not priced
    for label in ("2M", "6Y"):
        assert label in inputs.dropped
        assert label not in inputs.ir_swap
    assert set(inputs.dropped) >= {"2M", "6Y", "20Y", "30Y", "7D", "14D"}
    assert inputs.tenors == [
        "1M",
        "3M",
        "6M",
        "9M",
        "1Y",
        "2Y",
        "3Y",
        "4Y",
        "5Y",
        "7Y",
        "10Y",
    ]


def test_rate_inputs_json_roundtrip(tmp_path):
    inputs = parse_interest_rate_csv(SAMPLE_CSV)
    path = inputs.to_json(tmp_path / "rates.json")

    again = RateInputs.from_json(path)

    assert again.to_dict() == inputs.to_dict()


# ----------------------------------------------------------------- borrow curve
def _forwards_from_borrow(borrow: float = BORROW):
    forwards = {}
    for days in (31, 92, 182, 274):
        forwards[VALUATION + timedelta(days=days)] = SPOT * math.exp(
            (RATE - borrow) * days / 365.0
        )
    return forwards


def test_borrow_curve_recovers_the_input_borrow():
    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    forwards = _forwards_from_borrow()

    pillars = build_borrow_curve(
        VALUATION,
        SPOT,
        forwards,
        curve,
        forward_source={expiry: "future" for expiry in forwards},
    )

    assert pillars.observed == 4
    assert pillars.rates == pytest.approx([BORROW] * 4, abs=1e-12)
    assert pillars.forwards == {
        expiry.isoformat(): pytest.approx(price) for expiry, price in forwards.items()
    }
    assert set(pillars.forward_source.values()) == {"future"}


def test_borrow_curve_skips_expiries_too_close_and_non_positive_forwards():
    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    forwards = _forwards_from_borrow()
    forwards[VALUATION + timedelta(days=2)] = SPOT * 0.99
    forwards[VALUATION + timedelta(days=60)] = 0.0

    pillars = build_borrow_curve(VALUATION, SPOT, forwards, curve)

    assert pillars.observed == 4
    assert any("skipped" in note for note in pillars.notes)


def test_borrow_curve_round_trip_json(tmp_path):
    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    pillars = build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve)
    path = pillars.to_json(tmp_path / "borrow.json")

    from surface_pricer.core.borrow_curve import BorrowCurvePillars

    assert BorrowCurvePillars.from_json(path).to_dict() == pillars.to_dict()


def test_extended_borrow_curve_round_trips_through_json(tmp_path):
    """The OU block mixes numbers (kappa/mu/f0) and text (anchor_date)."""
    from surface_pricer.core.borrow_curve import BorrowCurvePillars

    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    extended = extend_borrow_tail(
        build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve),
        extension_years=3,
        calendar=DateHelperBusinessCalendar("SHX"),
    )
    assert isinstance(extended.ou.get("anchor_date"), str)

    again = BorrowCurvePillars.from_json(extended.to_json(tmp_path / "borrow.json"))

    assert again.to_dict() == extended.to_dict()
    assert again.ou["anchor_date"] == extended.ou["anchor_date"]
    assert again.rates[-1] == pytest.approx(extended.rates[-1])


# ---------------------------------------------------------------------- OU tail
def test_integrate_ou_forward_matches_quadrature():
    f0, kappa, mu, horizon = 0.02, 1.7, 0.005, 2.3

    value = integrate_ou_forward(f0, kappa, mu, horizon)
    expected, _ = quad(lambda s: mu + (f0 - mu) * math.exp(-kappa * s), 0.0, horizon)

    assert value == pytest.approx(expected, rel=1e-10)


def test_instantaneous_f0_and_zero_rate_at_are_consistent():
    T_anchor, q_anchor, T_prev, q_prev = 0.75, 0.11, 0.5, 0.10

    f0 = instantaneous_f0(T_anchor, q_anchor, T_prev, q_prev)
    assert f0 == pytest.approx(0.13)

    # a flat OU (mu == f0 == q_anchor) leaves the zero rate unchanged
    assert zero_rate_at(q_anchor, T_anchor, q_anchor, 1.0, q_anchor, 1.5) == pytest.approx(
        q_anchor
    )
    # non-degenerate case, checked against the integral it is built from
    expected = (q_anchor * T_anchor + integrate_ou_forward(f0, 1.0, 0.13, 0.75)) / 1.5
    assert zero_rate_at(q_anchor, T_anchor, f0, 1.0, 0.13, 1.5) == pytest.approx(expected)


def test_next_quarter_expiry_is_third_friday():
    for base in (date(2026, 9, 28), date(2026, 12, 18), date(2027, 1, 1)):
        expiry = next_quarter_expiry(base)

        assert expiry > base
        assert expiry.month in (3, 6, 9, 12)
        assert expiry.weekday() == 4  # Friday
        assert 15 <= expiry.day <= 21


def test_extend_borrow_tail_adds_quarterly_pillars_up_to_horizon():
    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    pillars = build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve)

    extended = extend_borrow_tail(
        pillars,
        extension_years=3,
        calendar=DateHelperBusinessCalendar("SHX"),
    )

    assert extended.observed == 4
    assert extended.extended >= 8
    tail_dates = extended.pillar_dates[extended.observed :]
    assert tail_dates == sorted(tail_dates)
    assert all(value.month in (3, 6, 9, 12) for value in tail_dates)
    assert tail_dates[-1] <= add_tenor(VALUATION, "3Y")
    # f0 == mu == q on this synthetic input, so the OU tail is flat
    assert extended.rates[extended.observed :] == pytest.approx(
        [BORROW] * extended.extended, abs=1e-9
    )
    assert extended.ou["kappa"] == pytest.approx(1.0)
    assert extended.ou["calibrated"] == 0.0
    assert any("prior" in note for note in extended.notes)


# ----------------------------------------------------------------- IR bootstrap
def test_ir_curve_bootstrap_reproduces_the_quoted_par_rates(tmp_path):
    pytest.importorskip("QuantLib")
    from surface_pricer.core.ir_curve import IRCurvePillars, build_fr007_curve

    inputs = parse_interest_rate_csv(SAMPLE_CSV)
    pillars = build_fr007_curve(
        inputs.valuation_date,
        inputs.ir_swap,
        fr007=inputs.fr007,
        source=str(SAMPLE_CSV),
    )

    assert pillars.curve_name == "CNY-FR007"
    assert pillars.tenors == inputs.tenors
    assert pillars.max_par_residual() < 1e-9
    assert pillars.pillar_days == sorted(pillars.pillar_days)

    curve = pillars.to_piecewise_curve()
    discounts = [
        curve.discount_factor(pillars.valuation_date, expiry)
        for expiry in pillars.pillar_dates
    ]
    assert discounts == sorted(discounts, reverse=True)
    # ~1.6% over 10Y on ACT/360 gives a discount factor near 0.85
    assert 0.8 < discounts[-1] < 1.0

    json_path = pillars.to_json(tmp_path / "ir_curve.json")
    assert IRCurvePillars.from_json(json_path).to_dict() == pillars.to_dict()


# ------------------------------------------------------- dividends and loading
def test_cum_div_factors_turn_the_result_into_pure_borrow():
    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    forwards = _forwards_from_borrow()
    factors = {expiry: 0.99 for expiry in forwards}

    pure = build_borrow_curve(VALUATION, SPOT, forwards, curve, cum_div_factors=factors)

    for expiry, rate in zip(pure.pillar_dates, pure.rates):
        dcf = year_fraction(VALUATION, expiry)
        assert rate == pytest.approx(BORROW + math.log(0.99) / dcf, abs=1e-12)
    assert set(pure.cum_div_factors) == {expiry.isoformat() for expiry in forwards}
    assert any("pure borrow" in note for note in pure.notes)

    # and the forward side can rebuild the future: F = S * exp((r - q_pure) * t) * D
    for expiry, forward in forwards.items():
        dcf = year_fraction(VALUATION, expiry)
        rate = pure.rates[pure.pillar_dates.index(expiry)]
        rebuilt = SPOT * math.exp((RATE - rate) * dcf) * factors[expiry]
        assert rebuilt == pytest.approx(forward, rel=1e-12)


def test_curve_files_load_and_reject_wrong_payloads(tmp_path):
    from surface_pricer.core.ir_curve import IRCurvePillars
    from surface_pricer.io.curve_files import (
        curve_valuation_date,
        load_borrow_curve,
        load_ir_curve,
    )

    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    borrow_path = (
        build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve)
        .to_json(tmp_path / "borrow_curve.json")
    )
    loaded = load_borrow_curve(borrow_path)
    assert curve_valuation_date(loaded) == VALUATION
    assert loaded.zero_rate(VALUATION + timedelta(days=31)) == pytest.approx(BORROW, abs=1e-9)

    ir_path = IRCurvePillars(
        valuation_date=VALUATION,
        tenors=["1M", "1Y"],
        pillar_dates=[VALUATION + timedelta(days=31), VALUATION + timedelta(days=366)],
        pillar_days=[31, 366],
        zero_rates=[0.014, 0.015],
    ).to_json(tmp_path / "ir_curve.json")
    ir_curve = load_ir_curve(ir_path)
    assert ir_curve.zero_rate(VALUATION + timedelta(days=31)) == pytest.approx(0.014, abs=1e-12)

    # cross-loading and missing files are rejected with a clear message
    with pytest.raises(ValueError):
        load_ir_curve(borrow_path)
    with pytest.raises(ValueError):
        load_borrow_curve(ir_path)
    with pytest.raises(ValueError):
        load_borrow_curve(tmp_path / "nope.json")


def test_the_flat_rate_stands_for_the_curve():
    """A run's recorded flat rate follows its curve (3M), not the program default."""
    from surface_pricer.apps.fit_surface import flat_rate
    from surface_pricer.core.curves import ConstantRateCurve, PiecewiseRateCurve

    # no curve -> exactly the --rate that was passed
    assert flat_rate(None, VALUATION, fallback=0.015) == pytest.approx(0.015)

    flat = ConstantRateCurve(0.0205, anchor=VALUATION)
    assert flat_rate(flat, VALUATION, fallback=0.015) == pytest.approx(0.0205)

    # 3M is a pillar of this curve, and that pillar is what gets recorded
    curve = PiecewiseRateCurve(
        anchor=VALUATION, tenors=[1, 90, 365], rates=[0.0100, 0.0200, 0.0250]
    )
    assert flat_rate(curve, VALUATION, fallback=0.015) == pytest.approx(0.0200)


def test_borrow_runs_are_filed_in_the_index_folder(tmp_path):
    """One folder per index: ``latest`` is the newest run **of that folder**."""
    import json

    from surface_pricer.io import curve_runs

    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    pillars = build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve)

    first = curve_runs.record_curve_run(
        "borrow_curve", pillars, output_root=tmp_path, stamp="20261009_090000", index="000852.SH"
    )
    other = curve_runs.record_curve_run(
        "borrow_curve", pillars, output_root=tmp_path, stamp="20261009_091000", index="510500"
    )
    newer = curve_runs.record_curve_run(
        "borrow_curve", pillars, output_root=tmp_path, stamp="20261009_100000", index="000852"
    )

    # the **folder** says which index it is - the file name does not have to
    assert first == (
        tmp_path / "borrow_curve" / "000852" / "borrow_curve_20261009_090000.json"
    )
    assert other.parent == tmp_path / "borrow_curve" / "510500"
    assert first.is_file()  # the superseded run stays on disk

    pointer = json.loads((first.parent / "latest.json").read_text(encoding="utf-8"))
    assert pointer["borrow_curve"] == newer.name
    assert curve_runs.latest_curve_path("borrow_curve", tmp_path, index="000852.SH") == newer
    assert curve_runs.latest_curve_path("borrow_curve", tmp_path, index="510500.SH") == other
    assert curve_runs.list_curve_runs("borrow_curve", tmp_path, index="000852") == [
        newer,
        first,
    ]

    # an index with no folder is an error naming the ones there are - never a neighbour's
    with pytest.raises(ValueError, match="159915"):
        curve_runs.resolve_curve_path(
            "borrow_curve", "latest", output_root=tmp_path, index="159915.SZ"
        )
    with pytest.raises(ValueError, match="000852, 510500"):
        curve_runs.latest_curve_path("borrow_curve", tmp_path, index="159915")

    # a per-index kind refuses a run that does not say whose it is
    with pytest.raises(ValueError, match="filed per index"):
        curve_runs.record_curve_run("borrow_curve", pillars, output_root=tmp_path)

    # the rate kind is one curve for everything: flat folder, index ignored
    ir = curve_runs.record_curve_run(
        "ir_curve", pillars, output_root=tmp_path, stamp="20261009_090000"
    )
    assert ir.parent == tmp_path / "ir_curve"
    assert curve_runs.latest_curve_path("ir_curve", tmp_path, index="510500") == ir
    assert curve_runs.resolve_curve_path(
        "ir_curve", "latest", output_root=tmp_path, index="510500"
    ) == ir


def test_a_flat_borrow_pointer_is_still_read_but_loses_to_the_folder(tmp_path):
    """Runs from before the index folders: read flat, then superseded by the folder."""
    import json

    from surface_pricer.io import curve_runs

    flat = curve_runs.curve_root("borrow_curve", tmp_path)
    flat.mkdir(parents=True, exist_ok=True)
    legacy = flat / "borrow_curve_20261008_101542.json"
    legacy.write_text("{}", encoding="utf-8")
    (flat / "latest.json").write_text(
        json.dumps({"borrow_curve": legacy.name}), encoding="utf-8"
    )

    # a single name is "the one borrow every index used", so any index resolves it
    for index in ("000852", "510500.SH"):
        assert (
            curve_runs.resolve_curve_path(
                "borrow_curve", "latest", output_root=tmp_path, index=index
            )
            == legacy
        )

    # ... the short-lived per-index **map** shape is read too
    other = flat / "borrow_curve_20261009_080000.json"
    other.write_text("{}", encoding="utf-8")
    (flat / "latest.json").write_text(
        json.dumps({"borrow_curve": {"000852": legacy.name, "510500": other.name}}),
        encoding="utf-8",
    )
    assert (
        curve_runs.latest_curve_path("borrow_curve", tmp_path, index="510500.SH") == other
    )

    # ... and once an index has its own folder, that folder wins over both
    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    pillars = build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve)
    fresh = curve_runs.record_curve_run(
        "borrow_curve", pillars, output_root=tmp_path, stamp="20261009_120000", index="000852"
    )
    assert curve_runs.latest_curve_path("borrow_curve", tmp_path, index="000852") == fresh
    # the index without a folder still reads the flat pointer
    assert curve_runs.latest_curve_path("borrow_curve", tmp_path, index="510500") == other


def test_curve_runs_are_stamped_and_latest_follows_the_newest(tmp_path):
    """Every build writes a **new** run; ``latest`` follows the pointer, nothing is overwritten."""
    from surface_pricer.io import curve_runs
    from surface_pricer.io.curve_files import curve_valuation_date, load_borrow_curve

    curve = ConstantRateCurve(RATE, anchor=VALUATION)
    pillars = build_borrow_curve(VALUATION, SPOT, _forwards_from_borrow(), curve)

    first = curve_runs.record_curve_run(
        "borrow_curve", pillars, output_root=tmp_path, stamp="20261008_101541", index="000852"
    )
    second = curve_runs.record_curve_run(
        "borrow_curve", pillars, output_root=tmp_path, stamp="20261008_120000", index="000852"
    )

    assert first.name == "borrow_curve_20261008_101541.json"
    assert second.name == "borrow_curve_20261008_120000.json"
    assert first.is_file() and second.is_file()  # the older run survives the newer one
    assert (
        curve_runs.latest_curve_path("borrow_curve", tmp_path, index="000852") == second
    )
    assert (
        curve_runs.resolve_curve_path(
            "borrow_curve", "latest", output_root=tmp_path, index="000852"
        )
        == second
    )
    assert curve_runs.resolve_curve_path("borrow_curve", "none", output_root=tmp_path) is None
    assert curve_runs.resolve_curve_path("borrow_curve", None, output_root=tmp_path) is None

    # the index lists both, newest first, and a run still loads as a pricing curve
    index = curve_runs.read_curve_index("borrow_curve", tmp_path, index="000852")
    assert [item["file"] for item in index] == [second.name, first.name]
    loaded = load_borrow_curve(second)
    assert curve_valuation_date(loaded) == VALUATION
    assert loaded.zero_rate(VALUATION + timedelta(days=31)) == pytest.approx(BORROW, abs=1e-9)

    # a path is taken as given: a miss names it, it does not go looking
    with pytest.raises(ValueError, match="not found"):
        curve_runs.resolve_curve_path("borrow_curve", str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="not found"):
        load_borrow_curve(tmp_path / "nope.json")


def test_latest_without_a_run_is_an_error(tmp_path):
    """``latest`` with an empty folder fails loudly - the flat rate is ``none``."""
    from surface_pricer.io import curve_runs

    with pytest.raises(ValueError, match="build-ir-curve"):
        curve_runs.resolve_curve_path("ir_curve", "latest", output_root=tmp_path)


def test_market_from_surface_accepts_curve_objects():
    from surface_pricer.fitting.surface import EDSSabrSurface
    from surface_pricer.io.serialization import market_from_surface

    surface = EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=[VALUATION + timedelta(days=90)],
        atm_vols=[0.2],
    )
    rate_curve = ConstantRateCurve(0.02, anchor=VALUATION)
    borrow_curve = ConstantRateCurve(0.05, anchor=VALUATION)

    market = market_from_surface(
        surface.to_dict(),
        spot=SPOT,
        rate=0.0,
        borrow=0.0,
        rate_curve=rate_curve,
        borrow_curve=borrow_curve,
    )

    assert market.rate_curve is rate_curve
    assert market.borrow_curve is borrow_curve
    assert market.forward(VALUATION + timedelta(days=90)) == pytest.approx(
        SPOT * math.exp((0.02 - 0.05) * 90 / 365.0), rel=1e-9
    )
