"""The JSON generator: the quick block becomes a payload ``price-json`` can eat.

The two product-specific fields of the autocall block get the attention here -
``guaranteed_period`` (months with no observation) and ``stepdown_size`` (the
per-observation knock-out step) - plus the coupon step schedule keyed by
observation number, because those three decide what the generated contract
actually pays.
"""

import json
from datetime import datetime

import pytest

from surface_pricer.__main__ import main as launcher_main
from surface_pricer.apps.build_json import QUICK_DEFAULTS
from surface_pricer.apps.build_json import main as build_main
from surface_pricer.apps.price_json import main as price_json_main
from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.fit_runs import record_fit_run
from surface_pricer.pricing.exotics.autocall import AutocallSchedule, accrual

VALUATION = datetime(2026, 1, 5, 15, 0)
SPOT = 100.0

#: A monthly 1Y snowball on the recorded 100-point underlying.
BLOCK = {
    "underlying": "MO",
    "start": None,
    "tenor": "1Y",
    "expiry": None,
    "obs_freq": "M",
    "guaranteed_period": 0,
    "ko": 1.0,
    "stepdown_size": 0.0,
    "ki": 0.75,
    "ki_frequency": "expiry",
    "ki_strike": 1.0,
    "ki_gearing": 1.0,
    "protection": 0.0,
    "settlement_days": 0,
    "coupon": 0.13,
    "rebate": None,
    "day_count": "act/365f",
    "notional": 1.0e6,
    "start_spot": None,
    "no_shift": False,
}

#: The vanilla block (the shipped one may be commented out by a user).
VANILLA = {
    "underlying": None,
    "strike": 7800.0,
    "strike_type": "absolute",
    "tenor": "3M",
    "expiry": None,
    "option_type": "call",
    "notional": 1.0,
}


def _record(tmp_path, calendar_name="TEST"):
    """A recorded fit run; ``SHX`` brings the bundled Chinese holiday list."""
    return record_fit_run(
        tmp_path / "vol_fit" / "MO_20260105_150000",
        surface=EDSSabrSurface(
            init_date=VALUATION,
            init_spot=SPOT,
            expiry_dates=[datetime(2026, 10, 5), datetime(2027, 1, 5)],
            atm_vols=[0.20, 0.21],
            calendar=BusinessCalendar(name=calendar_name),
            trading_days_per_year=252.0,
            holiday_weight=0.0,
        ),
        underlying="MO",
        valuation_datetime=VALUATION,
        spot=SPOT,
        rate=0.02,
        calendar_name=calendar_name,
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


def _market():
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(0.02, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(0.0, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


@pytest.fixture(autouse=True)
def _hermetic_block(monkeypatch):
    """The generator reads the live block - pin the keys a test must not inherit.

    ``QUICK_DEFAULTS`` is the user's editing surface: whichever spot, valuation
    date or output folder it currently holds would otherwise leak into every test.
    """
    for key, value in {
        "fit": "latest",
        "spot": None,
        "valuation_date": None,
        "output_dir": "output",
        "output_name": None,
        "json": False,
    }.items():
        monkeypatch.setitem(QUICK_DEFAULTS, key, value)


@pytest.fixture
def block(tmp_path, monkeypatch):
    """A recorded fit run plus the autocall block the generator reads."""
    _record(tmp_path)
    monkeypatch.setitem(QUICK_DEFAULTS, "autocall", dict(BLOCK))
    return QUICK_DEFAULTS["autocall"]


def _generate(tmp_path, *extra, name="payload.json"):
    target = tmp_path / name
    code = build_main(
        [
            "--fit", "latest",
            "--output-root", str(tmp_path),
            "--ir-curve", "none", "--borrow-curve", "none",
            "--output-name", str(target),
            *extra,
        ]
    )
    return code, target


def _payload(tmp_path, *extra, name="payload.json"):
    code, target = _generate(tmp_path, *extra, name=name)
    assert code == 0
    return json.loads(target.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the basics
def test_the_block_becomes_a_payload_the_pricing_entry_point_eats(capsys, tmp_path, block):
    code, target = _generate(tmp_path)
    output = capsys.readouterr().out

    assert code == 0
    assert target.is_file()
    assert "payload    :" in output
    assert "12 observation(s)" in output

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["kind"] == "autocall_schedule"
    assert payload["spot0"] == pytest.approx(SPOT)
    assert len(payload["observations"]) == 12
    assert payload["observations"][0]["date"] == "2026-02-05"
    assert payload["observations"][-1]["date"] == "2027-01-05"
    assert payload["observations"][0]["coupon_rate"] == pytest.approx(0.13)

    # ... and price-json prices it straight off the file
    assert (
        price_json_main(
            [
                "--fit", "latest", "--output-root", str(tmp_path),
                "--ir-curve", "none", "--borrow-curve", "none",
                "--method", "pde", "--pde-nodes", "101",
                str(target),
            ]
        )
        == 0
    )
    assert "npv" in capsys.readouterr().out


def test_the_generator_can_print_instead_of_writing(capsys, tmp_path, block):
    code, target = _generate(tmp_path, "--output-name", "-")

    assert code == 0
    assert not target.is_file()
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "autocall_schedule"


def test_a_relative_output_lands_in_the_package(capsys, tmp_path, monkeypatch, block):
    from surface_pricer.apps import _common

    monkeypatch.setattr(_common, "PACKAGE_DIR", tmp_path)

    assert (
        build_main(
            [
                "--fit", "latest", "--output-root", str(tmp_path),
                "--ir-curve", "none", "--borrow-curve", "none",
            ]
        )
        == 0
    )
    capsys.readouterr()

    written = tmp_path / "output" / "autocall.json"
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["kind"] == "autocall_schedule"


# ------------------------------------------------------------------ the calendar
def test_month_grid_rolls_every_date_onto_the_calendar():
    """A grid date on a weekend (or a holiday) moves to the next business day."""
    from datetime import date

    from surface_pricer.core.daycount import month_grid

    calendar = BusinessCalendar(name="T", holidays=["2026-04-06"])  # the Monday after Easter
    raw = month_grid("2026-07-05", 1, after="2026-01-01")
    rolled = month_grid("2026-07-05", 1, after="2026-01-01", calendar=calendar)

    assert len(raw) == len(rolled) == 7
    assert all(calendar.is_business_day(day) for day in rolled)
    assert rolled == tuple(sorted({calendar.next_business_day(day) for day in raw}))
    # Sunday 2026-04-05 followed by a holiday Monday: two days forward, and the raw
    # date itself is gone either way
    assert date(2026, 4, 5) in raw and date(2026, 4, 7) in rolled
    assert date(2026, 4, 5) not in rolled and date(2026, 4, 6) not in rolled
    # without a calendar nothing moves
    assert date(2026, 4, 5) in month_grid("2026-07-05", 1, after="2026-01-01")


def test_the_observation_grid_lands_on_business_days_only(capsys, tmp_path, monkeypatch):
    """The SHX calendar: 2027-01-09 is a Saturday, 2027-02-09 the Spring Festival."""
    from surface_pricer.apps.build_json import QUICK_DEFAULTS
    from surface_pricer.io.serialization import calendar_from_name

    _record(tmp_path, calendar_name="SHX")
    monkeypatch.setitem(QUICK_DEFAULTS, "autocall", {**BLOCK, "tenor": "2Y"})
    monkeypatch.setitem(QUICK_DEFAULTS, "output_dir", "output")

    code, target = _generate(tmp_path, name="shx.json")
    output = capsys.readouterr().out

    assert code == 0
    calendar = calendar_from_name("SHX")
    dates = [
        item["date"] for item in json.loads(target.read_text(encoding="utf-8"))["observations"]
    ]

    assert len(dates) >= 20
    assert all(calendar.is_business_day(day) for day in dates)
    assert "2027-01-09" not in dates  # a Saturday
    assert "2027-02-09" not in dates  # Spring Festival
    assert any(day.startswith("2027-02") for day in dates)  # ... its replacement is there
    assert "SHX" in output and "holiday" in output


def test_an_explicit_expiry_on_a_holiday_rolls_forward(capsys, tmp_path, monkeypatch):
    from surface_pricer.apps.build_json import QUICK_DEFAULTS
    from surface_pricer.io.serialization import calendar_from_name

    _record(tmp_path, calendar_name="SHX")
    monkeypatch.setitem(
        QUICK_DEFAULTS,
        "autocall",
        {**BLOCK, "tenor": None, "expiry": "2026-10-01"},  # National Day
    )
    monkeypatch.setitem(QUICK_DEFAULTS, "output_dir", "output")

    code, target = _generate(tmp_path, name="national-day.json")

    assert code == 0
    expiry = json.loads(target.read_text(encoding="utf-8"))["expiry_date"][:10]
    assert expiry != "2026-10-01"
    assert calendar_from_name("SHX").is_business_day(expiry)


# --------------------------------------------------------------- guaranteed period
def test_guaranteed_period_skips_the_observations_it_covers(capsys, tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "guaranteed_period", 3)

    payload = _payload(tmp_path, name="guaranteed.json")

    # the first three months have no observation: the 1st of the 9 left is at the
    # end of month 4
    assert [item["date"] for item in payload["observations"]][:2] == [
        "2026-05-05",
        "2026-06-05",
    ]
    assert len(payload["observations"]) == 9

    # and that first observation accrues the whole lock-up, not one period
    schedule = AutocallSchedule.from_dict(payload, _market())
    covered = accrual(schedule, schedule.observation_dates[0])
    assert covered == pytest.approx(120.0 / 365.0, rel=1e-9)
    assert covered > 3.0 / 12.0  # act/365, and four month-ends after the start


def test_a_guarantee_that_leaves_no_observation_is_refused(capsys, tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "guaranteed_period", 12)

    code, _ = _generate(tmp_path, name="none.json")

    assert code == 2
    assert "no observation left" in capsys.readouterr().out


# ----------------------------------------------------------------- stepdown size
def test_stepdown_size_steps_the_knock_out_at_every_observation(tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "stepdown_size", 0.005)

    payload = _payload(tmp_path, name="stepdown.json")
    levels = [item["ko"] for item in payload["observations"]]

    assert levels[0] == pytest.approx(100.0)
    assert levels[1] == pytest.approx(99.5)
    assert levels[-1] == pytest.approx(100.0 - 11 * 0.5)
    # the term sheet's own step replaces the packaged knock-out rule, so the two
    # can never stack
    assert payload["shift"]["ko"]["mode"] == "none"
    assert [level for level in AutocallSchedule.from_dict(payload, _market()).ko_levels] == (
        pytest.approx(levels)
    )


def test_a_stepdown_that_breaks_the_knock_out_is_refused(capsys, tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "stepdown_size", 0.1)

    code, _ = _generate(tmp_path, name="broken.json")

    assert code == 2
    assert "stepdown_size" in capsys.readouterr().out


# ---------------------------------------------------------------------- coupon
def test_the_coupon_schedule_expands_by_observation_number(tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "coupon", {1: 0.19, 11: 0.1})

    rates = [item["coupon_rate"] for item in _payload(tmp_path, name="coupon.json")["observations"]]

    assert rates[:10] == pytest.approx([0.19] * 10)
    assert rates[10:] == pytest.approx([0.1] * 2)


def test_the_coupon_list_must_cover_every_observation(capsys, tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "coupon", [0.1, 0.2])

    code, _ = _generate(tmp_path, name="short.json")

    assert code == 2
    assert "coupon list has 2 entries" in capsys.readouterr().out


def test_a_coupon_schedule_must_start_at_one(capsys, tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "coupon", {2: 0.1})

    code, _ = _generate(tmp_path, name="late.json")

    assert code == 2
    assert "must start at period 1" in capsys.readouterr().out


def test_a_coupon_schedule_counts_nominal_periods_under_a_guarantee(tmp_path, block, monkeypatch):
    """The lock-up hides observations, so a step written at month 11 stays at month 11."""
    monkeypatch.setitem(block, "guaranteed_period", 2)
    monkeypatch.setitem(block, "coupon", {1: 0.19, 11: 0.1})

    rates = [
        item["coupon_rate"]
        for item in _payload(tmp_path, name="nominal.json")["observations"]
    ]

    # 12 nominal periods, 2 locked up -> 10 observed: nominal 11 is the 9th of them
    assert rates[:8] == pytest.approx([0.19] * 8)
    assert rates[8:] == pytest.approx([0.1] * 2)


def test_a_coupon_schedule_beyond_the_nominal_periods_is_refused(capsys, tmp_path, block, monkeypatch):
    # the numbering is nominal, so 12 periods are written even under a lock-up
    monkeypatch.setitem(block, "guaranteed_period", 2)
    monkeypatch.setitem(block, "coupon", {1: 0.19, 13: 0.1})

    code, _ = _generate(tmp_path, name="beyond.json")

    assert code == 2
    assert "names period 13" in capsys.readouterr().out


# -------------------------------------------------------------------- knock-in
def test_ki_strike_reaches_the_payload(tmp_path, block, monkeypatch):
    monkeypatch.setitem(block, "ki_strike", 0.9)
    monkeypatch.setitem(block, "ki", 0.7)

    knock_in = _payload(tmp_path, name="ki.json")["knock_in"]

    assert knock_in["strike"] == pytest.approx(0.9 * SPOT)
    assert knock_in["level"] == pytest.approx(0.7 * SPOT)
    assert knock_in["frequency"] == "expiry"


# --------------------------------------------------------------------- vanilla
def test_the_coupon_ladder_counts_nominal_periods():
    """A guaranteed lock-up hides observations, not period numbers (2026-10)."""
    from surface_pricer.apps.build_json import _coupon_rates

    # 12 nominal periods of which the first 3 are inside the lock-up -> 9 observed
    assert _coupon_rates(0.10, 9, offset=3) == (0.10,) * 9
    # a step inside the lock-up applies from the first observed period on
    assert _coupon_rates({1: 0.10, 4: 0.20}, 9, offset=3) == (0.20,) * 9
    # a step past it lands where it is written: nominal 7 -> the 4th observation
    assert _coupon_rates({1: 0.10, 7: 0.20}, 9, offset=3) == (0.10,) * 3 + (0.20,) * 6
    # the list spelling counts the full grid too
    assert _coupon_rates([0.10] * 12, 9, offset=3) == (0.10,) * 9

    with pytest.raises(ValueError, match="names period 13"):
        _coupon_rates({1: 0.10, 13: 0.20}, 9, offset=3)
    with pytest.raises(ValueError, match="does not shorten the list"):
        _coupon_rates([0.10] * 9, 9, offset=3)


def test_the_vanilla_block_generates_a_spec(capsys, tmp_path, monkeypatch):
    _record(tmp_path)
    monkeypatch.setitem(
        QUICK_DEFAULTS,
        "vanilla",
        {**VANILLA, "strike": 1.05, "strike_type": "percentage", "option_type": "put", "notional": 10.0},
    )

    code, target = _generate(tmp_path, "--product", "vanilla", name="van.json")

    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["kind"] == "vanilla_spec"
    assert payload["option_type"] == "put"
    assert payload["strike_type"] == "percentage"
    assert payload["strike"] == pytest.approx(1.05 * SPOT)
    assert payload["notional"] == pytest.approx(10.0)
    assert payload["underlying"] == "MO"  # None -> the run's underlying

    assert "vanilla_spec" in capsys.readouterr().out


def test_an_unknown_product_is_refused(capsys):
    with pytest.raises(SystemExit):
        build_main(["--product", "note"])

    assert "invalid choice" in capsys.readouterr().err


# -------------------------------------------------------------------- launcher
def test_the_launcher_dispatches_build_json(capsys, tmp_path, monkeypatch):
    _record(tmp_path)
    # the vanilla block is the user's to comment out; the launcher path needs one
    monkeypatch.setitem(QUICK_DEFAULTS, "vanilla", dict(VANILLA))
    target = tmp_path / "launcher.json"

    assert (
        launcher_main(
            [
                "build-json", "--product", "vanilla",
                "--fit", "latest", "--output-root", str(tmp_path),
                "--ir-curve", "none", "--borrow-curve", "none",
                "--output-name", str(target),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert json.loads(target.read_text(encoding="utf-8"))["kind"] == "vanilla_spec"
