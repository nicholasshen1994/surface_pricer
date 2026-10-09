"""Spot slide: the ladder, one full repricing per rung, and the CLI output.

The load-bearing check is that a rung is *not* an approximation: the row a slide
reports for a spot must equal the same payload priced with ``--spot`` - an
independent route through the CLI - which is what makes the ladder a risk table
rather than an extrapolation.
"""

import json
from datetime import datetime

import pytest

from surface_pricer.__main__ import main as launcher_main
from surface_pricer.apps.price_json import main as price_json_main
from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.fit_runs import record_fit_run
from surface_pricer.pricing.results import RiskSettings
from surface_pricer.pricing.risk.slide import run_slide, spot_ladder
from surface_pricer.pricing.vanilla import (
    VanillaContract,
    VanillaPricer,
    calculate_greeks_spec,
    resolve_spec,
)

VALUATION = datetime(2026, 9, 28, 15, 0)
EXPIRY = datetime(2027, 3, 19, 15, 0)
SPOT = 7500.0
STRIKE = 7600.0
RATE = 0.02
BORROW = 0.01
VOL = 0.22

#: The autocall fixtures sit on their own spot / calendar (a 100-point underlying).
AUTOCALL_VALUATION = datetime(2026, 1, 5, 15, 0)
AUTOCALL_SPOT = 100.0


def _surface(spot=SPOT, valuation=VALUATION, expiry=EXPIRY, vol=VOL, days=243.0, weight=0.05):
    return EDSSabrSurface(
        init_date=valuation,
        init_spot=spot,
        expiry_dates=[expiry],
        atm_vols=[vol],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=days,
        holiday_weight=weight,
    )


def _market(spot=SPOT):
    return MarketState(
        valuation_date=VALUATION,
        spot=spot,
        rate_curve=ConstantRateCurve(RATE, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(BORROW, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
        surface=_surface(),
    )


def _contract(**overrides):
    params = dict(expiry=EXPIRY, strike=STRIKE, option_type="call", underlying="MO")
    params.update(overrides)
    return VanillaContract(**params)


def _record(tmp_path):
    return record_fit_run(
        tmp_path / "vol_fit" / "MO_20260928_150000",
        surface=_surface(),
        underlying="MO",
        valuation_datetime=VALUATION,
        spot=SPOT,
        rate=RATE,
        calendar_name="TEST",
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


def _record_autocall(tmp_path):
    return record_fit_run(
        tmp_path / "vol_fit" / "MO_20260105_150000",
        surface=_surface(
            spot=AUTOCALL_SPOT,
            valuation=AUTOCALL_VALUATION,
            expiry=datetime(2027, 1, 5),
            vol=0.21,
            days=252.0,
            weight=0.0,
        ),
        underlying="MO",
        valuation_datetime=AUTOCALL_VALUATION,
        spot=AUTOCALL_SPOT,
        rate=0.02,
        calendar_name="TEST",
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


def _vanilla_payload(tmp_path):
    """A resolved payload, the shape ``build-json --product vanilla`` writes."""
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(resolve_spec(_contract(), _market()).to_dict()), encoding="utf-8"
    )
    return path


def _snowball_payload(tmp_path):
    """A snowball payload on the 100-point autocall run (hand-written, §13.1)."""
    path = tmp_path / "snowball.json"
    path.write_text(
        json.dumps(
            {
                "kind": "autocall_schedule",
                "underlying": "MO",
                "spot0": AUTOCALL_SPOT,
                "start_date": "2026-01-05T00:00:00",
                "expiry_date": "2026-10-05T00:00:00",
                "notional": 1.0e6,
                "observations": [
                    {"date": day, "ko": 103.0, "coupon_rate": 0.13}
                    for day in ("2026-04-05", "2026-07-05", "2026-10-05")
                ],
                "knock_in": {
                    "frequency": "observation_dates",
                    "strike": 100.0,  # absolute: the loss leg is struck at 100% of spot0
                    "gearing": 1.0,
                    "protected_principal": 0.0,
                    "level": 75.0,
                },
                "rebate": 0.13,  # annual rate, the same unit as coupon_rate
            }
        ),
        encoding="utf-8",
    )
    return path


def _cli(tmp_path, *extra):
    # the flat curve spelling keeps the ladder hermetic (no dependency on the
    # developer's output/ir_curve runs)
    return [
        "--fit",
        "latest",
        "--output-root",
        str(tmp_path),
        "--ir-curve",
        "none",
        "--borrow-curve",
        "none",
        *[str(item) for item in extra],
    ]


# -------------------------------------------------------------------- the ladder
def test_the_ladder_is_symmetric_and_always_holds_the_base():
    levels = spot_ladder(100.0, span=0.30, step=0.05)

    assert len(levels) == 13
    assert levels[0] == pytest.approx(70.0)
    assert levels[-1] == pytest.approx(130.0)
    assert any(level == pytest.approx(100.0) for level in levels)
    assert len(spot_ladder(100.0)) == 13  # the defaults

    # a step that does not divide the span still answers for "no move"
    odd = spot_ladder(100.0, span=0.30, step=0.07)
    assert odd == tuple(sorted(odd))
    assert any(level == pytest.approx(100.0) for level in odd)
    assert odd[-1] == pytest.approx(128.0)


def test_explicit_spots_win_and_bad_ladders_are_refused():
    assert spot_ladder(100.0, span=0.9, spots=[105.0, 95.0, 105.0]) == (95.0, 105.0)

    with pytest.raises(ValueError, match="must be positive"):
        spot_ladder(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        spot_ladder(100.0, spots=[0.0])
    with pytest.raises(ValueError, match="no spot levels"):
        spot_ladder(100.0, spots=[])
    with pytest.raises(ValueError, match="step"):
        spot_ladder(100.0, span=0.3, step=0.0)


# ---------------------------------------------------------------- one rung = one repricing
def test_a_rung_is_a_full_repricing_at_that_spot():
    market = _market()
    spec = resolve_spec(_contract(), market)
    settings = RiskSettings(greeks=("delta", "gamma", "vega"))
    rows = run_slide(
        market,
        greeks=("delta", "gamma", "vega"),
        spots=[6900.0, 7500.0],
        price_at=lambda moved: calculate_greeks_spec(spec.rebased(moved), moved, settings),
    )

    assert [row.spot for row in rows] == [6900.0, 7500.0]
    assert rows[0].bump == pytest.approx(6900.0 / SPOT - 1.0)
    assert rows[1].bump == pytest.approx(0.0)
    # the base rung *is* the ordinary quote
    assert rows[1].npv == pytest.approx(VanillaPricer(market).price_spec(spec).npv, rel=1e-12)
    # the strike is the trade: a rung moves the market, never the contract
    moved = spec.rebased(market.clone(spot=6900.0))
    assert moved.strike == pytest.approx(STRIKE)
    assert moved.spot == pytest.approx(6900.0)
    assert moved.forward < spec.forward


def test_the_greeks_are_read_at_the_rung_not_at_the_base():
    market = _market()
    spec = resolve_spec(_contract(), market)
    settings = RiskSettings(greeks=("delta", "gamma", "vega"))
    rows = run_slide(
        market,
        greeks=("delta", "gamma", "vega"),
        price_at=lambda moved: calculate_greeks_spec(spec.rebased(moved), moved, settings),
    )

    npvs = [row.npv for row in rows]
    deltas = [row.greeks["delta"] for row in rows]
    assert npvs == sorted(npvs)  # a call gains as the spot rises
    assert deltas == sorted(deltas)
    assert all(row.greeks["vega"] > 0.0 for row in rows)

    # gamma peaks at the money: the rung nearest the strike beats both tails
    middle = min(range(len(rows)), key=lambda index: abs(rows[index].spot - STRIKE))
    assert rows[middle].greeks["gamma"] > rows[0].greeks["gamma"]
    assert rows[middle].greeks["gamma"] > rows[-1].greeks["gamma"]


# ---------------------------------------------------------------------- the CLI
def test_a_vanilla_rung_equals_the_same_quote_at_that_spot(tmp_path, capsys):
    _record(tmp_path)
    payload = _vanilla_payload(tmp_path)
    common = ("--json", "--greeks", "delta,gamma")

    assert (
        price_json_main(
            _cli(tmp_path, *common, payload, "--slide", "--slide-spots", "7000,7500")
        )
        == 0
    )
    slide = json.loads(capsys.readouterr().out)
    assert slide["kind"] == "spot_slide"
    assert slide["greeks"] == ["delta", "gamma"]
    assert slide["base_spot"] == pytest.approx(SPOT)
    assert [row["spot"] for row in slide["rows"]] == [7000.0, 7500.0]
    assert slide["contract"]["kind"] == "vanilla_spec"

    for row in slide["rows"]:
        assert (
            price_json_main(
                _cli(tmp_path, *common, "--spot", row["spot"], payload)
            )
            == 0
        )
        quote = json.loads(capsys.readouterr().out)
        assert row["npv"] == pytest.approx(quote["npv"], rel=1e-9)
        assert row["delta"] == pytest.approx(quote["greeks"]["delta"], rel=1e-9)
        assert row["gamma"] == pytest.approx(quote["greeks"]["gamma"], rel=1e-9)


def test_the_ladder_prints_a_table_and_writes_csv(tmp_path, capsys):
    _record(tmp_path)
    payload = _vanilla_payload(tmp_path)
    csv_path = tmp_path / "ladder.csv"

    assert (
        price_json_main(
            _cli(
                tmp_path, "--greeks", "delta,gamma", payload,
                "--slide", "--slide-range", "10%", "--slide-step", "5%", "--csv", csv_path,
            )
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "5 rungs" in output
    assert "span=+-10.00% | step=5.00%" in output
    assert "contract   : vanilla_spec" in output
    assert "convention :" in output
    assert "<- base" in output
    assert "wrote" in output

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "spot,bump,npv,delta,gamma"
    assert len(lines) == 6  # header + five rungs


def test_buckets_are_dropped_and_bad_input_is_refused(tmp_path, capsys):
    _record(tmp_path)
    payload = _vanilla_payload(tmp_path)

    assert (
        price_json_main(
            _cli(
                tmp_path, payload, "--slide", "--slide-spots", "7000,7500",
                "--greeks", "delta,buckets", "--json",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    slide = json.loads(captured.out)
    assert slide["greeks"] == ["delta"]
    assert "bucketed_vega" not in slide["rows"][0]
    assert "dropped" in captured.err

    assert price_json_main(_cli(tmp_path, payload, "--slide", "--slide-spots", "0")) == 2
    assert "must be positive" in capsys.readouterr().out

    assert (
        price_json_main(
            _cli(tmp_path, payload, "--slide", "--slide-spots", "7000", "--json", "--csv", "-")
        )
        == 2
    )
    assert "pick one" in capsys.readouterr().out

    assert price_json_main(_cli(tmp_path, payload, "--slide", "--slide-range", "thirty")) == 2
    assert "ERROR" in capsys.readouterr().out


def test_an_autocall_ladder_matches_the_spot_override(tmp_path, capsys):
    _record_autocall(tmp_path)
    spec_path = _snowball_payload(tmp_path)
    engine = ["--method", "pde", "--pde-nodes", "101"]

    assert (
        price_json_main(
            _cli(
                tmp_path, *engine, "--greeks", "delta", "--json", spec_path,
                "--slide", "--slide-spots", "90,100,110",
            )
        )
        == 0
    )
    slide = json.loads(capsys.readouterr().out)
    assert slide["kind"] == "spot_slide"
    assert slide["contract"]["kind"] == "autocall_schedule"
    assert [row["spot"] for row in slide["rows"]] == [90.0, 100.0, 110.0]

    for row in slide["rows"]:
        assert (
            price_json_main(
                _cli(
                    tmp_path, *engine, "--greeks", "delta", "--json",
                    "--spot", row["spot"], spec_path,
                )
            )
            == 0
        )
        quote = json.loads(capsys.readouterr().out)
        assert row["npv"] == pytest.approx(quote["npv"], rel=1e-9)
        assert row["delta"] == pytest.approx(quote["greeks"]["delta"], rel=1e-9)


def test_the_launcher_dispatches_the_slide(tmp_path, capsys):
    _record(tmp_path)
    payload = _vanilla_payload(tmp_path)

    assert (
        launcher_main(
            [
                "slide", "--fit", "latest", "--output-root", str(tmp_path),
                "--ir-curve", "none", "--borrow-curve", "none",
                "--json", "--greeks", "delta", "--slide-spots", "7500", str(payload),
            ]
        )
        == 0
    )
    slide = json.loads(capsys.readouterr().out)
    assert slide["kind"] == "spot_slide"
    assert slide["rows"][0]["spot"] == pytest.approx(7500.0)
    assert slide["rows"][0]["bump"] == pytest.approx(0.0)
