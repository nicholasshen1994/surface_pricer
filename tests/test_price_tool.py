"""Fit-run storage (output/*.json) and the top-level price tool."""

import json
from datetime import datetime, timedelta

import pytest

from surface_pricer.__main__ import main as launcher_main
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.fit_runs import (
    LATEST_NAME,
    MANIFEST_NAME,
    RUN_INDEX_NAME,
    SURFACE_NAME,
    list_runs,
    record_fit_run,
    resolve_run,
)
from surface_pricer.price_tool import main as price_tool_main

VALUATION = datetime(2026, 9, 28, 15, 0)
EXPIRY_1 = datetime(2026, 12, 18, 15, 0)
EXPIRY_2 = datetime(2027, 3, 19, 15, 0)
SPOT = 7500.0


def _surface():
    return EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=[EXPIRY_1, EXPIRY_2],
        atm_vols=[0.22, 0.24],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )


def _record(tmp_path, name="MO_20260928_150000", **extra):
    return record_fit_run(
        tmp_path / name,
        surface=_surface(),
        underlying="MO",
        valuation_datetime=VALUATION,
        spot=SPOT,
        rate=0.015,
        calendar_name="TEST",
        trading_days_per_year=243.0,
        holiday_weight=0.05,
        **extra,
    )


# ------------------------------------------------------------------ fit runs
def test_record_fit_run_writes_surface_manifest_index_and_latest(tmp_path):
    run = _record(tmp_path)

    assert run.surface_path.name == SURFACE_NAME
    assert run.surface_path.is_file()
    manifest = json.loads((run.directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["underlying"] == "MO"
    assert manifest["n_expiries"] == 2
    assert manifest["expiries"] == ["2026-12-18", "2027-03-19"]
    assert manifest["spot"] == pytest.approx(SPOT)
    assert manifest["rate"] == pytest.approx(0.015)
    assert SURFACE_NAME in manifest["files"] and MANIFEST_NAME in manifest["files"]

    index = json.loads((tmp_path / RUN_INDEX_NAME).read_text(encoding="utf-8"))
    assert index["runs"][0]["name"] == run.name
    latest = json.loads((tmp_path / LATEST_NAME).read_text(encoding="utf-8"))
    assert latest["run"] == run.name


def test_resolve_run_by_latest_name_prefix_and_path(tmp_path):
    first = _record(tmp_path, name="MO_20260928_150000")
    second = _record(tmp_path, name="IO_20260929_090000")

    assert resolve_run("latest", output_root=tmp_path).name == second.name
    assert resolve_run(first.name, output_root=tmp_path).name == first.name
    assert resolve_run("io_2026", output_root=tmp_path).name == second.name
    assert resolve_run(str(first.directory), output_root=tmp_path).name == first.name
    assert resolve_run(str(second.surface_path), output_root=tmp_path).name == second.name

    with pytest.raises(ValueError):
        resolve_run("nope", output_root=tmp_path)


def test_list_runs_orders_newest_first(tmp_path):
    first = _record(tmp_path, name="MO_20260928_150000")
    second = _record(tmp_path, name="IO_20260929_090000")

    names = [run.name for run in list_runs(tmp_path)]
    assert names == [second.name, first.name]


# ---------------------------------------------------------------- price tool
def test_price_tool_quotes_npv_and_greeks(tmp_path, capsys):
    _record(tmp_path)

    code = price_tool_main(
        [
            "--fit",
            "latest",
            "--output-root",
            str(tmp_path),
            "--strike",
            "7500",
            "--tenor",
            "3M",
            "--call",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "npv" in out
    for label in ("delta", "gamma", "vega", "theta", "vanna", "volga", "rho", "rhoQ"):
        assert label in out
    assert "bucketed" in out
    assert "MO_20260928_150000" in out


def test_price_tool_json_output(tmp_path, capsys):
    _record(tmp_path)

    code = price_tool_main(
        [
            "--fit",
            "latest",
            "--output-root",
            str(tmp_path),
            "--strike",
            "7500",
            "--tenor",
            "3M",
            "--put",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"]["option_type"] == "put"
    assert payload["contract"]["expiry"] == "2026-12-28"
    assert payload["npv"] > 0.0
    for name in ("delta", "delta_cash", "delta_n", "gamma", "vega", "theta", "vanna", "volga", "rho", "rhoq"):
        assert name in payload["greeks"]
    assert payload["bucketed"]["bucketed_vega"]
    assert payload["bucketed"]["bucketed_delta"]
    assert payload["bucketed"]["bucketed_rhoq"]


def test_price_tool_percentage_strike_and_notional(tmp_path, capsys):
    _record(tmp_path)

    code = price_tool_main(
        [
            "--fit",
            "latest",
            "--output-root",
            str(tmp_path),
            "--strike",
            "100",
            "--strike-type",
            "percentage",
            "--tenor",
            "6M",
            "--call",
            "--notional",
            "1000000",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"]["strike_type"] == "percentage"
    assert payload["npv"] > 0.0


def test_price_tool_requires_strike_and_tenor(tmp_path, capsys):
    _record(tmp_path)
    assert price_tool_main(["--output-root", str(tmp_path), "--fit", "latest"]) == 2
    assert "--strike is required" in capsys.readouterr().out

    assert (
        price_tool_main(
            ["--output-root", str(tmp_path), "--fit", "latest", "--strike", "7500"]
        )
        == 2
    )
    assert "--tenor or --expiry is required" in capsys.readouterr().out


def test_price_tool_rejects_unknown_run_and_past_expiry(tmp_path, capsys):
    _record(tmp_path)
    assert (
        price_tool_main(
            [
                "--fit",
                "nope",
                "--output-root",
                str(tmp_path),
                "--strike",
                "7500",
                "--tenor",
                "3M",
            ]
        )
        == 2
    )
    assert "unknown fit run" in capsys.readouterr().out

    assert (
        price_tool_main(
            [
                "--fit",
                "latest",
                "--output-root",
                str(tmp_path),
                "--strike",
                "7500",
                "--expiry",
                "2026-09-01",
            ]
        )
        == 2
    )
    assert "is not after the valuation date" in capsys.readouterr().out


def test_price_tool_list_runs_and_launcher_dispatch(tmp_path, capsys):
    _record(tmp_path)

    assert price_tool_main(["--output-root", str(tmp_path), "--list-runs"]) == 0
    assert "MO_20260928_150000" in capsys.readouterr().out

    code = launcher_main(
        [
            "price-tool",
            "--output-root",
            str(tmp_path),
            "--fit",
            "latest",
            "--strike",
            "7500",
            "--tenor",
            "3M",
        ]
    )
    assert code == 0
    assert "npv" in capsys.readouterr().out


def test_price_tool_quick_defaults(monkeypatch, tmp_path, capsys):
    """``quick=True`` (or running the file with no arguments) uses QUICK_DEFAULTS."""
    import surface_pricer.price_tool as price_tool

    _record(tmp_path)
    monkeypatch.setitem(price_tool.QUICK_DEFAULTS, "output_root", str(tmp_path))
    monkeypatch.setitem(price_tool.QUICK_DEFAULTS, "strike", 7500.0)
    monkeypatch.setitem(price_tool.QUICK_DEFAULTS, "tenor", "3M")
    monkeypatch.setitem(price_tool.QUICK_DEFAULTS, "expiry", None)
    monkeypatch.setitem(price_tool.QUICK_DEFAULTS, "option_type", "put")
    monkeypatch.setitem(price_tool.QUICK_DEFAULTS, "json", True)

    assert price_tool.main([], quick=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"]["option_type"] == "put"
    assert payload["contract"]["strike"] == pytest.approx(7500.0)
    assert payload["contract"]["expiry"] == "2026-12-28"


def test_price_tool_curve_flags_override_the_flat_inputs(tmp_path, capsys):
    """--borrow-curve replaces the flat borrow rate and moves the forward."""
    from surface_pricer.core.borrow_curve import BorrowCurvePillars

    _record(tmp_path)
    pillar_dates = [VALUATION.date() + timedelta(days=days) for days in (30, 90, 180, 365)]
    curve_path = BorrowCurvePillars(
        valuation_date=VALUATION.date(),
        pillar_dates=pillar_dates,
        rates=[0.06] * len(pillar_dates),
        observed=len(pillar_dates),
        forward_source={value.isoformat(): "test" for value in pillar_dates},
    ).to_json(tmp_path / "borrow_curve.json")

    base = [
        "--output-root",
        str(tmp_path),
        "--strike",
        "7500",
        "--tenor",
        "3M",
        "--call",
        "--json",
    ]
    assert price_tool_main(base) == 0
    plain = json.loads(capsys.readouterr().out)

    assert price_tool_main(base + ["--borrow-curve", str(curve_path)]) == 0
    with_curve = json.loads(capsys.readouterr().out)

    # 6% borrow over ~0.25y should pull the forward down by roughly 1.5%
    assert with_curve["forward"] < plain["forward"]
    assert 0.97 < with_curve["forward"] / plain["forward"] < 0.995
