"""End-to-end smoke tests for the unified command line launcher."""

import json
from datetime import datetime

from surface_pricer.__main__ import main as launcher_main
from surface_pricer.apps.price_trades import main as price_main
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.fitting.surface import EDSSabrSurface

VALUATION = datetime(2026, 9, 28, 15, 0)
EXPIRY = datetime(2027, 3, 19, 15, 0)
SPOT = 7500.0
VOL = 0.22


def _write_surface(tmp_path):
    surface = EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=[EXPIRY],
        atm_vols=[VOL],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(surface.to_dict(), indent=2), encoding="utf-8")
    return path


def _write_terms(tmp_path):
    payload = {
        "trades": [
            {
                "trade_id": "TRD-001",
                "underlying": "MO",
                "product_type": "vanilla",
                "start_date": "2026-06-01",
                "expiry_date": "2027-03-19",
                "call_put": "call",
                "strike": 7600.0,
                "notional": 1000000,
            }
        ]
    }
    path = tmp_path / "trades.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_price_command_writes_reports(tmp_path, capsys):
    surface_path = _write_surface(tmp_path)
    terms_path = _write_terms(tmp_path)
    out_dir = tmp_path / "out"

    code = price_main(
        [
            "--terms",
            str(terms_path),
            "--surface",
            str(surface_path),
            "--rate",
            "0.02",
            "--out",
            str(out_dir),
            "--quiet",
        ]
    )

    assert code == 0
    summary = capsys.readouterr().out
    assert "TRD-001" in summary

    csv_path = out_dir / "trades.csv"
    json_path = out_dir / "trades.json"
    assert csv_path.is_file() and json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts_by_status"]["active"] == 1
    assert payload["totals"]["npv"] != 0.0
    assert payload["trades"][0]["terms"]["trade_id"] == "TRD-001"


def test_price_command_reports_missing_term_sheet(tmp_path, capsys):
    code = price_main(["--terms", str(tmp_path / "missing.json"), "--surface", "x.json"])
    assert code == 2
    assert "cannot read the term sheet" in capsys.readouterr().out


def test_launcher_dispatches_price(tmp_path):
    surface_path = _write_surface(tmp_path)
    terms_path = _write_terms(tmp_path)
    out_dir = tmp_path / "out_launcher"

    code = launcher_main(
        [
            "price",
            "--terms",
            str(terms_path),
            "--surface",
            str(surface_path),
            "--out",
            str(out_dir),
            "--quiet",
        ]
    )

    assert code == 0
    assert (out_dir / "trades.json").is_file()


def test_launcher_help_and_unknown_command(capsys):
    assert launcher_main([]) == 0
    assert "fit" in capsys.readouterr().out

    assert launcher_main(["nope"]) == 2
