"""``price-json``: hand it a resolved payload, get the NPV and the Greeks asked for.

The payloads here are **hand-written** (not exported by another command) on
purpose: that is the use case - a contract JSON from a trade store or a desk
spreadsheet, priced against a fit run, with the payload's ``kind`` picking the
pricer and its ``greeks`` list picking the risk run.
"""

import json
from datetime import datetime

import pytest

from surface_pricer.__main__ import main as launcher_main
from surface_pricer.apps.price_json import main as price_json_main
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.fit_runs import record_fit_run

VALUATION = datetime(2026, 1, 5, 15, 0)
SPOT = 100.0
OBSERVATIONS = ("2026-04-05", "2026-07-05", "2026-10-05")


def _surface():
    return EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=[datetime(2026, 10, 5), datetime(2027, 1, 5)],
        atm_vols=[0.20, 0.21],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


def _record(tmp_path):
    return record_fit_run(
        tmp_path / "vol_fit" / "MO_20260105_150000",
        surface=_surface(),
        underlying="MO",
        valuation_datetime=VALUATION,
        spot=SPOT,
        rate=0.02,
        calendar_name="TEST",
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


def _vanilla_payload(**overrides):
    payload = {
        "kind": "vanilla_spec",
        "underlying": "MO",
        "option_type": "call",
        "notional": 1.0,
        "expiry_date": "2026-10-05T00:00:00",
        "strike": 100.0,
        "strike_type": "absolute",
    }
    payload.update(overrides)
    return payload


def _autocall_payload(**overrides):
    """A hand-written snowball payload: absolute levels, no enumerated grid."""
    payload = {
        "kind": "autocall_schedule",
        "underlying": "MO",
        "spot0": SPOT,
        "start_date": "2026-01-05T00:00:00",
        "expiry_date": "2026-10-05T00:00:00",
        "notional": 1.0e6,
        "observations": [
            {"date": day, "ko": 103.0, "coupon_rate": 0.13} for day in OBSERVATIONS
        ],
        "knock_in": {
            "frequency": "observation_dates",
            "strike": 100.0,      # absolute: the loss leg is struck at 100% of spot0
            "gearing": 1.0,
            "protected_principal": 0.0,
            "level": 75.0,        # absolute barrier, one level for every period
        },
        "rebate": 0.13,  # annual rate, the same unit as coupon_rate
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(tmp_path, path, *extra):
    # ``--ir-curve none`` keeps the tests hermetic: the flat --rate is used instead
    # of whichever curve run the developer's output/ happens to hold
    return [
        str(path),
        "--fit", "latest",
        "--output-root", str(tmp_path),
        "--ir-curve", "none",
        "--borrow-curve", "none",
        *extra,
    ]


# ------------------------------------------------------------------ vanilla leg
def test_price_json_prices_a_vanilla_payload(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "vanilla.json", _vanilla_payload())

    code = price_json_main(
        _args(tmp_path, path, "--greeks", "delta,gamma_cash")
    )

    assert code == 0
    echoed, _, report = capsys.readouterr().out.partition("\nfit run")
    assert json.loads(echoed)["strike"] == pytest.approx(100.0)
    assert "npv" in report and "gamma cash" in report


def test_price_json_reads_the_selection_from_the_payload(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "vanilla.json", _vanilla_payload(greeks=["delta"]))

    assert price_json_main(_args(tmp_path, path, "--json")) == 0
    quoted = json.loads(capsys.readouterr().out)

    assert quoted["greeks"]["delta"] is not None
    assert quoted["greeks"]["gamma"] is None
    assert quoted["bucketed"]["bucketed_vega"] == {}

    # a whole quote can be fed back: the contract block is what gets priced, and
    # the quote's ``greeks`` table (a mapping) is not mistaken for a selection
    quote_path = _write(tmp_path, "quote.json", quoted)
    assert price_json_main(_args(tmp_path, quote_path, "--json")) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["npv"] == pytest.approx(quoted["npv"], rel=1e-12)
    assert again["greeks"]["delta"] is None  # NPV only: no selection was given


def test_price_json_prices_a_snowball_payload(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "snowball.json", _autocall_payload())

    code = price_json_main(
        _args(
            tmp_path,
            path,
            "--method", "pde", "--pde-nodes", "201",
            "--greeks", "delta,gamma,gamma_cash", "--json",
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"]["kind"] == "autocall_schedule"
    assert payload["npv"] > 0.0
    assert payload["method"]["method"] == "pde"
    assert payload["greeks"]["gamma_cash"] == pytest.approx(
        payload["greeks"]["gamma"] * SPOT ** 2 / 100.0, rel=1e-9
    )
    # only what was asked for: no bucket bumps
    assert payload["bucketed"]["bucketed_vega"] == {}


def test_price_json_bucket_selection_costs_a_bump_pair_per_pillar(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "snowball.json", _autocall_payload())

    code = price_json_main(
        _args(
            tmp_path,
            path,
            "--method", "pde", "--pde-nodes", "101",
            "--greeks", "buckets", "--json",
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bucketed"]["bucketed_vega"]
    assert payload["greeks"]["delta"] is None


# ------------------------------------------------------------------- refusals
def test_price_json_rejects_an_unknown_payload_kind(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "swap.json", {"kind": "swap_schedule"})

    assert price_json_main(_args(tmp_path, path)) == 2
    assert "unknown payload kind" in capsys.readouterr().out


def test_price_json_reports_an_unreadable_payload(tmp_path, capsys):
    _record(tmp_path)

    assert price_json_main(_args(tmp_path, tmp_path / "missing.json")) == 2
    assert "cannot read" in capsys.readouterr().out


def test_price_json_rejects_an_unknown_greek(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "vanilla.json", _vanilla_payload())

    assert price_json_main(_args(tmp_path, path, "--greeks", "delta,vomma")) == 2
    assert "vomma" in capsys.readouterr().out


def test_the_payload_path_is_read_as_given(tmp_path, monkeypatch, capsys):
    """``price-json contract.json`` - and nothing is searched for.

    No cwd / package / output fallback (2026-10): a wrong path fails here, naming
    the path it tried, instead of quietly pricing a different file that happened
    to share the name.  A *relative* path still resolves against the cwd - that is
    the OS default, not a lookup of its own.
    """
    _record(tmp_path)
    path = _write(tmp_path, "vanilla.json", _vanilla_payload())

    assert price_json_main(
        [
            str(path), "--fit", "latest", "--output-root", str(tmp_path),
            "--ir-curve", "none", "--borrow-curve", "none", "--json",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["npv"] > 0.0

    monkeypatch.chdir(tmp_path)
    assert price_json_main(
        [
            "vanilla.json", "--fit", "latest", "--output-root", str(tmp_path),
            "--ir-curve", "none", "--borrow-curve", "none", "--json",
        ]
    ) == 0
    capsys.readouterr()

    # from anywhere else the bare name is simply not there - and the fallback
    # roots (package, output/) are not consulted
    monkeypatch.chdir(tmp_path.parent)
    assert price_json_main(
        ["vanilla.json", "--fit", "latest", "--output-root", str(tmp_path)]
    ) == 2
    assert "cannot read vanilla.json" in capsys.readouterr().out


def test_a_missing_payload_is_refused(tmp_path, capsys):
    _record(tmp_path)

    assert price_json_main(["--fit", "latest", "--output-root", str(tmp_path)]) == 2
    assert "a contract payload is required" in capsys.readouterr().out


def test_launcher_dispatches_price_json(tmp_path, capsys):
    _record(tmp_path)
    path = _write(tmp_path, "vanilla.json", _vanilla_payload())

    code = launcher_main(
        ["price-json", str(path), "--fit", "latest", "--output-root", str(tmp_path),
         "--ir-curve", "none", "--borrow-curve", "none", "--json"]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["npv"] > 0.0


def test_price_json_quick_defaults(capsys, tmp_path, monkeypatch):
    """``quick=True`` (running the file with no arguments) uses QUICK_DEFAULTS.

    The block fills **arguments**: the same run given flag by flag has to agree
    to the last digit, which is what makes "edit the block, hit Run" a quote and
    not a second code path.
    """
    import surface_pricer.apps.price_json as price_json

    _record(tmp_path)
    payload = _write(tmp_path, "snowball.json", _autocall_payload())
    run_dir = str(tmp_path / "vol_fit" / "MO_20260105_150000")
    for key, value in {
        "fit": run_dir,
        "output_root": str(tmp_path),
        "payload": str(payload),
        "greeks": "delta_cash",
        "method": "pde",
        "pde_nodes": 41,
        "paths": None,
        "seed": None,
        "ir_curve": "none",
        "borrow_curve": "none",
        "spot": None,
        "trigger_basis": "contractual",
        "slide": False,
        "csv": None,
        "json": True,
    }.items():
        monkeypatch.setitem(price_json.QUICK_DEFAULTS, key, value)

    assert price_json_main([], quick=True) == 0
    quote = json.loads(capsys.readouterr().out)

    assert quote["npv"] > 0.0
    assert quote["greeks"]["delta_cash"] is not None

    # flag for flag, the same request - the block must not be a second code path
    assert (
        price_json_main(
            [
                str(payload),
                "--fit", run_dir,
                "--output-root", str(tmp_path),
                "--greeks", "delta_cash",
                "--method", "pde",
                "--pde-nodes", "41",
                "--ir-curve", "none",
                "--borrow-curve", "none",
                "--json",
            ]
        )
        == 0
    )
    again = json.loads(capsys.readouterr().out)
    assert again["npv"] == pytest.approx(quote["npv"])
    assert again["contract"] == quote["contract"]


def test_the_local_vol_table_is_built_once_and_then_loaded(tmp_path, capsys):
    """``local_vol_cache``: the first quote builds and stores the table, the next reads it.

    The table is the expensive part of an exotic quote, so the second run loads the
    coefficients instead of rebuilding them - and the stderr note says which one
    happened (and which file).  ``--no-local-vol-cache`` rebuilds every time.
    """
    _record(tmp_path)
    payload = _write(tmp_path, "snowball.json", _autocall_payload())
    flags = ("--method", "pde", "--pde-nodes", "41", "--json")

    assert price_json_main(_args(tmp_path, payload, *flags)) == 0
    first = capsys.readouterr()
    assert "local vol : 1 built" in first.err
    assert "lv_" in first.err and first.err.rstrip().endswith(")")

    assert price_json_main(_args(tmp_path, payload, *flags)) == 0
    second = capsys.readouterr()
    assert "1 from cache" in second.err

    # a cached table and a rebuilt one are the same table: same quote
    assert json.loads(second.out)["npv"] == pytest.approx(
        json.loads(first.out)["npv"], rel=1e-12
    )

    # ... and the switch turns the store off (nothing read, nothing written)
    assert price_json_main(_args(tmp_path, payload, *flags, "--no-local-vol-cache")) == 0
    assert "cache off" in capsys.readouterr().err


def test_every_command_renders_its_help(capsys):
    """argparse formats help strings, so one stray ``%`` used to crash ``--help``."""
    for command in (
        "fit",
        "build-json",
        "price-json",
        "slide",
        "build-ir-curve",
        "build-borrow-curve",
    ):
        with pytest.raises(SystemExit) as exit_info:
            launcher_main([command, "--help"])

        assert exit_info.value.code == 0
        assert "usage" in capsys.readouterr().out.lower()
