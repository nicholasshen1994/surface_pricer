"""``latest`` is filed **per underlying**: a quote cannot move between indices."""

import json
from datetime import datetime

import pytest

from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.fit_runs import (
    bare_code,
    read_latest_map,
    record_fit_run,
    resolve_run,
)

VALUATION = datetime(2026, 1, 5, 15, 0)
SPOT = 100.0


def _record(tmp_path, name, underlying, spot=SPOT):
    return record_fit_run(
        tmp_path / "vol_fit" / name,
        surface=EDSSabrSurface(
            init_date=VALUATION,
            init_spot=spot,
            expiry_dates=[datetime(2026, 10, 5), datetime(2027, 1, 5)],
            atm_vols=[0.20, 0.21],
        ),
        underlying=underlying,
        valuation_datetime=VALUATION,
        spot=spot,
        rate=0.015,
    )


def test_latest_is_filed_per_underlying(tmp_path):
    """A newer run of *another* index must not steal this index's ``latest``."""
    _record(tmp_path, "000852_20261008_100000", "000852.SH", spot=7138.0)
    _record(tmp_path, "000300_20261008_110000", "000300.SH", spot=4310.0)

    assert read_latest_map(tmp_path) == {
        "000852": "000852_20261008_100000",
        "000300": "000300_20261008_110000",
    }

    zhang = resolve_run("latest", tmp_path, underlying="000852.SH")
    assert zhang.name == "000852_20261008_100000"
    assert zhang.spot == pytest.approx(7138.0)
    # the suffix is not part of the key
    assert resolve_run("latest", tmp_path, underlying="000852").name == zhang.name
    assert (
        resolve_run("latest", tmp_path, underlying="000300").name
        == "000300_20261008_110000"
    )

    # an index with no run is an error that lists what there is
    with pytest.raises(ValueError, match="no fit run for 000905"):
        resolve_run("latest", tmp_path, underlying="000905.SH")
    # ... and without an index to go on, "latest" refuses to pick between two
    with pytest.raises(ValueError, match="several underlyings"):
        resolve_run("latest", tmp_path)


def test_latest_without_an_index_works_when_there_is_only_one(tmp_path):
    _record(tmp_path, "000852_20261008_100000", "000852.SH")

    assert resolve_run("latest", tmp_path).name == "000852_20261008_100000"
    assert (
        resolve_run("latest", tmp_path, underlying=["000852.SH", "MO"]).name
        == "000852_20261008_100000"
    )


def test_several_spellings_are_tried_in_order(tmp_path):
    """A run filed under the option venue (an older convention) is still found."""
    _record(tmp_path, "MO_20261008_100000", "MO")

    assert read_latest_map(tmp_path) == {"MO": "MO_20261008_100000"}
    assert (
        resolve_run("latest", tmp_path, underlying=["000852.SH", "MO"]).name
        == "MO_20261008_100000"
    )
    with pytest.raises(ValueError, match="no fit run for 000852"):
        resolve_run("latest", tmp_path, underlying="000852.SH")


def test_the_old_single_pointer_is_read(tmp_path):
    """A pre-2026-10 ``{"run": ...}`` file maps to that run's own underlying."""
    _record(tmp_path, "000852_20261008_100000", "000852.SH")
    (tmp_path / "vol_fit" / "latest.json").write_text(
        json.dumps({"run": "000852_20261008_100000"}), encoding="utf-8"
    )

    assert read_latest_map(tmp_path) == {"000852": "000852_20261008_100000"}
    assert (
        resolve_run("latest", tmp_path, underlying="000852").name
        == "000852_20261008_100000"
    )


def test_bare_code_drops_the_exchange_suffix():
    assert bare_code("000852.SH") == "000852"
    assert bare_code(" 510500.sh ") == "510500"
    assert bare_code(None) == ""


def test_runs_are_filed_in_the_index_folder(tmp_path):
    """``vol_fit/<index>/`` holds the runs and *its* pointer - the path says which."""
    from surface_pricer.io.fit_runs import (
        fit_run_root,
        latest_run_path,
        list_runs,
    )

    def _run(index, name, spot):
        return record_fit_run(
            fit_run_root(tmp_path, index=index) / name,
            surface=EDSSabrSurface(
                init_date=VALUATION,
                init_spot=spot,
                expiry_dates=[datetime(2026, 10, 5), datetime(2027, 1, 5)],
                atm_vols=[0.20, 0.21],
            ),
            underlying=index,
            valuation_datetime=VALUATION,
            spot=spot,
            rate=0.015,
        )

    zhang = _run("000852.SH", "000852_20261009_100000", 7138.0)
    hu = _run("000300.SH", "000300_20261009_110000", 4310.0)

    assert zhang.directory == tmp_path / "vol_fit" / "000852" / "000852_20261009_100000"
    assert hu.directory.parent == tmp_path / "vol_fit" / "000300"
    # the pointer lives **in** the index's folder and names one run (2026-10)
    pointer = json.loads(
        (tmp_path / "vol_fit" / "000852" / "latest.json").read_text(encoding="utf-8")
    )
    assert pointer == {"000852": "000852_20261009_100000"}

    assert latest_run_path(tmp_path, underlying="000852.SH") == zhang.directory
    assert latest_run_path(tmp_path, underlying="000300") == hu.directory
    assert resolve_run("latest", tmp_path, underlying="000852").name == zhang.name
    assert {run.name for run in list_runs(tmp_path)} == {zhang.name, hu.name}
    # the index folders did not merge: one index's pointer is still its own
    assert latest_run_path(tmp_path, underlying="000905.SH") is None
    with pytest.raises(ValueError, match="no fit run for 000905"):
        resolve_run("latest", tmp_path, underlying="000905.SH")
