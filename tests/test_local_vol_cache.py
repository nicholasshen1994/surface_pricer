"""The local-vol table cache: payload round trip, the disk store, and one table per run."""

import json
from datetime import date, datetime

import numpy as np
import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.io.local_vol_cache import LocalVolFileCache, table_digest
from surface_pricer.pricing.exotics.autocall import (
    AutocallContract,
    build_schedule,
)
from surface_pricer.pricing.exotics.autocall.pde import AutocallPDE
from surface_pricer.pricing.models.localvol import (
    LOCAL_VOL_NODES,
    LOCAL_VOL_SLICE_STEP,
    DupireLocalVol,
    make_table_cache,
)
from surface_pricer.pricing.results import RiskSettings

VALUATION = datetime(2026, 1, 5, 15, 0)
SPOT = 100.0
PILLARS = (date(2026, 4, 5), date(2026, 10, 5), date(2027, 1, 5))
OBSERVATIONS = (date(2026, 4, 5), date(2026, 7, 5), date(2026, 10, 5), date(2027, 1, 5))

#: What the app fingerprints: the fit run / surface, the *resolved* curve files
#: (name + hash), the valuation date.  Here written by hand.
FINGERPRINT = {
    "run": "MO_20260105_150000",
    "surface": {"file": "surface.json", "sha1": "0123456789abcdef"},
    "rate_curve": {"file": "ir_curve_20260105_090000.json", "sha1": "fedcba9876543210"},
    "borrow_curve": {"flat": 0.0},
    "valuation_date": VALUATION.isoformat(sep=" "),
}
SPOTS = np.linspace(80.0, 120.0, 9)


def _surface(atm_vols=(0.20, 0.26, 0.22)):
    return EDSSabrSurface(
        init_date=VALUATION,
        init_spot=SPOT,
        expiry_dates=list(PILLARS),
        atm_vols=list(atm_vols),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
    )


def _market(**surface_kwargs):
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(0.02, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(0.0, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=252.0,
        holiday_weight=0.0,
        surface=_surface(**surface_kwargs),
    )


def test_a_table_round_trips_through_its_payload():
    """The stored coefficients answer exactly like the rebuilt ones."""
    market = _market()
    table = DupireLocalVol(market)
    table.prepare(PILLARS)

    again = DupireLocalVol.from_payload(table.to_payload(), market)

    for day in PILLARS:
        assert again.local_vols(day, SPOTS) == pytest.approx(
            table.local_vols(day, SPOTS), rel=1e-15
        )
    # the discretisation and the diagnostics travel with the coefficients
    assert again.nodes == table.nodes
    assert again.slice_step == table.slice_step
    assert again.spot_anchor == pytest.approx(table.spot_anchor)
    assert again.describe() == table.describe()
    # and interpolation between the stored slices still works
    mid = datetime(2026, 7, 5)
    assert again.local_vols(mid, SPOTS) == pytest.approx(
        table.local_vols(mid, SPOTS), rel=1e-15
    )


def test_the_file_cache_records_the_inputs_and_the_build_time(tmp_path):
    market = _market()
    store = LocalVolFileCache(tmp_path)
    cache = make_table_cache(spot_anchor=SPOT, store=store, fingerprint=FINGERPRINT)

    table = cache.table(market, PILLARS)

    assert cache.builds == 1 and cache.loads == 0
    assert store.writes == 1 and store.hits == 0
    stored = json.loads(store.last_path.read_text(encoding="utf-8"))
    assert stored["kind"] == "local_vol_table"
    assert stored["created_at"] and stored["created_at"][:2] == "20"
    # everything a reader needs to check the file: the surface, the curves, the grid
    assert stored["inputs"]["run"] == FINGERPRINT["run"]
    assert stored["inputs"]["surface"] == FINGERPRINT["surface"]
    assert stored["inputs"]["rate_curve"] == FINGERPRINT["rate_curve"]
    assert stored["inputs"]["spot_anchor"] == pytest.approx(SPOT)
    assert stored["inputs"]["nodes"] == LOCAL_VOL_NODES
    assert stored["inputs"]["slice_step"] == pytest.approx(LOCAL_VOL_SLICE_STEP)
    assert stored["inputs"]["dates"][0].startswith("2026-04-05")

    # a second run with the same inputs *loads* instead of building
    second = make_table_cache(
        spot_anchor=SPOT, store=LocalVolFileCache(tmp_path), fingerprint=FINGERPRINT
    )
    loaded = second.table(market, PILLARS)

    assert second.loads == 1 and second.builds == 0 and second.source == "cache"
    assert loaded.local_vols(PILLARS[1], SPOTS) == pytest.approx(
        table.local_vols(PILLARS[1], SPOTS), rel=1e-15
    )

    # ... and a different time grid is a different key, not a hit
    third = make_table_cache(
        spot_anchor=SPOT, store=LocalVolFileCache(tmp_path), fingerprint=FINGERPRINT
    )
    third.table(market, PILLARS[:2])
    assert third.builds == 1 and third.loads == 0


def test_disabling_the_cache_writes_nothing_and_reads_nothing(tmp_path):
    store = LocalVolFileCache(tmp_path, enabled=False)
    cache = make_table_cache(store=store, fingerprint=FINGERPRINT)

    cache.table(_market(), PILLARS)

    assert cache.builds == 1 and cache.loads == 0
    assert store.writes == 0 and not any(tmp_path.iterdir())
    assert "cache off" in store.describe(cache)


def test_a_bumped_surface_is_never_served_from_the_file(tmp_path):
    """A vol bump is a different model: it must not read - or poison - the base file."""
    store = LocalVolFileCache(tmp_path)
    cache = make_table_cache(spot_anchor=SPOT, store=store, fingerprint=FINGERPRINT)
    market = _market()

    cache.table(market, PILLARS)
    bumped = market.clone(surface=_surface(atm_vols=(0.30, 0.36, 0.32)))
    cache.table(bumped, PILLARS)

    assert cache.builds == 2 and cache.loads == 0  # the bump rebuilt, in memory
    assert store.writes == 1  # and wrote nothing over the base table
    assert len(list(store.root.iterdir())) == 1


def test_one_run_builds_one_table_for_every_market(tmp_path):
    """The engine's table provider: a bump run / a ladder shares a single table."""
    market = _market()
    contract = AutocallContract(
        underlying="TEST",
        start_date=VALUATION,
        expiry_date=PILLARS[-1],
        observation_dates=OBSERVATIONS,
        ko_levels=(1.0, 1.0, 1.0, 1.0),
        ki_level=0.75,
        annual_coupon=0.10,
        notional=1.0,
    )
    schedule = build_schedule(contract, market)
    cache = make_table_cache(
        spot_anchor=SPOT,
        store=LocalVolFileCache(tmp_path),
        fingerprint=FINGERPRINT,
    )
    engine = AutocallPDE(local_vol_cache=cache)

    values = []
    for spot in (100.0, 95.0, 105.0):  # the base and two ladder rungs
        moved = market.clone(spot=spot)
        values.append(engine.price_schedule(schedule.rebased(moved), moved, RiskSettings()))

    assert cache.builds == 1  # one table, three markets
    assert cache.loads == 0
    assert len(values) == 3 and all(item.npv > 0.0 for item in values)


def test_tables_are_filed_per_index(tmp_path):
    """A table belongs to one index: two indices never share a folder."""
    per_index = LocalVolFileCache(tmp_path, index="000852.SH")
    other_index = LocalVolFileCache(tmp_path, index="510500")
    unfiled = LocalVolFileCache(tmp_path)

    assert per_index.root == tmp_path / "local_vol" / "000852"
    assert other_index.root == tmp_path / "local_vol" / "510500"
    assert unfiled.root == tmp_path / "local_vol"  # a caller that names no index

    cache = make_table_cache(
        spot_anchor=SPOT,
        store=per_index,
        fingerprint={**FINGERPRINT, "index": "000852"},
    )
    cache.table(_market(), PILLARS)

    assert per_index.writes == 1
    assert len(list(per_index.root.iterdir())) == 1
    assert not other_index.root.exists()  # nothing was filed for the other index

    # the other index starts from nothing, even with an otherwise identical key
    second = make_table_cache(
        spot_anchor=SPOT,
        store=other_index,
        fingerprint={**FINGERPRINT, "index": "510500"},
    )
    second.table(_market(), PILLARS)
    assert second.loads == 0 and second.builds == 1
    assert (other_index.root / list(per_index.root.iterdir())[0].name).is_file() is False


def test_the_table_key_covers_ir_borrow_vol_and_spot(tmp_path):
    """The four inputs of a local-vol table: any one of them changing is a new key."""
    import argparse

    from surface_pricer.apps._market import local_vol_fingerprint
    from surface_pricer.io.fit_runs import FitRun

    run_dir = tmp_path / "vol_fit" / "MO_20260105_150000"
    run_dir.mkdir(parents=True)
    surface_file = run_dir / "surface.json"
    surface_file.write_text('{"type": "eds_sabr", "atm_vols": [0.2]}', encoding="utf-8")
    run = FitRun(
        name=run_dir.name,
        directory=run_dir,
        surface_path=surface_file,
        manifest={"underlying": "000852.SH", "spot": SPOT},
    )
    ir_one = tmp_path / "ir_curve_one.json"
    ir_two = tmp_path / "ir_curve_two.json"
    borrow_one = tmp_path / "borrow_curve_one.json"
    borrow_two = tmp_path / "borrow_curve_two.json"
    for path in (ir_one, ir_two, borrow_one, borrow_two):
        path.write_text("{}", encoding="utf-8")

    def fingerprint(ir_path, borrow_path, spot):
        args = argparse.Namespace(
            output_root=tmp_path,
            ir_curve=str(ir_path),
            borrow_curve=str(borrow_path),
            spot=spot,
            rate=None,
            borrow=None,
        )
        return local_vol_fingerprint(run, _market().clone(spot=spot), args)

    base = fingerprint(ir_one, borrow_one, SPOT)
    assert base["index"] == "000852"
    assert base["spot_anchor"] == pytest.approx(SPOT)

    # the same four inputs are the same key (and the same file name) every time
    assert fingerprint(ir_one, borrow_one, SPOT) == base
    assert table_digest(fingerprint(ir_one, borrow_one, SPOT)) == table_digest(base)

    # ... and each of the four moves it
    assert fingerprint(ir_two, borrow_one, SPOT) != base
    assert fingerprint(ir_one, borrow_two, SPOT) != base
    assert fingerprint(ir_one, borrow_one, SPOT * 1.01) != base
    surface_file.write_text('{"type": "eds_sabr", "atm_vols": [0.3]}', encoding="utf-8")
    assert fingerprint(ir_one, borrow_one, SPOT) != base
    assert table_digest(fingerprint(ir_one, borrow_one, SPOT)) != table_digest(base)


def test_the_digest_is_stable_and_discriminating():
    assert table_digest(FINGERPRINT) == table_digest(dict(reversed(list(FINGERPRINT.items()))))
    assert table_digest(FINGERPRINT) != table_digest({**FINGERPRINT, "run": "OTHER"})


class _Store:
    """A store stub that always serves one payload (and counts the writes)."""

    def __init__(self, payload=None):
        self.payload = payload
        self.written = 0

    def read(self, fingerprint):
        return self.payload

    def write(self, fingerprint, table):
        self.written += 1


def test_a_stored_table_for_another_spot_is_refused_and_rebuilt(tmp_path):
    """The file must say *which spot* it was built around - and be held to it."""
    store = LocalVolFileCache(tmp_path)
    cache = make_table_cache(spot_anchor=SPOT, store=store, fingerprint=FINGERPRINT)
    market = _market()

    cache.table(market, PILLARS)
    assert cache.builds == 1 and store.writes == 1
    (path,) = list(store.root.iterdir())  # <tmp>/local_vol/lv_<hash>.json
    stored = json.loads(path.read_text(encoding="utf-8"))
    # the spot is in both halves of the file: it names the table (inputs) ...
    assert stored["inputs"]["spot_anchor"] == pytest.approx(SPOT)
    # ... and the coefficients were built around it
    assert stored["table"]["spot_anchor"] == pytest.approx(SPOT)

    # somebody edits the table's spot (or a hash collision produced this file)
    stored["table"]["spot_anchor"] = SPOT + 1.0
    path.write_text(json.dumps(stored), encoding="utf-8")

    again = make_table_cache(
        spot_anchor=SPOT, store=LocalVolFileCache(tmp_path), fingerprint=FINGERPRINT
    )
    table = again.table(market, PILLARS)

    assert again.loads == 0 and again.builds == 1  # refused, not served
    assert table.spot_anchor == pytest.approx(SPOT)
    healed = json.loads(path.read_text(encoding="utf-8"))
    assert healed["table"]["spot_anchor"] == pytest.approx(SPOT)  # rewritten in place

    # ... and a file that does not say which spot at all cannot be verified either
    del healed["table"]["spot_anchor"]
    path.write_text(json.dumps(healed), encoding="utf-8")
    third = make_table_cache(
        spot_anchor=SPOT, store=LocalVolFileCache(tmp_path), fingerprint=FINGERPRINT
    )
    third.table(market, PILLARS)
    assert third.loads == 0 and third.builds == 1


def test_the_engine_rebuilds_a_stored_table_built_at_another_spot():
    """Even a store that lies about it cannot move the table's anchor."""
    market = _market()
    foreign = DupireLocalVol(market, spot_anchor=SPOT + 5.0)
    foreign.prepare(PILLARS)
    store = _Store(foreign.to_payload())
    cache = make_table_cache(spot_anchor=SPOT, store=store, fingerprint=FINGERPRINT)

    table = cache.table(market, PILLARS)

    assert cache.stale == 1 and cache.loads == 0 and cache.builds == 1
    assert table.spot_anchor == pytest.approx(SPOT)
    assert store.written == 1  # the correct table replaced the one that was served
    assert table.local_vols(PILLARS[0], SPOTS).shape == SPOTS.shape


def test_an_unpinned_cache_resolves_the_anchor_once(tmp_path):
    """No pinned anchor: the **first** market's spot anchors the run - and the file.

    A bump (or a ladder rung) must not follow it: the coefficients are pinned to
    the anchor, so one run has one table and one file, whichever spot a later
    market is at.
    """
    store = LocalVolFileCache(tmp_path)
    cache = make_table_cache(store=store, fingerprint=FINGERPRINT)  # no pin
    base = _market()

    first = cache.table(base, PILLARS)
    bumped = cache.table(base.clone(spot=SPOT * 1.01), PILLARS)

    assert bumped is first and cache.builds == 1 and cache.loads == 0
    assert first.spot_anchor == pytest.approx(SPOT)
    assert store.writes == 1 and len(list(store.root.iterdir())) == 1
    stored = json.loads(next(store.root.iterdir()).read_text(encoding="utf-8"))
    assert stored["inputs"]["spot_anchor"] == pytest.approx(SPOT)
    assert stored["table"]["spot_anchor"] == pytest.approx(SPOT)
