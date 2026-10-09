"""The listed-option contract map: the local file, the merge, and the snapshot join."""

import json
from datetime import datetime

import pytest

from surface_pricer.marketdata import option_contracts as oc
from surface_pricer.marketdata.option_contracts import ContractMap, fetch_chain, record_key
from surface_pricer.marketdata.providers import QuoteApiDataProvider
from surface_pricer.marketdata.registry import get_underlying_spec

STAMP = datetime(2026, 10, 9, 10, 0, 0)


def _row(key, code, *, call_put="call", strike=6.0, maturity="2026-11-25", unit=10000.0):
    return {
        "key": key,
        "underlying": "510500",
        "code": code,
        "call_put": call_put,
        "strike": strike,
        "maturity": maturity,
        "unit": unit,
        "exchange": "SSE",
    }


class _Record:
    """The slice of a gateway snapshot record the ETF path reads."""

    def __init__(self, code, exch_id="0", last=1.0):
        self.resp_stk_code = code
        self.resp_exch_id = exch_id
        self.ticker = "{}.SH".format(code)
        self.last = last
        self.best_bid = 0.9
        self.best_ask = 1.1
        self.open_interest = 10.0
        self.volume = 5.0
        self.trading_day = 20261009
        self.time = 150000000


class _Client:
    """A gateway stub: an ETF chain keyed by numeric ids, exactly like the real one."""

    def __init__(self, ids=("10012493", "10012188")):
        self._ids = list(ids)

    def get_category_snapshots(self, category, *, exch_id="", stk_code="", subcategory=""):
        return [_Record(code) for code in self._ids]

    def get_snapshots(self, tickers):
        return [_Record("510500", last=7.279)]


@pytest.fixture(autouse=True)
def _isolated_file(tmp_path, monkeypatch):
    """Point the packaged file at ``tmp_path`` so no test reads the checked-in one."""
    monkeypatch.setattr(oc, "DATA_DIR", tmp_path)
    oc.clear_contract_map()
    yield
    oc.clear_contract_map()


def test_the_file_round_trips_and_merges_incrementally(tmp_path):
    spec = get_underlying_spec("510500")
    target = tmp_path / oc.FILE_NAME
    contract_map = ContractMap()

    added, total = contract_map.merge(
        spec, [_row("10012493.SH", "510500C2611M06000")], now=STAMP
    )
    assert (added, total) == (1, 1)
    contract_map.save(target)

    again = ContractMap.load(target)
    assert again.lookup("10012493.SH")["strike"] == pytest.approx(6.0)
    assert again.lookup("10012493.SH")["exchange"] == "SSE"
    assert again.venues["510500OP.SH"]["count"] == 1
    assert again.venues["510500OP.SH"]["chain_code"] == "510500OP.SH"
    assert again.updated_at == "2026-10-09 10:00:00"
    assert json.loads(target.read_text(encoding="utf-8"))["kind"] == oc.FILE_KIND

    # a re-fetch brings one contract the file already has plus a new one
    added, total = again.merge(
        spec,
        [
            _row("10012493.SH", "510500C2611M06000"),
            _row("10012494.SH", "510500P2611M06000", call_put="put", strike=6.5),
        ],
        now=datetime(2026, 10, 10, 9, 0, 0),
    )
    assert (added, total) == (1, 2)
    # the existing entry is left exactly as it was (the file only grows)
    assert again.contracts["10012493.SH"]["strike"] == pytest.approx(6.0)
    assert again.updated_at == "2026-10-10 09:00:00"


def test_spec_for_key_carries_the_terms_and_the_code():
    spec = get_underlying_spec("510500")
    contract_map = ContractMap()
    contract_map.merge(
        spec,
        [_row("10012493.SH", "510500C2611M06000", strike=6.75, maturity="2026-11-25")],
        now=STAMP,
    )

    parsed = contract_map.spec_for_key("10012493.sh")  # a lower-case key still reads

    assert parsed is not None
    assert parsed.option_type == "call"
    assert parsed.strike == pytest.approx(6.75)
    assert parsed.expiry == datetime(2026, 11, 25)
    assert parsed.underlying == "510500"
    assert parsed.raw_code == "510500C2611M06000"
    assert parsed.contract_month == "202611"
    assert contract_map.spec_for_key("999999.SH") is None  # a miss is a miss


def test_record_key_appends_the_exchange_suffix():
    assert record_key(_Record("10012493", "0")) == "10012493.SH"
    assert record_key(_Record("90008063", "1")) == "90008063.SZ"
    assert record_key(_Record("MO2610-C-7000", "F")) == "MO2610-C-7000.CFE"
    assert record_key(_Record("MO2610-C-7000.CFE", "F")) == "MO2610-C-7000.CFE"
    assert record_key(_Record("", "0")) == ""


def test_the_snapshot_join_turns_numeric_ids_into_terms(tmp_path):
    """The whole point: a numeric-id chain becomes a real option chain."""
    spec = get_underlying_spec("510500")
    contract_map = ContractMap()
    contract_map.merge(
        spec,
        [
            _row("10012493.SH", "510500C2611M06000", strike=6.0, maturity="2026-11-25"),
            _row(
                "10012188.SH",
                "510500P2612M06500",
                call_put="put",
                strike=6.5,
                maturity="2026-12-23",
            ),
        ],
        now=STAMP,
    )
    contract_map.save()  # DATA_DIR is the tmp dir (fixture)

    snapshot = QuoteApiDataProvider(_Client(), rate=0.0).load("510500")

    assert len(snapshot.option_records) == 2
    calls = [item for item in snapshot.option_records if item.option_type == "call"]
    assert len(calls) == 1
    assert calls[0].strike == pytest.approx(6.0)
    assert calls[0].expiry == datetime(2026, 11, 25)
    assert calls[0].raw_code == "510500C2611M06000"
    puts = [item for item in snapshot.option_records if item.option_type == "put"]
    assert puts[0].expiry == datetime(2026, 12, 23)
    assert puts[0].strike == pytest.approx(6.5)


def test_a_missing_file_means_an_empty_map_and_a_loud_failure():
    """No file -> nothing is guessed: the ETF chain still refuses to price blind."""
    provider = QuoteApiDataProvider(_Client(), rate=0.0)

    with pytest.raises(RuntimeError) as error:
        provider.load("510500")

    message = str(error.value)
    assert "fetch-contracts --underlying 510500" in message
    assert "510500" in message


def test_wind_venue_code_knows_both_families():
    assert oc.wind_venue_code(get_underlying_spec("510500")) == "510500OP.SH"
    assert oc.wind_venue_code(get_underlying_spec("159915")) == "159915OP.SZ"
    assert oc.wind_venue_code(get_underlying_spec("MO")) == "MO.CFE"
    assert oc.wind_venue_code(get_underlying_spec("IO")) == "IO.CFE"


def test_the_call_put_code_is_read_or_refused():
    assert oc._call_put(708001000) == "call"
    assert oc._call_put(708002000) == "put"
    with pytest.raises(ValueError, match="unknown Wind option kind"):
        oc._call_put(708003000)


def test_fetch_chain_is_the_only_database_door(monkeypatch):
    """``fetch_chain`` builds entries from rows; the pricing path never calls it."""
    spec = get_underlying_spec("510500")

    class _Cursor:
        def execute(self, sql, parameters):
            assert "WINDDF.CHINAOPTIONDESCRIPTION" in sql
            assert "CHINAOPTIONCONTPRO" not in sql  # not granted to the reader
            assert parameters == ["510500OP.SH"]

        def fetchall(self):
            return [("10012493.SH", "510500C2611M06000", 708001000, 6.75, "20261125", 10000.0)]

        def close(self):
            return None

    class _Connection:
        def cursor(self):
            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(oc, "connect", lambda url=None: _Connection())

    entries = fetch_chain(spec)

    assert len(entries) == 1
    assert entries[0]["key"] == "10012493.SH"
    assert entries[0]["code"] == "510500C2611M06000"
    assert entries[0]["call_put"] == "call"
    assert entries[0]["strike"] == pytest.approx(6.75)
    assert entries[0]["maturity"] == "2026-11-25"
    assert entries[0]["underlying"] == "510500"
