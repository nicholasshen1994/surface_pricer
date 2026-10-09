"""The ETF option path: numeric contract ids must fail loudly, not silently."""

import pytest

from surface_pricer.marketdata import option_contracts as oc
from surface_pricer.marketdata.providers import QuoteApiDataProvider


@pytest.fixture(autouse=True)
def _no_contract_file(tmp_path, monkeypatch):
    """Force an empty map, so this file tests the *no* file path."""
    monkeypatch.setattr(oc, "DATA_DIR", tmp_path)
    oc.clear_contract_map()
    yield
    oc.clear_contract_map()


class _Record:
    """The slice of a gateway snapshot record the ETF path reads."""

    def __init__(self, code, last=1.0, bid=0.9, ask=1.1):
        self.resp_stk_code = code
        self.ticker = "{}.SH".format(code)
        self.last = last
        self.best_bid = bid
        self.best_ask = ask
        self.open_interest = 10.0
        self.volume = 5.0
        self.trading_day = 20261008
        self.time = 150000000


class _Client:
    """A gateway stub: an 'O' chain keyed by numeric contract ids, like the real one."""

    def __init__(self, ids=("10012493", "10012188")):
        self._ids = list(ids)

    def get_category_snapshots(self, category, *, exch_id="", stk_code="", subcategory=""):
        return [_Record(code) for code in self._ids]

    def get_snapshots(self, tickers):
        return [_Record("510500", last=7.279, bid=7.27, ask=7.29)]


def test_a_numeric_id_chain_says_what_is_missing():
    """0 quotes out of 2 raw records must name the reason (no code, no spec)."""
    provider = QuoteApiDataProvider(_Client(), rate=0.0)

    with pytest.raises(RuntimeError) as error:
        provider.load("510500")

    message = str(error.value)
    assert "fetch-contracts --underlying 510500" in message
    assert "510500" in message


def test_a_code_keyed_chain_parses():
    """The same path works the moment the feed carries the option code."""
    provider = QuoteApiDataProvider(
        _Client(ids=("510500C2610M00600", "510500P2610M00600")), rate=0.0
    )

    snapshot = provider.load("510500")

    assert len(snapshot.option_records) == 2
    assert {record.option_type for record in snapshot.option_records} == {"call", "put"}
    assert snapshot.spot == pytest.approx(7.279)
