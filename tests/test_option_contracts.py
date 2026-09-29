from surface_pricer.marketdata.listed_contracts import (
    parse_etf_option_code,
    parse_index_option_ticker,
    parse_listed_option,
    underlying_future_ticker,
)


def test_cffex_index_option_parsing():
    parsed = parse_listed_option("MO2610-C-7600.CFE")
    assert parsed is not None
    assert parsed.underlying == "MO"
    assert parsed.exchange == "CFE"
    assert parsed.option_type == "call"
    assert parsed.strike == 7600.0
    assert parsed.expiry.date().isoformat() == "2026-10-16"
    assert underlying_future_ticker("MO", "202610") == "IM2610.CFE"


def test_direct_index_option_api_still_available():
    parsed = parse_index_option_ticker("IO2612-P-4000.CFE")
    assert parsed is not None
    assert parsed.underlying == "IO"
    assert parsed.is_put
    assert parsed.strike == 4000.0


def test_sse_etf_option_parsing():
    parsed = parse_listed_option("510500C2610M06000")
    assert parsed is not None
    assert parsed.underlying == "510500"
    assert parsed.exchange == "SSE"
    assert parsed.strike == 6.0
    assert parsed.option_type == "call"
    # 2026-10-28 is the fourth Wednesday of October 2026.
    assert parsed.expiry.date().isoformat() == "2026-10-28"


def test_szse_etf_option_parsing():
    parsed = parse_etf_option_code("159915P2612M02000", exchange="SZSE")
    assert parsed is not None
    assert parsed.underlying == "159915"
    assert parsed.exchange == "SZSE"
    assert parsed.strike == 2.0
    assert parsed.is_put


def test_unsupported_code_returns_none():
    assert parse_listed_option("") is None
    assert parse_listed_option("NOT-A-CONTRACT") is None
