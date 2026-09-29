"""Portfolio layer: term sheets, lifecycle, valuation and reports."""

import csv
import json
from datetime import datetime

import pytest

from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.portfolio.report import portfolio_to_dict, write_csv, write_json
from surface_pricer.portfolio.schedule import TradeStatus, build_schedule
from surface_pricer.portfolio.terms import ObservationRecord, TradeTerms, load_terms
from surface_pricer.portfolio.valuation import value_portfolio, value_trade
from surface_pricer.pricing.results import RiskSettings

VALUATION = datetime(2026, 9, 28, 15, 0)
EXPIRY = datetime(2027, 3, 19, 15, 0)
SPOT = 7500.0
VOL = 0.22


def _market():
    return MarketState(
        valuation_date=VALUATION,
        spot=SPOT,
        rate_curve=ConstantRateCurve(0.02, anchor=VALUATION),
        borrow_curve=ConstantRateCurve(0.01, anchor=VALUATION),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
        surface=EDSSabrSurface(
            init_date=VALUATION,
            init_spot=SPOT,
            expiry_dates=[EXPIRY],
            atm_vols=[VOL],
            calendar=BusinessCalendar(name="TEST"),
            trading_days_per_year=243.0,
            holiday_weight=0.05,
        ),
    )


def _payload(**overrides):
    payload = {
        "trade_id": "TRD-001",
        "underlying": "MO",
        "product_type": "vanilla",
        "booked_date": "2026-06-01",
        "start_date": "2026-06-01",
        "expiry_date": "2027-03-19",
        "call_put": "call",
        "strike": 7600.0,
        "strike_type": "absolute",
        "notional": 1000.0,
    }
    payload.update(overrides)
    return payload


def _terms(**overrides):
    return TradeTerms.from_dict(_payload(**overrides))


# ------------------------------------------------------------------ loading
def test_load_terms_from_json(tmp_path):
    source = tmp_path / "trades.json"
    source.write_text(json.dumps({"trades": [_payload(), _payload(trade_id="TRD-002")]}), encoding="utf-8")

    trades = load_terms(str(source))

    assert [trade.trade_id for trade in trades] == ["TRD-001", "TRD-002"]
    assert trades[0].expiry_date.date() == EXPIRY.date()
    assert trades[0].notional == pytest.approx(1000.0)


def test_load_terms_from_csv(tmp_path):
    source = tmp_path / "trades.csv"
    header = [
        "trade_id",
        "underlying",
        "product_type",
        "start_date",
        "expiry_date",
        "call_put",
        "strike",
        "notional",
        "ki_barrier",
        "ko_barrier",
        "ki_flag",
        "ko_flag",
        "observations",
    ]
    row = [
        "TRD-010",
        "MO",
        "vanilla",
        "2026-06-01",
        "2027-03-19",
        "put",
        "7300",
        "500000",
        "7200",
        "",
        "false",
        "false",
        '[{"date": "2026-09-18", "spot": 7480.0}]',
    ]
    with open(source, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(row)

    trade = load_terms(str(source))[0]

    assert trade.trade_id == "TRD-010"
    assert trade.call_put == "put"
    assert trade.strike == pytest.approx(7300.0)
    assert trade.ki_barrier == pytest.approx(7200.0)
    assert trade.ko_barrier is None
    assert trade.observations == (ObservationRecord(date="2026-09-18", spot=7480.0),)


def test_load_terms_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        _terms(trade_id="")
    with pytest.raises(ValueError):
        _terms(strike=0.0)
    with pytest.raises(ValueError):
        _terms(call_put="straddle")
    with pytest.raises(ValueError):
        _terms(start_date="2028-01-01")

    duplicate = tmp_path / "dup.json"
    duplicate.write_text(json.dumps([_payload(), _payload()]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_terms(str(duplicate))

    missing = tmp_path / "none.csv"
    missing.write_text("trade_id,underlying\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_terms(str(missing))


# ----------------------------------------------------------------- lifecycle
def test_schedule_status_transitions():
    market = _market()
    assert build_schedule(_terms(), market.valuation_date).status == TradeStatus.ACTIVE
    assert (
        build_schedule(_terms(start_date="2026-12-01"), market.valuation_date).status
        == TradeStatus.NOT_STARTED
    )
    assert (
        build_schedule(_terms(expiry_date="2026-08-21"), market.valuation_date).status
        == TradeStatus.EXPIRED
    )


def test_schedule_derives_ki_ko_from_observation_history():
    terms = _terms(
        ki_barrier=7400.0,
        ko_barrier=7900.0,
        ki_flag=True,
        ko_flag=False,
        observations=[
            {"date": "2026-06-18", "spot": 7350.0},
            {"date": "2026-07-18", "spot": 7600.0},
        ],
    )
    schedule = build_schedule(terms, VALUATION)

    assert schedule.ki_triggered is True
    assert schedule.ko_triggered is False
    assert schedule.ki_source == "history"
    assert schedule.next_observation is None
    assert "knock-in already triggered" in schedule.notes

    forward_looking = _terms(
        observations=[
            {"date": "2026-06-18", "spot": 7350.0},
            {"date": "2026-12-18", "spot": 7600.0},
        ]
    )
    future_schedule = build_schedule(forward_looking, VALUATION)
    assert future_schedule.past_observations == (datetime(2026, 6, 18).date(),)
    assert future_schedule.next_observation == datetime(2026, 12, 18).date()


def test_schedule_rejects_flag_mismatch():
    terms = _terms(
        ki_barrier=7400.0,
        ki_flag=False,
        observations=[{"date": "2026-06-18", "spot": 7350.0}],
    )
    with pytest.raises(ValueError, match="KI flag mismatch"):
        build_schedule(terms, VALUATION)


# ----------------------------------------------------------------- valuation
def test_value_trade_scales_with_notional():
    market = _market()
    small = value_trade(_terms(notional=1.0), market)
    large = value_trade(_terms(notional=1000.0), market)

    assert large.npv == pytest.approx(small.npv * 1000.0, rel=1e-12)
    for name in ("delta", "vega", "volga", "rho", "rhoq"):
        assert getattr(large.result, name) == pytest.approx(
            getattr(small.result, name) * 1000.0, rel=1e-9
        )
    label = next(iter(small.result.bucketed_delta))
    assert large.result.bucketed_delta[label] == pytest.approx(
        small.result.bucketed_delta[label] * 1000.0, rel=1e-9
    )


def test_value_trade_skips_expired_and_not_started():
    market = _market()
    expired = value_trade(_terms(expiry_date="2026-08-21"), market)
    assert expired.status == TradeStatus.EXPIRED
    assert expired.npv == 0.0
    assert expired.result is None
    assert "expired" in expired.message

    pending = value_trade(_terms(start_date="2026-12-01"), market)
    assert pending.status == TradeStatus.NOT_STARTED
    assert pending.npv is None
    assert pending.result is None


def test_value_trade_rejects_unsupported_product_type():
    market = _market()
    with pytest.raises(ValueError, match="not supported yet"):
        value_trade(_terms(product_type="barrier"), market)


def test_value_portfolio_aggregates_totals_and_buckets():
    market = _market()
    trades = [
        _terms(trade_id="TRD-001", notional=1000.0),
        _terms(trade_id="TRD-002", notional=250.0, call_put="put", strike=7300.0),
        _terms(trade_id="TRD-003", expiry_date="2026-08-21"),
    ]
    portfolio = value_portfolio(trades, market)

    counts = portfolio.counts_by_status()
    assert counts[TradeStatus.ACTIVE] == 2
    assert counts[TradeStatus.EXPIRED] == 1

    expected_npv = sum(
        trade.npv for trade in portfolio.trades if trade.npv is not None
    )
    assert portfolio.totals["npv"] == pytest.approx(expected_npv, rel=1e-12)
    assert portfolio.totals["delta"] == pytest.approx(
        sum(trade.result.delta for trade in portfolio.trades if trade.result), rel=1e-12
    )

    buckets = portfolio.bucketed_totals()
    assert set(buckets["vega"]) == {EXPIRY.date().isoformat()}
    assert buckets["vega"][EXPIRY.date().isoformat()] == pytest.approx(
        portfolio.totals["vega"], rel=1e-9
    )
    assert set(buckets["delta"]) == set(buckets["rhoq"])


# ------------------------------------------------------------------- reports
def test_reports_are_written(tmp_path):
    portfolio = value_portfolio([_terms(), _terms(trade_id="TRD-002", notional=200.0)], _market())

    csv_path = write_csv(portfolio, str(tmp_path / "trades.csv"))
    json_path = write_json(portfolio, str(tmp_path / "trades.json"))

    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["trade_id"] == "TRD-001"
    assert float(rows[0]["npv"]) == pytest.approx(portfolio.trades[0].npv, rel=1e-9)
    assert json.loads(rows[0]["bucketed_delta"])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts_by_status"]["active"] == 2
    assert payload["totals"]["npv"] == pytest.approx(portfolio.totals["npv"], rel=1e-12)
    assert payload["trades"][0]["terms"]["trade_id"] == "TRD-001"
    assert payload["trades"][0]["schedule"]["status"] == "active"

    summary = portfolio_to_dict(portfolio)
    assert summary["valuation_date"] == VALUATION.date().isoformat()
