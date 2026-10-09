"""The vanilla JSON layer: resolve once, then re-price a payload.

``VanillaSpec`` is what the pricer and the Greeks actually consume, so a strike
expressed as a percentage is turned into a number *before* any bump (a spot bump
measures the trade instead of re-striking it) and a quote can be exported,
hand-edited and priced again.
"""

import json
from datetime import datetime, timedelta

import pytest

from surface_pricer import api
from surface_pricer.core.curves import ConstantRateCurve
from surface_pricer.core.daycount import BusinessCalendar
from surface_pricer.core.market import MarketState
from surface_pricer.fitting.surface import EDSSabrSurface
from surface_pricer.pricing.results import RiskSettings
from surface_pricer.pricing.vanilla import (
    VanillaContract,
    VanillaPricer,
    VanillaSpec,
    calculate_greeks,
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


def _market(spot=SPOT, valuation=VALUATION):
    surface = EDSSabrSurface(
        init_date=valuation,
        init_spot=SPOT,
        expiry_dates=[EXPIRY],
        atm_vols=[VOL],
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
    )
    return MarketState(
        valuation_date=valuation,
        spot=spot,
        rate_curve=ConstantRateCurve(RATE, anchor=valuation),
        borrow_curve=ConstantRateCurve(BORROW, anchor=valuation),
        calendar=BusinessCalendar(name="TEST"),
        trading_days_per_year=243.0,
        holiday_weight=0.05,
        surface=surface,
    )


def _contract(**overrides):
    params = dict(
        expiry=EXPIRY, strike=STRIKE, option_type="call", notional=1.0, underlying="MO"
    )
    params.update(overrides)
    return VanillaContract(**params)


# ------------------------------------------------------------------- resolving
def test_resolve_spec_turns_a_relative_strike_into_a_number():
    market = _market()
    forward = market.forward(EXPIRY)

    absolute = resolve_spec(_contract(), market)
    by_spot = resolve_spec(
        _contract(strike=STRIKE / SPOT, strike_type="percentage"), market
    )
    by_forward = resolve_spec(
        _contract(strike=STRIKE / forward, strike_type="fwd_percentage"), market
    )

    assert by_spot.strike == pytest.approx(STRIKE)
    assert by_forward.strike == pytest.approx(STRIKE, rel=1e-12)
    # provenance is kept, the resolved strike is what the engine reads
    assert absolute.strike_type == "absolute"
    assert absolute.strike_input == pytest.approx(STRIKE)
    assert by_spot.strike_type == "percentage"
    assert by_spot.strike_input == pytest.approx(STRIKE / SPOT)
    assert absolute.underlying == "MO"
    # the market mapping that was priced is recorded on the spec
    assert absolute.spot == pytest.approx(market.spot)
    assert absolute.forward == pytest.approx(forward)
    assert absolute.discount_factor == pytest.approx(market.discount_factor(EXPIRY))
    assert absolute.year_fraction == pytest.approx(market.year_fraction(EXPIRY))


def test_a_relative_strike_is_resolved_once_and_then_frozen():
    """A spot bump must measure the trade, not re-strike a percentage."""
    market = _market()

    absolute = calculate_greeks(_contract(strike=SPOT * 1.01), market)
    relative = calculate_greeks(
        _contract(strike=1.01, strike_type="percentage"), market
    )

    assert relative.npv == pytest.approx(absolute.npv, rel=1e-12)
    assert relative.delta == pytest.approx(absolute.delta, rel=1e-12)
    assert relative.gamma == pytest.approx(absolute.gamma, rel=1e-12)
    assert relative.bucketed_delta == pytest.approx(absolute.bucketed_delta, rel=1e-12)


def test_the_report_shows_how_a_relative_strike_was_resolved():
    from surface_pricer.reporting.quote_report import format_quote

    market = _market()
    spec = resolve_spec(
        _contract(strike=1.08, strike_type="percentage"), market
    )

    text = format_quote(
        VanillaPricer(market).price_spec(spec), spec=spec, request={"tenor": "3M"}
    )
    assert "= 1.08 x spot" in text
    assert "8,100" in text  # the absolute strike the pricer used
    assert "tenor=3M" in text

    # an edited payload must not print arithmetic that does not add up
    from dataclasses import replace

    edited = replace(spec, strike=7000.0)
    text = format_quote(VanillaPricer(market).price_spec(edited), spec=edited)
    assert "used as written" in text
    assert "1.08 x spot" not in text


def test_expired_or_invalid_payloads_are_rejected():
    market = _market()
    payload = resolve_spec(_contract(), market).to_dict()

    with pytest.raises(ValueError, match="already expired"):
        VanillaSpec.from_dict(payload, _market(valuation=EXPIRY + timedelta(days=1)))

    payload["strike"] = 0.0
    with pytest.raises(ValueError, match="strike must be positive"):
        VanillaSpec.from_dict(payload, market)


# ------------------------------------------------------------------- JSON layer
def test_the_resolved_payload_round_trips_and_stays_editable():
    market = _market()
    pricer = VanillaPricer(market)
    spec = resolve_spec(_contract(), market)
    payload = json.loads(json.dumps(spec.to_dict()))

    rebuilt = VanillaSpec.from_dict(payload, market)

    assert rebuilt.strike == pytest.approx(spec.strike)
    assert rebuilt.expiry_date == spec.expiry_date
    assert rebuilt.notional == spec.notional
    assert rebuilt.strike_type == "absolute"
    assert pricer.npv_spec(rebuilt) == pytest.approx(pricer.npv_spec(spec), rel=1e-12)

    # ``strike`` is the field the engine reads: editing it re-prices the option
    payload["strike"] = spec.strike * 0.9
    edited = VanillaSpec.from_dict(payload, market)
    assert edited.strike == pytest.approx(spec.strike * 0.9)
    assert pricer.npv_spec(edited) > pricer.npv_spec(spec)  # a lower strike is worth more

    # ... while ``strike_type`` is provenance only: a percentage label next to a
    # resolved strike does not re-strike it
    payload["strike_type"] = "percentage"
    assert VanillaSpec.from_dict(payload, market).strike == pytest.approx(
        spec.strike * 0.9
    )

    # the raw input ratio is not a payload field any more (2026-10): a payload (or
    # a saved quote) that still carries it is refused, not quietly ignored
    with pytest.raises(ValueError, match="strike_input"):
        VanillaSpec.from_dict({**payload, "strike_input": 999.0}, market)

    # the whole ``--json`` quote (a wrapper around the spec) is accepted too
    assert VanillaSpec.from_dict({"contract": payload}, market).strike == pytest.approx(
        spec.strike * 0.9
    )


def test_from_dict_prices_a_payload_on_another_day():
    """The trade stays put, the market mapping moves with the market."""
    payload = resolve_spec(_contract(), _market()).to_dict()

    later = _market(valuation=VALUATION + timedelta(days=30))
    rebuilt = VanillaSpec.from_dict(payload, later)

    assert rebuilt.strike == pytest.approx(STRIKE)
    assert rebuilt.valuation_date == later.valuation_date
    assert rebuilt.forward == pytest.approx(later.forward(EXPIRY))
    assert rebuilt.year_fraction == pytest.approx(later.year_fraction(EXPIRY))
    assert rebuilt.spot == pytest.approx(later.spot)


def test_rebased_moves_the_market_only():
    market = _market()
    spec = resolve_spec(_contract(), market)

    moved = spec.rebased(
        market.clone(spot=SPOT * 0.95, valuation_date=VALUATION + timedelta(days=7))
    )

    assert moved.strike == spec.strike
    assert moved.expiry_date == spec.expiry_date
    assert moved.notional == spec.notional
    assert moved.spot == pytest.approx(SPOT * 0.95)
    assert moved.year_fraction < spec.year_fraction


def test_spec_greeks_match_the_contract_greeks():
    market = _market()
    settings = RiskSettings()

    from_contract = calculate_greeks(_contract(), market, settings)
    from_spec = calculate_greeks_spec(resolve_spec(_contract(), market), market, settings)

    for name in ("npv", "delta", "delta_cash", "gamma", "vega", "theta", "rho", "rhoq"):
        assert getattr(from_spec, name) == pytest.approx(
            getattr(from_contract, name), rel=1e-12
        )
    assert from_spec.bucketed_vega == pytest.approx(from_contract.bucketed_vega, rel=1e-12)


def test_price_json_routes_a_resolved_payload_as_written():
    """A payload of kind ``vanilla_spec`` is priced as written, not re-resolved."""
    market_payload = {
        "valuation_date": VALUATION.isoformat(),
        "products": {
            "MO": {
                "type": "equity",
                "spot": SPOT,
                "currency": "CNY",
                "borrow_rate": BORROW,
                "vol_surface": dict(_market().surface.to_dict(), type="eds_sabr"),
            },
            "CNY": {"type": "currency", "rate_curve": RATE},
        },
    }
    spec = resolve_spec(_contract(), _market())
    payload = spec.to_dict()
    # a percentage provenance next to an absolute strike: the resolved route must
    # use the number as written, the raw route would multiply it by the spot
    payload["strike_type"] = "percentage"

    resolved = api.price_json(payload, market_payload)
    raw = api.price_json(
        {
            "expiry_date": EXPIRY.isoformat(),
            "strike": STRIKE / SPOT,
            "strike_type": "percentage",
            "option_type": "call",
        },
        market_payload,
    )

    assert resolved.npv == pytest.approx(VanillaPricer(_market()).npv_spec(spec), rel=1e-9)
    assert raw.npv == pytest.approx(resolved.npv, rel=1e-9)
    assert api.price_json(payload, market_payload, with_risk=True).delta is not None
