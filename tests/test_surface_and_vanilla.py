import numpy as np

from surface_pricer import (
    BusinessCalendar,
    ConstantRateCurve,
    EDSSabrSurface,
    MarketState,
    RiskSettings,
    VanillaContract,
    price_vanilla_with_risk,
)


def test_surface_short_end_matches_legacy_get_vanilla_infos_slice():
    surface = EDSSabrSurface(
        init_date="2022-11-03 23:59:59.999999",
        init_spot=6566.84,
        calendar=BusinessCalendar("SSE"),
        trading_days_per_year=252,
        expiry_dates=["2022-11-18 23:59:59.999999", "2023-01-20 23:59:59.999999"],
        atm_vols=[0.25, 0.28],
        skews=[-0.5, -0.25],
        convs=[0.15, 0.05],
        left_skews_1=[-0.45, -0.30],
        right_skews_1=[-0.3, -0.15],
        left_skews_2=[-0.20, -0.10],
        right_skews_2=[-0.1, -0.05],
        stickiness_ratio=1.0,
    )
    strikes = np.array(
        [5253.472, 5581.814, 5910.156, 6238.498, 6566.84, 6895.182, 7223.524, 7551.866, 7880.208]
    )

    vols = surface.implied_vol(
        "2022-11-15 23:59:59.999999",
        strikes,
        current_forward=6564.086816243905,
        initial_forward=6564.086816243905,
    )

    expected = np.array(
        [
            0.320845569754885,
            0.3168663727905403,
            0.3043855929732443,
            0.28435776872131513,
            0.24967734310945683,
            0.2119978098759433,
            0.1534084243466957,
            0.0695408258068009,
            0.0695408258068009,
        ]
    )
    np.testing.assert_allclose(vols, expected, rtol=0.0, atol=2.0e-8)


def test_flat_surface_vanilla_price_and_greeks():
    surface = EDSSabrSurface(
        init_date="2026-01-01",
        init_spot=100.0,
        expiry_dates=["2027-01-01"],
        atm_vols=[0.2],
    )
    market = MarketState(
        valuation_date="2026-01-01",
        spot=100.0,
        rate_curve=ConstantRateCurve(0.0, anchor="2026-01-01"),
        surface=surface,
    )
    contract = VanillaContract(expiry="2027-01-01", strike=100.0, option_type="call")

    result = price_vanilla_with_risk(
        contract,
        market,
        RiskSettings(
            delta_bump_pct=0.001,
            gamma_bump_pct=0.001,
            vega_bump=0.0001,
            vanna_vol_bump=0.0001,
            theta_days=1,
        ),
    )

    assert result.implied_vol == 0.2
    assert result.forward == 100.0
    assert result.bucketed_vega["2027-01-01"] == result.vega
    np.testing.assert_allclose(result.npv, 7.965567455405804, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.delta, 0.5398273410892074, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.gamma, 0.01984759222608545, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.vega, 0.39695254731277885, rtol=0.0, atol=1.0e-12)
