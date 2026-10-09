"""Greek selection: ``parse_greeks`` and the bumps ``bump_greeks`` skips.

The point of the selection is cost: a run that asks for ``delta`` must not pay
for the vol / rate / borrow / theta bumps.  The stub market below records which
bump helpers were called, so the saving is asserted rather than assumed.
"""

from datetime import datetime

import pytest

from surface_pricer.pricing.risk import diff
from surface_pricer.pricing.results import RiskSettings


class _StubSurface:
    """Two vol pillars; records the per-pillar bumps."""

    def __init__(self):
        self.expiry_dates = (datetime(2026, 4, 5), datetime(2026, 7, 5))
        self.pillars = []

    def bump_pillar(self, pillar, amount):
        self.pillars.append((pillar, amount))
        return self

    def bump_cumulative_backward(self, pillar, amount):
        return self.bump_pillar(pillar, amount)


class _StubCurve:
    """Two curve pillars (so no rebuild is needed) with recording bumps."""

    def __init__(self):
        self.pillar_dates = (datetime(2026, 4, 5), datetime(2026, 7, 5))
        self.pillars = []

    def bump_pillar(self, pillar, amount):
        self.pillars.append((pillar, amount))
        return self


class _StubMarket:
    """Minimal market for the bump harness: spot, curves and ``clone``."""

    def __init__(self):
        self.spot = 100.0
        self.surface = _StubSurface()
        self.rate_curve = _StubCurve()
        self.borrow_curve = _StubCurve()
        self.valuation_date = datetime(2026, 1, 5, 15, 0)

    def clone(self, **changes):
        copy = _StubMarket()
        copy.__dict__.update(self.__dict__)
        copy.__dict__.update(changes)
        return copy


@pytest.fixture
def bumps(monkeypatch):
    """Record which bump helpers run, without touching a surface or a curve."""
    calls = []

    def record(kind):
        def stub(*args, **_kwargs):
            calls.append(kind)
            return args[0]

        return stub

    monkeypatch.setattr(diff, "spot_bump", record("spot"))
    monkeypatch.setattr(diff, "vol_bump", record("vol"))
    monkeypatch.setattr(diff, "parallel_bump", record("curve"))
    return calls


@pytest.mark.parametrize(
    "text, expected",
    [
        (None, ()),
        ("", ()),
        ("none", ()),
        ("npv", ()),
        ("delta", ("delta",)),
        ("delta,vega", ("delta", "vega")),
        ("Delta VEGA", ("delta", "vega")),
        ("delta_shares", ("delta_n",)),
        (["gamma", "theta"], ("gamma", "theta")),
        ("gamma_cash", ("gamma_cash",)),
        ("all", diff.GREEK_NAMES),
    ],
)
def test_parse_greeks(text, expected):
    assert diff.parse_greeks(text) == expected


def test_parse_greeks_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="vomma"):
        diff.parse_greeks("delta,vomma")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("buckets", diff.BUCKET_NAMES),
        ("bucketed_vega", ("bucketed_vega",)),
        ("all,buckets", diff.GREEK_NAMES + diff.BUCKET_NAMES),
        ("volga,vanna", diff.SECOND_ORDER_GREEKS),
        ("all,buckets,volga,vanna", diff.ALL_GREEK_NAMES),
    ],
)
def test_parse_greeks_knows_the_bucket_names(text, expected):
    assert diff.parse_greeks(text) == expected


def test_an_unknown_greek_is_refused_by_name():
    """A name the selection cannot produce is named - not silently ``None``."""

    def value(market):
        return 42.0

    with pytest.raises(ValueError, match="vomma"):
        diff.bump_greeks(value, _StubMarket(), RiskSettings(greeks=("vomma",)))


def test_the_second_order_stencils_are_exact(monkeypatch):
    """volga / vanna use the vanilla stencils and the vanilla reporting units."""
    market = _StubMarket()
    market.vol = 0.20

    monkeypatch.setattr(
        diff, "vol_bump", lambda state, amount: state.clone(vol=state.vol + amount)
    )
    monkeypatch.setattr(
        diff, "spot_bump", lambda state, amount: state.clone(spot=state.spot + amount)
    )

    def value(state):
        # V = spot x vol + vol^2 / 2: d2V/dvol2 = 1 and d2V/(dspot dvol) = 1 exactly, so
        # a stencil (or a reporting scaling) that is off shows up as a number != 1
        return state.spot * state.vol + 0.5 * state.vol ** 2

    greeks = diff.bump_greeks(value, market, RiskSettings(greeks=("vega", "volga", "vanna")))

    assert greeks["vega"] == pytest.approx((market.spot + market.vol) / 100.0)
    assert greeks["volga"] == pytest.approx(1.0 / 10000.0)  # per (1 vol point)^2
    assert greeks["vanna"] == pytest.approx(1.0 / 100.0)  # per 1 vol point


def test_the_second_order_greeks_share_the_vol_states(bumps):
    """volga rides the vega pair; vanna only adds the crossed spot states."""

    def value(market):
        return 42.0

    greeks = diff.bump_greeks(
        value, _StubMarket(), RiskSettings(greeks=("vega", "volga", "vanna"))
    )

    # two vol states (+/-) for vega / volga / vanna, then the four crossed states
    assert bumps.count("vol") == 2
    assert bumps == ["vol", "vol", "spot", "spot", "spot", "spot"]
    assert greeks["volga"] == 0.0 and greeks["vanna"] == 0.0


def test_a_second_order_selection_only_pays_for_its_states(bumps):
    """The valuation count: base + 2 vol states (+ 4 crossed ones for vanna)."""
    calls = []

    def value(market):
        calls.append(market.spot)
        return 42.0

    diff.bump_greeks(value, _StubMarket(), RiskSettings(greeks=("vega", "volga")))
    assert len(calls) == 3  # base, vol up, vol down

    calls.clear()
    diff.bump_greeks(value, _StubMarket(), RiskSettings(greeks=("vanna",)))
    assert len(calls) == 5  # base plus the four crossed states

    calls.clear()
    diff.bump_greeks(
        value, _StubMarket(), RiskSettings(greeks=("vega", "volga", "vanna"))
    )
    assert len(calls) == 7  # ... and no state is valued twice


def test_all_excludes_the_buckets():
    """Buckets are a bump pair each - they never ride along with ``all``."""
    assert diff.parse_greeks("all") == diff.GREEK_NAMES
    assert not set(diff.BUCKET_NAMES).intersection(diff.parse_greeks("all"))
    # the second-order Greeks are opt-in on both sides, exactly like the buckets
    assert not set(diff.SECOND_ORDER_GREEKS).intersection(diff.parse_greeks("all"))


def test_bucket_greeks_are_opt_in_and_bump_every_pillar():
    from surface_pricer.pricing.risk import buckets

    def value(market):
        return 42.0

    market = _StubMarket()

    empty = buckets.bucket_greeks(value, market, RiskSettings())
    assert all(not item for item in empty.values())
    assert market.surface.pillars == []  # nothing was bumped

    selected = buckets.bucket_greeks(
        value,
        market,
        RiskSettings(greeks=("bucketed_vega", "bucketed_rhoq", "bucketed_rho")),
    )

    assert set(selected["bucketed_vega"]) == {"2026-04-05", "2026-07-05"}
    assert set(selected["bucketed_rhoq"]) == {"2026-04-05", "2026-07-05"}
    assert set(selected["bucketed_rho"]) == {"2026-04-05", "2026-07-05"}
    assert selected["bucketed_delta"] == {}
    assert len(market.surface.pillars) == 4  # 2 pillars x up/down
    assert len(market.borrow_curve.pillars) == 4
    assert len(market.rate_curve.pillars) == 4


def test_a_subset_only_pays_for_the_bumps_it_needs(bumps):
    def value(market):
        return 42.0

    greeks = diff.bump_greeks(
        value, _StubMarket(), RiskSettings(greeks=("delta", "gamma"))
    )

    assert bumps == ["spot", "spot"]  # one pair, shared by delta and gamma
    assert greeks["delta"] == 0.0
    assert greeks["gamma"] == 0.0
    assert greeks["vega"] is None
    assert greeks["theta"] is None
    assert greeks["npv"] == 42.0


def test_cash_gamma_rides_the_spot_pair(monkeypatch):
    """``gamma_cash`` = ``gamma x spot^2 / 100``: one more field, no extra bump."""
    calls = []

    def spot_bump(market, amount):  # a real bump, so gamma is not trivially zero
        calls.append(amount)
        return market.clone(spot=market.spot + amount)

    monkeypatch.setattr(diff, "spot_bump", spot_bump)

    def value(market):  # quadratic in spot, so gamma is a clean 6.0
        return 3.0 * market.spot ** 2

    spot = _StubMarket().spot
    greeks = diff.bump_greeks(
        value, _StubMarket(), RiskSettings(greeks=("gamma_cash",))
    )

    assert len(calls) == 2  # exactly the pair plain gamma costs
    assert greeks["gamma"] is None  # only what was asked for is filled
    assert greeks["gamma_cash"] == pytest.approx(6.0 * spot ** 2 / 100.0, rel=1e-9)


def test_no_selection_prices_without_a_single_bump(bumps):
    def value(market):
        return 42.0

    greeks = diff.bump_greeks(value, _StubMarket(), RiskSettings(greeks=()))

    assert bumps == []
    assert greeks["npv"] == 42.0
    assert all(greeks[name] is None for name in diff.GREEK_NAMES)


def test_the_default_settings_still_run_every_bump(bumps):
    def value(market):
        return 42.0

    greeks = diff.bump_greeks(value, _StubMarket(), RiskSettings())

    # spot up/down, vol up/down, rate up/down, borrow up/down (theta clones)
    assert bumps == ["spot", "spot", "vol", "vol", "curve", "curve", "curve", "curve"]
    assert all(greeks[name] is not None for name in diff.GREEK_NAMES)
