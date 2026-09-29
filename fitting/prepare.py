"""Quote cleaning, forward implication and slice preparation.

The chain mirrors the edslib CN preparation pipeline:

1. drop large bid/ask spreads (MAD filter) and enforce monotone OTM prices;
2. imply the forward from put-call parity (single-strike parity by default);
3. select OTM quotes and invert them into bid/mid/ask vols (Jaeckel);
4. eliminate residual vertical/butterfly arbitrage (Jaeckel ``Clamping Down
   on Arbitrage``);
5. build vega/equal/spread weights and emit :class:`SliceData`.

Contract parsing and raw snapshots live in :mod:`providers`; the fit engine
only consumes :class:`SliceData`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from ..core.market import MarketState
from ..core.math.black import black_price
from ..core.math.jaeckel import implied_vol_jaeckel
from ..marketdata.data import OptionQuoteRecord, RawSnapshot
from .settings import FitSettings


@dataclass
class SliceData:
    """Cleaned quotes of one expiry, ready for calibration."""

    expiry: datetime
    tau: float
    forward: float
    discount_factor: float
    strikes: np.ndarray
    ln_moneyness: np.ndarray
    vols: np.ndarray
    bid_vols: np.ndarray
    ask_vols: np.ndarray
    weights: np.ndarray
    option_types: Tuple[str, ...] = ()
    diagnostics: Dict[str, float] = field(default_factory=dict)


def prepare_slices(
    snapshot: RawSnapshot,
    settings: FitSettings,
    market: MarketState,
) -> Tuple[List[SliceData], Dict[str, float]]:
    """Build one :class:`SliceData` per expiry from a raw snapshot.

    Returns ``(slices, forward_overrides)`` where ``forward_overrides`` maps
    the ISO expiry date to the forward used by the calibration, so the caller
    can keep ``MarketState`` / vanilla pricing consistent with the fit.
    """
    grouped = _group_records(snapshot.option_records)
    slices: List[SliceData] = []
    forward_overrides: Dict[str, float] = {}

    for expiry in sorted(grouped):
        if not _within_maturity_window(expiry, market, settings):
            continue
        slice_info = build_slice(
            expiry,
            grouped[expiry],
            snapshot,
            settings,
            market,
        )
        if slice_info is None:
            continue
        slices.append(slice_info)
        forward_overrides[slice_info.expiry.date().isoformat()] = slice_info.forward

    return slices, forward_overrides


def build_slice(
    expiry: datetime,
    quotes: Dict[float, Dict[str, float]],
    snapshot: RawSnapshot,
    settings: FitSettings,
    market: MarketState,
) -> Optional[SliceData]:
    """Clean one expiry's quote table and derive its calibration inputs."""
    discount_factor = market.discount_factor(expiry)
    tau = market.year_fraction(expiry)
    if tau <= 0.0:
        return None

    cleaned = _clean_quote_table(quotes, settings)
    call_quotes = cleaned["call"]
    put_quotes = cleaned["put"]
    if len(call_quotes) < 1 or len(put_quotes) < 1:
        return None

    if settings.forward_source == "future":
        forward = _future_forward(snapshot, expiry)
        forward_source = "future"
    else:
        forward = None
        forward_source = "parity"
    if forward is None:
        forward, forward_source = _implied_forward(call_quotes, put_quotes, discount_factor)
    if forward is None or not np.isfinite(forward) or forward <= 0.0:
        return None

    otm_rows = _select_otm_quotes(call_quotes, put_quotes, forward)
    implied = _imply_volatilities(otm_rows, forward, tau, discount_factor)
    if implied is None:
        return None
    strikes, option_types, mid_vols, bid_vols, ask_vols = implied

    strikes, mid_vols, bid_vols, ask_vols = _eliminate_arbitrage_jaeckel(
        strikes, mid_vols, bid_vols, ask_vols, forward, tau
    )
    if len(strikes) < max(1, int(settings.min_strikes_per_expiry)):
        return None

    weights = _generate_default_weights(
        forward, tau, strikes, mid_vols, bid_vols, ask_vols, settings.weight_mode
    )
    if weights.sum() <= 0.0:
        return None
    weights = weights / weights.sum()

    return SliceData(
        expiry=expiry,
        tau=float(tau),
        forward=float(forward),
        discount_factor=float(discount_factor),
        strikes=np.asarray(strikes, dtype=float),
        ln_moneyness=np.log(np.asarray(strikes, dtype=float) / float(forward)),
        vols=np.asarray(mid_vols, dtype=float),
        bid_vols=np.asarray(bid_vols, dtype=float),
        ask_vols=np.asarray(ask_vols, dtype=float),
        weights=np.asarray(weights, dtype=float),
        option_types=tuple(option_types),
        diagnostics={
            "call_quotes": float(len(call_quotes)),
            "put_quotes": float(len(put_quotes)),
            "n_quotes": float(len(strikes)),
            "forward_source": forward_source,
        },
    )


# --------------------------------------------------------------- quote tables
def _group_records(
    records: Sequence[OptionQuoteRecord],
) -> Dict[datetime, Dict[float, Dict[str, float]]]:
    grouped: Dict[datetime, Dict[float, Dict[str, float]]] = {}
    for record in records:
        by_strike = grouped.setdefault(record.expiry, {})
        slot = by_strike.setdefault(float(record.strike), {})
        prefix = "call" if record.option_type == "call" else "put"
        slot[prefix + "_bid"] = float(record.bid)
        slot[prefix + "_ask"] = float(record.ask)
    return grouped


def _within_maturity_window(
    expiry: datetime,
    market: MarketState,
    settings: FitSettings,
) -> bool:
    valuation = market.valuation_date
    calendar_days = (expiry.date() - valuation.date()).days
    if calendar_days <= 0:
        return False
    if settings.max_expiry_calendar_days and calendar_days > int(settings.max_expiry_calendar_days):
        return False
    if settings.min_expiry_business_days and market.calendar is not None:
        business_days = market.calendar.business_days(valuation.date(), expiry.date())
        if business_days < float(settings.min_expiry_business_days):
            return False
    return True


def _clean_quote_table(
    quotes: Dict[float, Dict[str, float]],
    settings: FitSettings,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """MAD spread filter plus monotonicity cleanup.

    Mirrors ``remove_large_price_spread_by_mad``: call and put quotes are
    cleaned independently, then the surviving OTM mid prices must be monotone
    (puts increasing, calls decreasing in strike).
    """
    strikes = np.asarray(sorted(quotes), dtype=float)
    call_bid = np.asarray([quotes[k].get("call_bid", 0.0) for k in strikes], dtype=float)
    call_ask = np.asarray([quotes[k].get("call_ask", 0.0) for k in strikes], dtype=float)
    put_bid = np.asarray([quotes[k].get("put_bid", 0.0) for k in strikes], dtype=float)
    put_ask = np.asarray([quotes[k].get("put_ask", 0.0) for k in strikes], dtype=float)

    call_keep = _mad_survivors(strikes, call_bid, call_ask, settings.spread_mad_factor)
    put_keep = _mad_survivors(strikes, put_bid, put_ask, settings.spread_mad_factor)

    call_entries = _monotone_otm_entries(
        strikes[call_keep],
        (call_bid[call_keep] + call_ask[call_keep]) * 0.5,
        call_bid[call_keep],
        call_ask[call_keep],
        is_call=True,
    )
    put_entries = _monotone_otm_entries(
        strikes[put_keep],
        (put_bid[put_keep] + put_ask[put_keep]) * 0.5,
        put_bid[put_keep],
        put_ask[put_keep],
        is_call=False,
    )
    return {"call": call_entries, "put": put_entries}


def _mad_survivors(
    strikes: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Return the boolean mask of quotes surviving the MAD spread filter."""
    keep = (bid > 0.0) & (ask > 0.0) & (ask >= bid)
    if not np.any(keep):
        return keep
    spread = (ask - bid)[keep]
    median = float(np.median(spread))
    mad = float(np.mean(np.abs(spread - median)))
    if mad <= 0.0:
        return keep
    limit = float(factor) * mad
    drop = np.zeros_like(keep)
    indices = np.where(keep)[0]
    bad = (spread - median) > limit
    drop[indices[bad]] = True
    return keep & ~drop


def _monotone_otm_entries(
    strikes: np.ndarray,
    mids: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    *,
    is_call: bool,
) -> List[Tuple[float, float, float]]:
    """Drop mid prices that violate monotonicity (Jaeckel style, local checks)."""
    strikes = np.asarray(strikes, dtype=float)
    mids = np.asarray(mids, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    while len(strikes) >= 2:
        drop_index = _first_monotonic_violation(mids, is_call=is_call)
        if drop_index is None:
            break
        left = drop_index - 1 if drop_index - 1 >= 0 else None
        right = drop_index
        residual_left = _point_residual(strikes, mids, left) if left is not None else -1.0
        residual_right = _point_residual(strikes, mids, right)
        drop = left if (left is not None and residual_left >= residual_right) else right
        strikes = np.delete(strikes, drop)
        mids = np.delete(mids, drop)
        bid = np.delete(bid, drop)
        ask = np.delete(ask, drop)
    return [(float(k), float(b), float(a)) for k, b, a in zip(strikes, bid, ask)]


def _first_monotonic_violation(values: np.ndarray, *, is_call: bool) -> Optional[int]:
    """First index where monotonicity breaks (calls decrease, puts increase)."""
    if len(values) < 2:
        return None
    for i in range(1, len(values)):
        if is_call and values[i] >= values[i - 1]:
            return i
        if not is_call and values[i] <= values[i - 1]:
            return i
    return None


def _point_residual(strikes: np.ndarray, values: np.ndarray, index: int) -> float:
    if index < 0 or index >= len(values):
        return -1.0
    if 0 < index < len(values) - 1 and strikes[index + 1] != strikes[index - 1]:
        weight = (strikes[index] - strikes[index - 1]) / (strikes[index + 1] - strikes[index - 1])
        linear = values[index - 1] + weight * (values[index + 1] - values[index - 1])
        return float(abs(values[index] - linear))
    if index > 0:
        return float(abs(values[index] - values[index - 1]))
    if index + 1 < len(values):
        return float(abs(values[index] - values[index + 1]))
    return 0.0


# ------------------------------------------------------------------- forward
def _future_forward(snapshot: RawSnapshot, expiry: datetime) -> Optional[float]:
    key = expiry.date().isoformat()
    price = snapshot.future_price_by_expiry.get(key)
    if price is None or price <= 0.0:
        return None
    return float(price)


def _implied_forward(
    call_quotes: Sequence[Tuple[float, float, float]],
    put_quotes: Sequence[Tuple[float, float, float]],
    discount_factor: float,
) -> Tuple[Optional[float], str]:
    """Imply the forward from put-call parity (edslib single-strike mode)."""
    call_strikes = np.asarray([row[0] for row in call_quotes], dtype=float)
    call_mids = np.asarray([0.5 * (row[1] + row[2]) for row in call_quotes], dtype=float)
    put_strikes = np.asarray([row[0] for row in put_quotes], dtype=float)
    put_mids = np.asarray([0.5 * (row[1] + row[2]) for row in put_quotes], dtype=float)

    common = np.intersect1d(call_strikes, put_strikes)
    if len(common) > 0:
        c_index = np.searchsorted(call_strikes, common)
        p_index = np.searchsorted(put_strikes, common)
        diff = np.abs(call_mids[c_index] - put_mids[p_index])
        pick = int(np.argmin(diff))
        strike = float(common[pick])
        forward = strike + (call_mids[c_index[pick]] - put_mids[p_index[pick]]) / discount_factor
        return float(forward), "parity_single_strike"

    # No common strike: minimise the call/put vol difference of the closest pair.
    c_pick, p_pick = _closest_pair(call_strikes, put_strikes)
    if c_pick is None:
        return None, "parity_no_overlap"
    k_call, k_put = float(call_strikes[c_pick]), float(put_strikes[p_pick])
    guess = 0.5 * (k_call + k_put) + (
        call_mids[c_pick] - put_mids[p_pick]
    ) / discount_factor
    if not np.isfinite(guess) or guess <= 0.0:
        return None, "parity_no_overlap"

    def _vol_diff(forward_value: float) -> np.ndarray:
        call_vol = implied_vol_jaeckel(
            call_mids[c_pick] / discount_factor, forward_value, k_call, 1.0, "call"
        )
        put_vol = implied_vol_jaeckel(
            put_mids[p_pick] / discount_factor, forward_value, k_put, 1.0, "put"
        )
        if call_vol is None or put_vol is None:
            return np.asarray([1.0])
        return np.asarray([call_vol - put_vol])

    try:
        result = least_squares(_vol_diff, guess, bounds=(guess * 0.5, guess * 2.0))
        forward = float(result.x[0])
    except Exception:
        forward = float(guess)
    return forward, "parity_no_overlap"


def _closest_pair(
    call_strikes: np.ndarray,
    put_strikes: np.ndarray,
) -> Tuple[Optional[int], Optional[int]]:
    if len(call_strikes) == 0 or len(put_strikes) == 0:
        return None, None
    if call_strikes[0] >= put_strikes[-1]:
        return 0, len(put_strikes) - 1
    distance = np.abs(call_strikes[:, None] - put_strikes[None, :])
    row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return int(row), int(column)


# --------------------------------------------------------------- OTM and vols
def _select_otm_quotes(
    call_quotes: Sequence[Tuple[float, float, float]],
    put_quotes: Sequence[Tuple[float, float, float]],
    forward: float,
) -> List[Tuple[float, str, float, float]]:
    """Put quotes below the forward, call quotes at/above it (edslib rule)."""
    rows: List[Tuple[float, str, float, float]] = []
    for strike, bid, ask in put_quotes:
        if strike < forward:
            rows.append((float(strike), "put", float(bid), float(ask)))
    for strike, bid, ask in call_quotes:
        if strike >= forward:
            rows.append((float(strike), "call", float(bid), float(ask)))
    rows.sort(key=lambda row: row[0])
    return rows


def _imply_volatilities(
    rows: Sequence[Tuple[float, str, float, float]],
    forward: float,
    tau: float,
    discount_factor: float,
):
    strikes: List[float] = []
    option_types: List[str] = []
    bid_vols: List[float] = []
    ask_vols: List[float] = []
    for strike, option_type, bid, ask in rows:
        bid_vol = implied_vol_jaeckel(
            bid / discount_factor, forward, strike, tau, option_type
        )
        ask_vol = implied_vol_jaeckel(
            ask / discount_factor, forward, strike, tau, option_type
        )
        if bid_vol is None or ask_vol is None:
            continue
        if ask_vol < bid_vol:
            bid_vol, ask_vol = ask_vol, bid_vol
        strikes.append(float(strike))
        option_types.append(option_type)
        bid_vols.append(float(bid_vol))
        ask_vols.append(float(ask_vol))
    if len(strikes) == 0:
        return None
    mid_vols = [0.5 * (b + a) for b, a in zip(bid_vols, ask_vols)]
    return (
        np.asarray(strikes, dtype=float),
        option_types,
        np.asarray(mid_vols, dtype=float),
        np.asarray(bid_vols, dtype=float),
        np.asarray(ask_vols, dtype=float),
    )


# ------------------------------------------------------------------ arbitrage
def _eliminate_arbitrage_jaeckel(
    strikes: np.ndarray,
    mid_vols: np.ndarray,
    bid_vols: np.ndarray,
    ask_vols: np.ndarray,
    forward: float,
    tau: float,
):
    """Jaeckel "Clamping Down on Arbitrage": drop vertical/butterfly violators."""
    strikes = np.asarray(strikes, dtype=float)
    mid_vols = np.asarray(mid_vols, dtype=float)
    bid_vols = np.asarray(bid_vols, dtype=float)
    ask_vols = np.asarray(ask_vols, dtype=float)
    if len(strikes) <= 2:
        return strikes, mid_vols, bid_vols, ask_vols

    call_prices = black_price(forward, strikes, tau, mid_vols, 1.0, "call")
    call_prices = np.atleast_1d(np.asarray(call_prices, dtype=float))
    put_prices = black_price(forward, strikes, tau, mid_vols, 1.0, "put")
    put_prices = np.atleast_1d(np.asarray(put_prices, dtype=float))

    # left wing: put/K must be decreasing in K (zero probability mass at K=0)
    left_index: List[int] = []
    index = 0
    while index + 1 <= len(strikes) - 1:
        if put_prices[index] / strikes[index] >= put_prices[index + 1] / strikes[index + 1]:
            left_index.append(index)
        index += 1
    strikes = np.delete(strikes, left_index)
    mid_vols = np.delete(mid_vols, left_index)
    bid_vols = np.delete(bid_vols, left_index)
    ask_vols = np.delete(ask_vols, left_index)
    put_prices = np.delete(put_prices, left_index)
    call_prices = np.delete(call_prices, left_index)

    # right wing: call prices must be decreasing in K
    right_index: List[int] = []
    index = len(strikes) - 1
    while index - 1 >= 0:
        if call_prices[index] >= call_prices[index - 1]:
            right_index.append(index)
        index -= 1
    strikes = np.delete(strikes, right_index)
    mid_vols = np.delete(mid_vols, right_index)
    bid_vols = np.delete(bid_vols, right_index)
    ask_vols = np.delete(ask_vols, right_index)
    put_prices = np.delete(put_prices, right_index)
    call_prices = np.delete(call_prices, right_index)

    # interior: monotonicity and butterfly
    interior_index: List[int] = []
    index = 1
    while index + 1 <= len(strikes) - 1:
        if strikes[index] < forward and put_prices[index] >= put_prices[index + 1]:
            interior_index.append(index)
            index += 1
            continue
        if strikes[index] > forward and call_prices[index] >= call_prices[index - 1]:
            interior_index.append(index)
            index += 1
            continue
        put_butterfly = (
            put_prices[index - 1] / (strikes[index] - strikes[index - 1])
            - put_prices[index] / (strikes[index] - strikes[index - 1])
            - put_prices[index] / (strikes[index + 1] - strikes[index])
            + put_prices[index + 1] / (strikes[index + 1] - strikes[index])
        )
        call_butterfly = (
            call_prices[index - 1] / (strikes[index] - strikes[index - 1])
            - call_prices[index] / (strikes[index] - strikes[index - 1])
            - call_prices[index] / (strikes[index + 1] - strikes[index])
            + call_prices[index + 1] / (strikes[index + 1] - strikes[index])
        )
        if max(put_butterfly, call_butterfly) < 0:
            interior_index.append(index)
        index += 1

    strikes = np.delete(strikes, interior_index)
    mid_vols = np.delete(mid_vols, interior_index)
    bid_vols = np.delete(bid_vols, interior_index)
    ask_vols = np.delete(ask_vols, interior_index)
    return strikes, mid_vols, bid_vols, ask_vols


# -------------------------------------------------------------------- weights
def _generate_default_weights(
    forward: float,
    tau: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    bid_vols: np.ndarray,
    ask_vols: np.ndarray,
    weight_mode: str,
) -> np.ndarray:
    """edslib ``_generate_default_weights`` (vega weighting by default)."""
    mode = str(weight_mode or "vega").strip().lower()
    if mode == "equal":
        return np.ones_like(vols, dtype=float)
    if mode == "spread":
        return 1.0 / np.maximum(np.abs(ask_vols - bid_vols), 1.0e-8)
    if mode == "atm_vega":
        fwd_index = int(np.searchsorted(strikes, forward))
        atm_vol = float(vols[min(fwd_index, len(vols) - 1)])
        sigma = np.full_like(vols, atm_vol * np.sqrt(tau), dtype=float)
    else:  # vega
        sigma = np.asarray(vols, dtype=float) * np.sqrt(tau)
    sigma = np.maximum(sigma, 1.0e-8)
    ln_moneyness = np.log(np.asarray(strikes, dtype=float) / float(forward))
    d1 = -ln_moneyness / sigma + 0.5 * sigma
    weights = norm.pdf(d1) * float(forward) * np.sqrt(tau)
    return np.maximum(np.asarray(weights, dtype=float), 1.0e-12)


__all__ = ["SliceData", "build_slice", "prepare_slices"]
