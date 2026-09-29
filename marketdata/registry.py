"""Supported underlyings: how to request and parse each one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .listed_contracts import EXCHANGE_CFFEX, EXCHANGE_SSE, EXCHANGE_SZSE


@dataclass(frozen=True)
class UnderlyingSpec:
    """How to request and parse one supported underlying."""

    underlying: str
    kind: str  # "index" | "etf"
    exchange_id: str
    code_mode: str  # "cffex" | "etf"
    index_ticker: Optional[str] = None
    spot_ticker: Optional[str] = None
    future_prefix: Optional[str] = None
    option_prefix: Optional[str] = None

    @property
    def etf_exchange(self) -> str:
        if self.code_mode != "etf":
            return EXCHANGE_CFFEX
        return EXCHANGE_SSE if str(self.exchange_id) == "0" else EXCHANGE_SZSE


UNDERLYING_SPECS: Dict[str, UnderlyingSpec] = {
    "MO": UnderlyingSpec(
        underlying="MO",
        kind="index",
        exchange_id="F",
        code_mode="cffex",
        index_ticker="000852.SH",
        future_prefix="IM",
    ),
    "IO": UnderlyingSpec(
        underlying="IO",
        kind="index",
        exchange_id="F",
        code_mode="cffex",
        index_ticker="000300.SH",
        future_prefix="IF",
    ),
    "HO": UnderlyingSpec(
        underlying="HO",
        kind="index",
        exchange_id="F",
        code_mode="cffex",
        index_ticker="000016.SH",
        future_prefix="IH",
    ),
    "510500": UnderlyingSpec(
        underlying="510500",
        kind="etf",
        exchange_id="0",
        code_mode="etf",
        spot_ticker="510500.SH",
        option_prefix="510500",
    ),
    "588000": UnderlyingSpec(
        underlying="588000",
        kind="etf",
        exchange_id="0",
        code_mode="etf",
        spot_ticker="588000.SH",
        option_prefix="588000",
    ),
    "159915": UnderlyingSpec(
        underlying="159915",
        kind="etf",
        exchange_id="1",
        code_mode="etf",
        spot_ticker="159915.SZ",
        option_prefix="159915",
    ),
}


def get_underlying_spec(underlying: str) -> UnderlyingSpec:
    key = str(underlying or "").strip().upper()
    spec = UNDERLYING_SPECS.get(key)
    if spec is None:
        raise KeyError(
            "Unsupported underlying {!r}; known underlyings: {}".format(
                underlying, sorted(UNDERLYING_SPECS)
            )
        )
    return spec


__all__ = ["UNDERLYING_SPECS", "UnderlyingSpec", "get_underlying_spec"]
