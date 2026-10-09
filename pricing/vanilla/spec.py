"""Resolved vanilla terms - the single input of the analytic pricer.

A :class:`VanillaContract` carries *raw* terms: a strike that may be a
percentage of the spot or of the forward, an expiry someone typed, a notional.
:func:`resolve_spec` turns them into a :class:`VanillaSpec` against one market
state - an absolute strike, an absolute expiry date, and the market mapping
Black-76 needs (forward, discount factor, vol time, the surface's anchor
forward).

Everything downstream works on the spec, which buys three things:

* **the strike is resolved once.**  A ``strike_type`` of ``percentage`` is
  turned into a number before any bump; a spot bump then measures a *fixed*
  strike instead of re-resolving it (the vanilla counterpart of the exotic
  engines' frozen barrier grid - otherwise a percentage strike would silently
  re-strike itself and the delta would carry the strike move);
* **it is the JSON payload.**  ``to_dict`` / ``from_dict`` export and read it, so
  a quote can be saved, hand-edited and priced again;
* **``rebased`` re-prices the same contract on another market**, which is what
  every bump in a Greek run asks for.

The spec carries the market mapping it was resolved on, but the **payload does
not**: ``to_dict`` writes the contract terms only (absolute strike, expiry, option
type, notional, how the strike was expressed), and ``from_dict`` rebuilds the
mapping - spot, forward, discount factor, vol time, the surface's anchor forward -
from the market you hand it.  The implied vol is not stored anywhere: it is a market
observation looked up live on the surface, so a vol bump moves it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from ...core.daycount import to_datetime
from ...core.market import MarketState
from .contract import VanillaContract, resolve_strike_type

#: Payload discriminator written by :meth:`VanillaSpec.to_dict`.
PAYLOAD_KIND = "vanilla_spec"


@dataclass(frozen=True)
class VanillaSpec:
    """Resolved (engine-ready) terms of one European vanilla option."""

    expiry_date: datetime
    #: Absolute strike the option is priced at (never a ratio).
    strike: float
    option_type: str = "call"
    notional: float = 1.0
    #: How the caller expressed the strike - provenance only, the payload is
    #: post-resolution: editing ``strike`` moves the option, editing these does
    #: not (they explain what was priced).
    strike_type: str = "absolute"
    strike_input: Optional[float] = None
    underlying: str = ""
    product_type: str = "vanilla"
    # ---- market mapping (rebuilt by ``from_dict`` / ``rebased``) -------------
    valuation_date: Optional[datetime] = None
    spot: float = 0.0
    forward: float = 0.0
    discount_factor: float = 1.0
    #: Vol time (``MarketState.year_fraction``: the EDS trading-day convention, the
    #: ``tau`` in ``vol^2 tau`` - *not* the act/365 accrual rho and coupons use).
    #: Kept on the spec because Black-76 needs it, never written to the payload.
    year_fraction: float = 0.0
    #: Forward off the surface's anchor spot - the SABR lookup needs it.
    initial_forward: float = 0.0

    # ------------------------------------------------------------ derived
    @property
    def is_expired(self) -> bool:
        if self.valuation_date is None:
            return False
        return self.expiry_date <= to_datetime(self.valuation_date)

    def rebased(self, market: MarketState) -> "VanillaSpec":
        """The same contract priced on another market state.

        Only the market mapping moves; the strike, the expiry and the option type
        are the trade and stay put - so a spot bump measures the sensitivity of a
        *fixed* strike.
        """
        return replace(
            self,
            valuation_date=to_datetime(market.valuation_date),
            spot=float(market.spot),
            forward=float(market.forward(self.expiry_date)),
            discount_factor=float(market.discount_factor(self.expiry_date)),
            year_fraction=float(market.year_fraction(self.expiry_date)),
            initial_forward=initial_forward(market, self.expiry_date),
        )

    # --------------------------------------------------------- JSON layer
    def to_dict(self) -> Dict[str, Any]:
        """The resolved option as a hand-editable payload - contract terms only.

        **One strike field**: ``strike`` is the absolute strike that was priced
        (``strike_type`` stays as provenance - a percentage strike says what it was
        a percentage *of*).  The raw input ratio is not in the payload any more
        (2026-10): it is not a term, and "the ratio it came from" and "the strike"
        would disagree the moment someone edits ``strike`` by hand.  The market
        mapping - spot, forward, discount factor, vol time, the surface's anchor
        forward - is deliberately out too: those are read from the live market at
        pricing time, so a payload written yesterday is priced on today's curves
        while the trade stays identical (see :meth:`from_dict`).
        """
        return {
            "kind": PAYLOAD_KIND,
            "version": 1,
            "underlying": self.underlying,
            "product_type": self.product_type,
            "option_type": self.option_type,
            "notional": float(self.notional),
            "valuation_date": _stamp(self.valuation_date),
            "expiry_date": _stamp(self.expiry_date),
            "strike": float(self.strike),
            "strike_type": self.strike_type,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], market: MarketState
    ) -> "VanillaSpec":
        """Rebuild a spec from :meth:`to_dict` output (or a whole ``--json`` quote).

        The payload is post-resolution: ``strike`` is used as written.  The market
        mapping is recomputed on ``market``, so a payload written yesterday is
        priced with today's curves while the trade stays identical.
        """
        payload = _unwrap(payload)
        if "strike_input" in payload:
            raise ValueError(
                "strike_input is not a payload field any more (2026-10): 'strike' is "
                "the only strike of a resolved option - delete the key (or regenerate "
                "the payload) and price it again"
            )
        expiry = to_datetime(payload["expiry_date"])
        valuation = to_datetime(market.valuation_date)
        if expiry <= valuation:
            raise ValueError(
                "contract already expired at {}: expiry is {}".format(
                    valuation.date(), expiry.date()
                )
            )
        spec = cls(
            expiry_date=expiry,
            strike=float(payload["strike"]),
            option_type=str(payload.get("option_type", "call")),
            notional=float(payload.get("notional", 1.0)),
            strike_type=resolve_strike_type(payload.get("strike_type")),
            # provenance only, and no longer carried by the payload: kept as ``None``
            # so a rebased spec has no ratio to print (the report falls back to the
            # strike itself)
            strike_input=None,
            underlying=str(payload.get("underlying", "")),
            product_type=str(payload.get("product_type", "vanilla")),
        )
        if spec.strike <= 0.0:
            raise ValueError("strike must be positive")
        return spec.rebased(market)


# ------------------------------------------------------------------- resolving
def initial_forward(market: MarketState, expiry: Any) -> float:
    """Forward off the surface's anchor spot (what the SABR lookup compares to).

    Without a surface there is nothing to anchor: the current forward is used, so
    the spec still resolves and the pricer raises its own "surface required"
    error when a vol is actually needed.
    """
    expiry = to_datetime(expiry)
    if market.surface is None:
        return float(market.forward(expiry))
    return float(market.forward(expiry, spot=market.surface.init_spot))


def resolve_spec(contract: VanillaContract, market: MarketState) -> VanillaSpec:
    """Resolve raw terms against ``market`` - the only place ``strike_type`` is read."""
    valuation = to_datetime(market.valuation_date)
    expiry = to_datetime(contract.expiry)
    if expiry <= valuation:
        raise ValueError(
            "contract already expired at {}: expiry is {}".format(
                valuation.date(), expiry.date()
            )
        )
    forward = float(market.forward(expiry))
    strike = float(contract.absolute_strike(market.spot, forward))
    if strike <= 0.0:
        raise ValueError("strike must be positive")

    return VanillaSpec(
        expiry_date=expiry,
        strike=strike,
        option_type=contract.option_type,
        notional=float(contract.notional),
        strike_type=str(contract.strike_type),
        strike_input=float(contract.strike),
        underlying=str(getattr(contract, "underlying", "") or "").upper(),
        valuation_date=valuation,
        spot=float(market.spot),
        forward=forward,
        discount_factor=float(market.discount_factor(expiry)),
        year_fraction=float(market.year_fraction(expiry)),
        initial_forward=initial_forward(market, expiry),
    )


def _stamp(value: Optional[datetime]) -> Optional[str]:
    """ISO stamp of a spec date (dates keep midnight, timestamps keep the time)."""
    return None if value is None else to_datetime(value).isoformat()


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a bare spec payload or a whole quote (``{"contract": ..}``).

    ``price-tool --json`` writes a quote *around* the resolved option, so an
    exported quote can be edited and fed straight back without stripping it.
    """
    if "expiry_date" not in payload and isinstance(payload.get("contract"), Mapping):
        return payload["contract"]
    return payload


__all__ = ["PAYLOAD_KIND", "VanillaSpec", "initial_forward", "resolve_spec"]
