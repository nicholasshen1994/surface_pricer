"""Contract definitions priced by the :mod:`surface_pricer.pricing` layer.

Exotic products (barrier / autocallable / accumulator) are added in a later
phase as extra contract classes plus an entry in
:mod:`surface_pricer.pricing.exotics`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.daycount import DateLike, to_datetime


#: How a strike may be expressed - exactly one spelling each (2026-10: the
#: synonyms ``spot_percentage`` / ``forward_relative`` / ... are refused, and an
#: unknown name is an error instead of a silent fall-back to ``absolute``).
STRIKE_TYPES = ("absolute", "percentage", "fwd_percentage")


def resolve_strike_type(value: Optional[str]) -> str:
    """Normalise ``strike_type``; raise on anything but the three names."""
    text = str(value or "absolute").strip().lower()
    if text not in STRIKE_TYPES:
        raise ValueError(
            "strike_type must be one of {} (no aliases), got {!r}".format(
                ", ".join(STRIKE_TYPES), value
            )
        )
    return text


@dataclass
class VanillaContract:
    """Raw (pre-resolution) terms of one European vanilla option.

    ``strike`` is interpreted through ``strike_type`` (``absolute``, a
    ``percentage`` of the spot, or a ``fwd_percentage`` of the forward);
    :func:`...vanilla.spec.resolve_spec` turns the pair into the absolute strike
    the pricer works with.
    """

    expiry: DateLike
    strike: float
    option_type: str = "call"
    notional: float = 1.0
    strike_type: str = "absolute"
    #: Bookkeeping tag: carried into the resolved spec so a payload says what it
    #: belongs to (nothing in the pricing depends on it).
    underlying: str = ""

    def __post_init__(self):
        self.expiry = to_datetime(self.expiry)
        option_type = self.option_type.lower()
        if option_type in {"c", "call"}:
            self.option_type = "call"
        elif option_type in {"p", "put"}:
            self.option_type = "put"
        else:
            raise ValueError("option_type must be call or put")
        # checked once, at construction: an unknown name must not price as absolute
        self.strike_type = resolve_strike_type(self.strike_type)

    def absolute_strike(self, spot: float, fwd: float) -> float:
        if self.strike_type == "percentage":
            return float(self.strike * spot)
        if self.strike_type == "fwd_percentage":
            return float(self.strike * fwd)
        return float(self.strike)


__all__ = ["STRIKE_TYPES", "VanillaContract", "resolve_strike_type"]
