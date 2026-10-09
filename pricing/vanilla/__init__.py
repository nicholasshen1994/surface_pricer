"""European vanilla product: contract, resolved spec, analytic pricer and risk.

The layout mirrors the exotic side: ``contract.py`` holds the **raw** terms (a
strike that may be a percentage, a tenor someone typed), ``spec.py`` resolves them
into ``VanillaSpec`` - absolute strike, absolute expiry and the market mapping
Black-76 needs - and that spec is what the pricer and the Greeks consume.  It is
also the JSON layer (``to_dict`` / ``from_dict`` / ``rebased``), so a quote can be
exported, hand-edited and priced again.

The pricer is analytic (Black on the fitted EDS SABR surface), so this package
carries no engine of its own; everything shared with the exotic products lives in
:mod:`surface_pricer.pricing.results` and :mod:`surface_pricer.pricing.risk`.
"""

from .contract import VanillaContract
from .greeks import calculate_greeks, calculate_greeks_spec, convert_bucketed_delta
from .pricer import VanillaPricer, price_vanilla, price_vanilla_spec
from .spec import VanillaSpec, resolve_spec

__all__ = [
    "VanillaContract",
    "VanillaPricer",
    "VanillaSpec",
    "calculate_greeks",
    "calculate_greeks_spec",
    "convert_bucketed_delta",
    "price_vanilla",
    "price_vanilla_spec",
    "resolve_spec",
]
