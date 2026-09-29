"""JSON / dict adapters bridging files and the domain containers.

This is a cross-cutting adapter layer: it may depend on ``core``,
``marketdata``, ``fitting`` and ``pricing``, but nothing depends on it except
``apps``.
"""

from .fit_runs import (
    FitRun,
    default_output_root,
    list_runs,
    read_index,
    record_fit_run,
    resolve_run,
)
from .serialization import (
    contract_from_dict,
    curve_from_dict,
    load_json,
    market_from_dict,
    market_from_surface,
    quote_slices_from_dict,
    surface_from_dict,
)

__all__ = [
    "FitRun",
    "contract_from_dict",
    "curve_from_dict",
    "default_output_root",
    "list_runs",
    "load_json",
    "market_from_dict",
    "market_from_surface",
    "quote_slices_from_dict",
    "read_index",
    "record_fit_run",
    "resolve_run",
    "surface_from_dict",
]
