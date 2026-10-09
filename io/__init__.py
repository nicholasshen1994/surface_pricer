"""JSON / dict adapters bridging files and the domain containers.

This is a cross-cutting adapter layer: it may depend on ``core``,
``marketdata``, ``fitting`` and ``pricing``, but nothing depends on it except
``apps``.
"""

from .curve_runs import (
    BORROW_CURVE,
    CURVE_KINDS,
    IR_CURVE,
    NO_CURVE,
    curve_details,
    curve_file_name,
    curve_root,
    curve_run_name,
    latest_curve_path,
    list_curve_runs,
    read_curve_index,
    record_curve_run,
    resolve_curve_path,
)
from .fit_runs import (
    FitRun,
    bare_code,
    default_output_root,
    fit_run_root,
    latest_run_path,
    list_runs,
    read_index,
    read_latest_map,
    record_fit_run,
    resolve_run,
)
from .local_vol_cache import (
    LocalVolFileCache,
    local_vol_root,
    table_digest,
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
    "BORROW_CURVE",
    "CURVE_KINDS",
    "FitRun",
    "IR_CURVE",
    "NO_CURVE",
    "bare_code",
    "contract_from_dict",
    "curve_details",
    "curve_file_name",
    "curve_from_dict",
    "curve_root",
    "curve_run_name",
    "default_output_root",
    "fit_run_root",
    "latest_curve_path",
    "latest_run_path",
    "list_curve_runs",
    "list_runs",
    "load_json",
    "local_vol_root",
    "LocalVolFileCache",
    "table_digest",
    "market_from_dict",
    "market_from_surface",
    "quote_slices_from_dict",
    "read_curve_index",
    "read_index",
    "read_latest_map",
    "record_curve_run",
    "record_fit_run",
    "resolve_curve_path",
    "resolve_run",
    "surface_from_dict",
]
