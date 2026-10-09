"""Load one curve **run** file (``output/ir_curve/ir_curve_<stamp>.json`` ...).

Both curve kinds are produced by ``python -m surface_pricer build-ir-curve`` /
``build-borrow-curve`` (see :mod:`surface_pricer.core.ir_curve` and
:mod:`surface_pricer.core.borrow_curve`) as timestamped runs; naming one
(``latest`` / a path / ``none``) is
:func:`surface_pricer.io.curve_runs.resolve_curve_path`'s job.  The pricing entry
points end up with a :class:`surface_pricer.core.curves.PiecewiseRateCurve` -
linear on zero rates with the same time scale the curve was built with - so the
pricing runtime never needs QuantLib.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.borrow_curve import BorrowCurvePillars
from ..core.curves import PiecewiseRateCurve
from ..core.daycount import to_date
from ..core.ir_curve import IRCurvePillars


def load_ir_curve(path: Union[str, Path]) -> PiecewiseRateCurve:
    """Read an ``ir_curve`` run (``build-ir-curve``) as a pricing curve.

    ``path`` is a **file** - resolve ``latest`` / ``none`` first with
    :func:`surface_pricer.io.curve_runs.resolve_curve_path` (what the CLI flags
    do).  A missing file is an error naming it: nothing is searched for.
    """
    payload = _read_payload(path, kind="ir_curve.json", required=("curve_name",))
    return IRCurvePillars.from_dict(payload).to_piecewise_curve()


def load_borrow_curve(path: Union[str, Path]) -> PiecewiseRateCurve:
    """Read a ``borrow_curve`` run (``build-borrow-curve``) as a pricing curve.

    Same contract as :func:`load_ir_curve`: the path is taken as given.
    """
    payload = _read_payload(
        path, kind="borrow_curve.json", required=("forward_source", "observed")
    )
    return BorrowCurvePillars.from_dict(payload).to_piecewise_curve()


def curve_valuation_date(curve: Any) -> Optional[date]:
    """Valuation date (anchor) of a loaded curve, when it exposes one."""
    anchor = getattr(curve, "anchor", None)
    if anchor is None:
        return None
    if isinstance(anchor, datetime):
        return anchor.date()
    return to_date(anchor)


def _read_payload(
    path: Union[str, Path],
    *,
    kind: str,
    required: tuple,
) -> Dict[str, Any]:
    """Read one curve file - taken **as given**, no fallback locations."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise ValueError(
            "{} not found; build it with '{}', or pass 'latest' (the newest run) "
            "or 'none' (the flat rate)".format(
                target,
                "python -m surface_pricer build-ir-curve"
                if "ir" in kind
                else "python -m surface_pricer build-borrow-curve",
            )
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("{} is not valid JSON: {}".format(target, error)) from error
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(target))
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "{} does not look like a {} (missing {})".format(
                target, kind, ", ".join(repr(key) for key in missing)
            )
        )
    return payload


__all__ = ["curve_valuation_date", "load_borrow_curve", "load_ir_curve"]
