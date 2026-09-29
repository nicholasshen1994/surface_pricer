"""Load the generated curve files (``ir_curve.json`` / ``borrow_curve.json``).

Both files are produced by ``python -m surface_pricer build-ir-curve`` /
``build-borrow-curve`` (see :mod:`surface_pricer.core.ir_curve` and
:mod:`surface_pricer.core.borrow_curve`).  The pricing entry points accept them
through ``--ir-curve`` / ``--borrow-curve`` and receive a
:class:`surface_pricer.core.curves.PiecewiseRateCurve` - linear on zero rates
with the same time scale the curve was built with - so the pricing runtime
never needs QuantLib.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.borrow_curve import BorrowCurvePillars
from ..core.curves import PiecewiseRateCurve
from ..core.daycount import to_date
from ..core.ir_curve import IRCurvePillars

#: Default output directory of ``build-ir-curve`` / ``build-borrow-curve``.
_PACKAGE_OUTPUT = Path(__file__).resolve().parents[1] / "output"


def load_ir_curve(path: Union[str, Path]) -> PiecewiseRateCurve:
    """Read an ``ir_curve.json`` (``build-ir-curve``) as a pricing curve.

    Relative paths resolve against the cwd first and then against the package
    ``output`` directory, so the plain spelling ``output/ir_curve.json`` works
    from any working directory.
    """
    payload = _read_payload(path, kind="ir_curve.json", required=("curve_name",))
    return IRCurvePillars.from_dict(payload).to_piecewise_curve()


def load_borrow_curve(path: Union[str, Path]) -> PiecewiseRateCurve:
    """Read a ``borrow_curve.json`` (``build-borrow-curve``) as a pricing curve.

    Relative paths resolve against the cwd first and then against the package
    ``output`` directory, so the plain spelling ``output/borrow_curve.json``
    works from any working directory.
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


def _locate_curve_file(path: Union[str, Path]) -> Tuple[Optional[Path], List[Path]]:
    """Find a curve file: as given first, then in the package ``output`` dir.

    Returns the resolved path (``None`` when nothing matched) plus every
    candidate that was tried, for the error message.
    """
    target = Path(path)
    tried: List[Path] = [target]
    if target.is_file():
        return target, tried
    if not target.is_absolute():
        fallback = _PACKAGE_OUTPUT / target.name
        if fallback != target:
            tried.append(fallback)
            if fallback.is_file():
                return fallback, tried
    return None, tried


def _read_payload(
    path: Union[str, Path],
    *,
    kind: str,
    required: tuple,
) -> Dict[str, Any]:
    target, tried = _locate_curve_file(path)
    if target is None:
        raise ValueError(
            "{} not found (tried {}); build it with "
            "'python -m surface_pricer build-{}'".format(
                Path(path),
                ", ".join(str(candidate) for candidate in tried),
                "ir-curve" if "ir" in kind else "borrow-curve",
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
