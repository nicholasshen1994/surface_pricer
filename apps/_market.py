"""Shared market construction and engine plumbing for the quoting entry points.

Every quoting entry point must see exactly the same market for the same fit run,
so the run -> :class:`MarketState` mapping (spot / rate overrides, curve runs,
calendar, anchor warnings) lives here instead of one entry point importing the
private helper of another.  The same goes for the local-vol table cache and the
flag -> :class:`RiskSettings` mapping: ``price-json`` and ``autocall-pricer`` must
read ``--pde-nodes`` / ``--no-local-vol-cache`` the same way, or one flag would
mean two different runs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..core.daycount import to_datetime
from ..core.market import MarketState
from ..io.curve_files import curve_valuation_date, load_borrow_curve, load_ir_curve
from ..io.curve_runs import BORROW_CURVE, IR_CURVE, NO_CURVE, resolve_curve_path
from ..io.fit_runs import FitRun, bare_code
from ..io.local_vol_cache import LocalVolFileCache
from ..io.serialization import market_from_surface
from ..pricing.models.localvol import make_table_cache
from ..pricing.results import RiskSettings


def _curve_spec(value: Any) -> Optional[str]:
    """The curve value as given: ``None`` / ``none`` mean "no curve, flat rate".

    Only that one spelling is honoured.  ``flat`` / ``off`` / ``no`` used to be
    accepted as well - four ways to say the same thing, and one more way to
    mistype it into something that looks configured but is not.
    """
    text = str(value if value is not None else "").strip()
    if not text or text.lower() == NO_CURVE:
        return None
    return text


def resolved_curves(
    args: argparse.Namespace,
    *,
    index: Any = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """The ``(ir, borrow)`` curve files a market build resolves - resolved in one place.

    ``index`` picks the **borrow** curve, which is filed per index (two indices are
    two borrow curves): pass the index being priced - normally the fit run's
    ``underlying`` - and the newest run *for it* is used, never another index's.
    The interest-rate curve is one curve for everything and ignores it.

    :func:`market_from_run` uses exactly this, so a caller that needs the same
    answer - the local-vol fingerprint names the curves - cannot disagree with it.
    """
    output_root = getattr(args, "output_root", None)
    return (
        resolve_curve_path(
            IR_CURVE,
            _curve_spec(getattr(args, "ir_curve", None)),
            output_root=output_root,
        ),
        resolve_curve_path(
            BORROW_CURVE,
            _curve_spec(getattr(args, "borrow_curve", None)),
            output_root=output_root,
            index=index,
        ),
    )


def local_vol_fingerprint(run: FitRun, market, args: argparse.Namespace) -> Dict[str, Any]:
    """The inputs a local-vol table is built from - the on-disk cache's key.

    The four inputs of a local-vol table - the fitted surface (the **vol**), the
    **rate** curve, the **borrow** curve (resolved per index, so another index's is
    a different file) and the **spot** anchor - plus the run name and the valuation
    date.  Each curve/surface enters as name + **content hash**, so any of the four
    changing rebuilds the table (see
    :class:`surface_pricer.pricing.models.localvol.LocalVolCache`); the flat
    fallbacks replace the curves that are switched off.  The store keeps this next
    to the coefficients, so "is this table still current, and what exactly produced
    it?" is answered by reading the file.
    """
    ir_file, borrow_file = resolved_curves(args, index=run.underlying)
    return {
        "run": run.name,
        "index": bare_code(run.underlying),
        "surface": _file_identity(run.surface_path),
        "rate_curve": _file_identity(ir_file)
        or {"flat": _flat_value(args.rate, run.rate)},
        "borrow_curve": _file_identity(borrow_file)
        or {"flat": _flat_value(getattr(args, "borrow", None), run.borrow)},
        "valuation_date": to_datetime(market.valuation_date).isoformat(sep=" "),
        "spot_anchor": float(market.spot),
    }


def _file_identity(path) -> Optional[Dict[str, str]]:
    """``{"file": name, "sha1": content hash}`` - or ``None`` for no file."""
    if path is None:
        return None
    target = Path(path)
    try:
        digest = hashlib.sha1(target.read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - unreadable file
        digest = ""
    return {"file": target.name, "sha1": digest}


def _flat_value(flag, fallback) -> float:
    """The flat rate/borrow a curve file would replace (``None`` -> the run's)."""
    if flag is not None:
        return float(flag)
    return float(fallback or 0.0)


def market_from_run(run: FitRun, args: argparse.Namespace) -> MarketState:
    """Build the market of ``run``, honouring the CLI overrides on ``args``.

    ``--ir-curve`` / ``--borrow-curve`` take ``latest`` (the newest run under the
    output root), a path taken as given, or ``none`` for the flat rate / borrow
    (:func:`surface_pricer.io.curve_runs.resolve_curve_path`).  Every entry point
    goes through here, so the same run and the same curve flags give the same
    market whichever command prices the payload.
    """
    payload = run.surface_payload()
    spot = args.spot if args.spot is not None else run.spot
    rate = args.rate if args.rate is not None else run.rate
    # ``--borrow`` unset means "the run's", exactly like ``--spot`` / ``--rate``: a
    # run that recorded none prices on a zero flat borrow.  Either way the borrow
    # *curve*, when one is given, replaces this number.
    borrow_flag = getattr(args, "borrow", None)
    borrow = (
        float(borrow_flag) if borrow_flag is not None else float(run.borrow or 0.0)
    )
    valuation_date = args.valuation_date or run.valuation_datetime

    # the borrow curve is filed per index: the run's underlying is the index it
    # prices (000852 for a MO fit), so that is the entry it follows
    ir_curve_file, borrow_curve_file = resolved_curves(args, index=run.underlying)
    rate_curve = load_ir_curve(ir_curve_file) if ir_curve_file else None
    borrow_curve = load_borrow_curve(borrow_curve_file) if borrow_curve_file else None
    warn_curve_anchor(rate_curve, valuation_date, "--ir-curve", ir_curve_file)
    warn_curve_anchor(borrow_curve, valuation_date, "--borrow-curve", borrow_curve_file)

    return market_from_surface(
        payload,
        spot=spot,
        rate=rate,
        borrow=borrow,
        valuation_date=valuation_date,
        calendar_file=args.calendar_file,
        rate_curve=rate_curve,
        borrow_curve=borrow_curve,
    )


# ------------------------------------------------------------------- engines
def table_cache_for(args: argparse.Namespace, market, run: FitRun):
    """The local-vol table provider for one run - shared by every market it meets.

    One cache per run means **one table**: the greeks' bumped markets reuse it, a
    spot ladder - a fresh market per rung - reuses it too (the anchor is pinned to
    the *base* spot, the convention a risk run uses), and a coupon search
    (``autocall-pricer``) evaluates every trial on the same coefficients.  With the
    store on, an identical table from an earlier run is loaded instead of built;
    ``--no-local-vol-cache`` turns every read into a miss and every write into a
    no-op.

    The store is filed under the run's **index** (``local_vol/<index>/``): the table
    is Dupire of that index's surface, discounted off that index's borrow curve, so
    two indices never share a folder - let alone a file.
    """
    store = LocalVolFileCache(
        getattr(args, "output_root", None),
        index=run.underlying,  # the table is filed per index (local_vol/<index>/)
        enabled=bool(getattr(args, "local_vol_cache", True)),
    )
    return make_table_cache(
        spot_anchor=float(market.spot),
        store=store,
        fingerprint=local_vol_fingerprint(run, market, args),
    )


def note_table_cache(cache) -> None:
    """One stderr line saying where the table came from (keeps ``--json`` clean)."""
    store = getattr(cache, "store", None)
    if store is None:
        return
    print(store.describe(cache), file=sys.stderr)


def risk_settings(
    args: argparse.Namespace, selection: Tuple[str, ...], **overrides
) -> RiskSettings:
    """Engine knobs and the Greek selection, exactly like the other entry points.

    ``overrides`` is for the caller's own extras - the coupon solver pins the
    discounting and the time grid itself while it searches.
    """
    changes: Dict[str, Any] = {"greeks": selection}
    if getattr(args, "paths", None) is not None:
        # one knob for the price and the bumps: the paired common random numbers
        # make a smaller greek path count pointless (it only adds noise)
        changes["mc_paths"] = int(args.paths)
    if getattr(args, "seed", None) is not None:
        changes["mc_seed"] = int(args.seed)
    if getattr(args, "pde_nodes", None) is not None:
        changes["pde_nodes"] = int(args.pde_nodes)
    if getattr(args, "pde_theta", None) is not None:
        changes["pde_theta"] = float(args.pde_theta)
    if getattr(args, "theta_days", None) is not None:
        changes["theta_days"] = int(args.theta_days)
    if getattr(args, "full_bucket_grid", False):
        # one bump pair per pillar: the pre-2026-10 grid, finer and slower
        changes["bucket_group_after"] = None
    changes.update(overrides)
    return RiskSettings(**changes)


def warn_underlying_mismatch(run: FitRun, wanted: str, flag: str = "--fit") -> None:
    """Warn (stderr) when a payload is priced on a surface of another underlying.

    ``--fit latest`` picks the run of the payload's own index, so this can only
    fire when a run was named explicitly - exactly the case where a 000852 quote
    would otherwise be priced on a 000300 surface without a word.  Both sides are
    compared by bare code (``000852.SH`` == ``000852``).
    """
    if not wanted or not run.underlying:
        return
    left, right = bare_code(wanted), bare_code(run.underlying)
    if left and right and left != right:
        print(
            "WARNING: {} is a {} surface but the payload is for {} - the barriers "
            "and the spot live in different spaces".format(flag, run.underlying, wanted),
            file=sys.stderr,
        )


def warn_curve_anchor(curve, valuation_date, flag: str, path=None) -> None:
    """Warn (stderr, so ``--json`` stays clean) on a curve/run date mismatch.

    The message names the **file** that was used, so a quote that warns can be
    traced back to the exact run without re-running the resolution.
    """
    if curve is None:
        return
    anchor = curve_valuation_date(curve)
    if anchor is None:
        return
    reference = to_datetime(valuation_date).date()
    if (anchor - reference).days != 0:
        print(
            "WARNING: {} ({}) was built for {} but the valuation date is {}".format(
                flag,
                Path(path).name if path else "?",
                anchor.isoformat(),
                reference.isoformat(),
            ),
            file=sys.stderr,
        )


__all__ = ["market_from_run", "warn_curve_anchor"]
