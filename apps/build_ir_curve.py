"""Build the CNY FR007 interest-rate curve from the desk rate export.

``python -m surface_pricer build-ir-curve`` reads ``data/interest_rate.csv``
(FR007 fixing + ``FR007S<tenor>.IR`` IRS par quotes), bootstraps the curve with
QuantLib exactly like edslib's ``CNY-FR007`` curve (see
:mod:`surface_pricer.core.ir_curve`) and writes the pillars as JSON.  The
pricing layer then consumes that JSON through
:class:`surface_pricer.core.curves.PiecewiseRateCurve`, so QuantLib is only
needed for this build step.

Every run writes a **new** file under ``<output-root>/ir_curve/``
(``ir_curve_<YYYYmmdd_HHMMSS>.json``) and points ``latest.json`` at it - nothing
is overwritten, so a quote can always be traced back to the curve it used
(:mod:`surface_pricer.io.curve_runs`).  ``--out FILE`` bypasses the run
bookkeeping and writes exactly that file.
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # IDE "Run" launches the file directly, with no package context; put the
    # repository root on the path and hand control to the package module
    # (``python -m ...`` takes this branch never).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from surface_pricer.apps.build_ir_curve import main

    raise SystemExit(main())

import argparse
from pathlib import Path
from typing import Iterable, Optional

from ..core.ir_curve import build_fr007_curve
from ..io.curve_runs import (
    IR_CURVE,
    LATEST_NAME,
    curve_details,
    curve_root,
    record_curve_run,
)
from ..marketdata.rate_inputs import SAMPLE_CSV, parse_interest_rate_csv
from ._common import reporter_for


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    reporter = reporter_for(args.quiet)

    inputs = parse_interest_rate_csv(
        args.rate_file,
        drop_constant_columns=not args.keep_constant_columns,
    )
    reporter(
        "rate inputs | {} | FR007 {:.4f} | {} IRS pillar(s)".format(
            inputs.valuation_date.isoformat(),
            (inputs.fr007 or 0.0) * 100.0,
            len(inputs.ir_swap),
        )
    )
    for label, reason in sorted(inputs.dropped.items()):
        reporter("  dropped {:<6} {}".format(label, reason))
    for note in inputs.notes:
        reporter("  note: {}".format(note))

    pillars = build_fr007_curve(
        inputs.valuation_date,
        inputs.ir_swap,
        fr007=inputs.fr007,
        source=str(args.rate_file),
    )

    lines = [
        "curve    : {} | valuation {} | day counter {}".format(
            pillars.curve_name, pillars.valuation_date.isoformat(), pillars.day_counter
        ),
        "pillars  :",
    ]
    for tenor, pillar_date, days, zero in zip(
        pillars.tenors, pillars.pillar_dates, pillars.pillar_days, pillars.zero_rates
    ):
        lines.append(
            "  {:<4} {:>10} {:>6}d  zero {:>8.4f}%  par {:>8.4f}%".format(
                tenor,
                pillar_date.isoformat(),
                days,
                zero * 100.0,
                pillars.par_rates[tenor] * 100.0,
            )
        )
    lines.append("par check: max |rebuilt - quoted| = {:.2e}".format(pillars.max_par_residual()))
    report = "\n".join(lines)
    print(report)

    if args.out:
        target = Path(args.out)
        pillars.to_json(str(target))
        print("written  : {}".format(target))
        return 0
    try:
        target = record_curve_run(
            IR_CURVE,
            pillars,
            output_root=args.output_root,
            details=curve_details(pillars),
        )
    except OSError as error:
        print("ERROR: cannot write the curve run: {}".format(error))
        return 2
    print("written  : {}".format(target))
    print("latest   : {}".format(curve_root(IR_CURVE, args.output_root) / LATEST_NAME))
    return 0


def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="surface_pricer build-ir-curve")
    parser.add_argument(
        "--rate-file",
        default=str(SAMPLE_CSV),
        help="desk interest-rate export (default: surface_pricer/data/interest_rate.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "write exactly this file, no run bookkeeping (default: a new stamped "
            "run under <output-root>/ir_curve/, with latest.json pointing at it)"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="output root (default: surface_pricer/output; runs live in its ir_curve/)",
    )
    parser.add_argument(
        "--keep-constant-columns",
        action="store_true",
        help="do not drop columns whose whole history is one constant value",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["main"]
