"""Build the CNY FR007 interest-rate curve from the desk rate export.

``python -m surface_pricer build-ir-curve`` reads ``data/interest_rate.csv``
(FR007 fixing + ``FR007S<tenor>.IR`` IRS par quotes), bootstraps the curve with
QuantLib exactly like edslib's ``CNY-FR007`` curve (see
:mod:`surface_pricer.core.ir_curve`) and writes the pillars as JSON.  The
pricing layer then consumes that JSON through
:class:`surface_pricer.core.curves.PiecewiseRateCurve`, so QuantLib is only
needed for this build step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from ..core.ir_curve import build_fr007_curve
from ..marketdata.rate_inputs import SAMPLE_CSV, parse_interest_rate_csv
from ._common import reporter_for

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "ir_curve.json"


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

    target = Path(args.out) if args.out else DEFAULT_OUTPUT
    pillars.to_json(str(target))
    print("written  : {}".format(target))
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
        help="output JSON path (default: surface_pricer/output/ir_curve.json)",
    )
    parser.add_argument(
        "--keep-constant-columns",
        action="store_true",
        help="do not drop columns whose whole history is one constant value",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["main"]
