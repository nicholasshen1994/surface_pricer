"""Text report for one EDS SABR calibration run."""

from __future__ import annotations

from typing import List

import numpy as np

from ..fitting.pipeline import FitResult


def format_fit_report(
    fit_result: FitResult,
    *,
    max_smile_rows_per_expiry: int = 0,
) -> str:
    """Render the fitted surface parameters and fit quality.

    Setting ``max_smile_rows_per_expiry`` above zero appends the per-strike
    market/fitted vol comparison for each expiry (first N strikes near the
    forward).
    """
    lines: List[str] = []
    header = "{} | valuation={:%Y-%m-%d %H:%M:%S} | spot={:.4f}".format(
        fit_result.underlying or "surface",
        fit_result.valuation_datetime,
        fit_result.spot,
    )
    lines.append(header)
    lines.append(
        "Settings: weight={} forward={} maxiter={} bounds=custom".format(
            fit_result.settings.weight_mode,
            fit_result.settings.forward_source,
            fit_result.settings.max_iterations,
        )
    )
    lines.append("")
    lines.append(
        "Expiry       Fwd        Tau      ATM      skew     conv     "
        "L1       L2       R1       R2       RMSE     WRMSE    N   OutBA"
    )
    lines.append("-" * 118)
    for result in fit_result.slices:
        params = result.params
        lines.append(
            (
                "{:<12} "
                "{:>9.3f} {:>8.4f} {:>8.4f} "
                "{:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} "
                "{:>8.5f} {:>8.5f} {:>3d} {:>6d}"
            ).format(
                result.expiry.date().isoformat(),
                result.forward,
                result.tau,
                result.atm_vol,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                result.rmse,
                result.weighted_rmse,
                len(result.slice_info.strikes),
                len(result.out_of_bid_ask),
            )
            + _pillar_tag(result)
        )
        if max_smile_rows_per_expiry > 0:
            lines.extend(
                _smile_rows(result, max_smile_rows_per_expiry)
            )
    lines.append("")
    lines.append(
        "Parameters are scaled surface values: raw fit params x max(0.3, sqrt(tau))."
    )
    return "\n".join(lines)


def _pillar_tag(result) -> str:
    """``[MANUAL]`` / ``[SYNTH]`` marker for hand-adjusted pillars."""
    if getattr(result, "is_override", False):
        return "  [MANUAL]"
    if getattr(result, "is_synthetic", False):
        return "  [SYNTH]"
    return ""


def _smile_rows(result, max_rows: int) -> List[str]:
    slice_info = result.slice_info
    fitted = result.fitted.get_implied_vol(slice_info.strikes)
    distance = np.abs(slice_info.strikes - slice_info.forward)
    selected = np.argsort(distance)[: min(len(slice_info.strikes), int(max_rows))]
    selected = sorted(int(index) for index in selected)
    lines = ["  strike      side   market     bid       ask       fitted    diff"]
    for index in selected:
        diff = float(fitted[index] - slice_info.vols[index])
        lines.append(
            "  {:>9.1f}  {:<5} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>8.5f}".format(
                slice_info.strikes[index],
                slice_info.option_types[index],
                slice_info.vols[index],
                slice_info.bid_vols[index],
                slice_info.ask_vols[index],
                fitted[index],
                diff,
            )
        )
    return lines


__all__ = ["format_fit_report"]
