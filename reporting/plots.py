"""Plot helpers for the EDS SABR fit: one smile chart per expiry plus a term
structure summary."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..fitting.pipeline import FitResult


def plot_fit_result(
    fit_result: FitResult,
    output_dir: str,
    *,
    title_prefix: str = "",
    show: bool = False,
    dpi: int = 150,
    grid_size: int = 161,
) -> List[Path]:
    """Write one smile chart per expiry plus a term-structure summary."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for result in fit_result.slices:
        written.append(
            _plot_one_slice(result, output_path, title_prefix, show, dpi, grid_size, plt)
        )
    written.append(
        _plot_term_structure(fit_result, output_path / "term_structure.png", show, dpi, plt)
    )
    return written


def _plot_one_slice(result, output_path, title_prefix, show, dpi, grid_size, plt) -> Path:
    slice_info = result.slice_info
    forward = float(slice_info.forward)
    moneyness = slice_info.strikes / forward

    x_min = max(0.70, float(np.min(moneyness)) - 0.025)
    x_max = min(1.30, float(np.max(moneyness)) + 0.025)
    curve_x = np.linspace(x_min, x_max, max(41, int(grid_size)))
    curve_y = result.fitted.get_implied_vol(curve_x * forward)

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.fill_between(
        moneyness,
        slice_info.bid_vols,
        slice_info.ask_vols,
        color="#4C78A8",
        alpha=0.18,
        linewidth=0.0,
        label="Market bid-offer",
    )
    axis.scatter(
        moneyness,
        slice_info.bid_vols,
        color="#1F4E79",
        marker="v",
        s=26,
        label="Market bid IV",
        zorder=3,
    )
    axis.scatter(
        moneyness,
        slice_info.ask_vols,
        color="#D95F02",
        marker="^",
        s=26,
        label="Market offer IV",
        zorder=3,
    )
    axis.plot(curve_x, curve_y, color="#111111", linewidth=1.8, label="Fitted EDS SABR", zorder=4)
    axis.axvline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    axis.set_xlim(x_min, x_max)
    lower = float(np.min(np.concatenate([slice_info.bid_vols, curve_y])))
    upper = float(np.max(np.concatenate([slice_info.ask_vols, curve_y])))
    span = max(upper - lower, 0.005)
    padding = max(0.12 * span, 0.0025)
    axis.set_ylim(max(0.0, lower - padding), upper + padding)
    title = "{} | F={:.2f} | ATM={:.2%} | RMSE={:.5f}".format(
        result.expiry.date().isoformat(), forward, result.atm_vol, result.rmse
    )
    if title_prefix:
        title = "{} | {}".format(title_prefix, title)
    axis.set_title(title)
    axis.set_xlabel("Moneyness K / F")
    axis.set_ylabel("Implied volatility")
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: "{:.1f}%".format(100.0 * value)))
    axis.grid(True, alpha=0.22)
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()

    path = output_path / "smile_{}.png".format(result.expiry.date().isoformat())
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _plot_term_structure(fit_result: FitResult, path: Path, show, dpi, plt) -> Path:
    slices = fit_result.slices
    expiries = [result.expiry.date() for result in slices]
    atm = [result.atm_vol for result in slices]
    surface = fit_result.surface

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    axes[0].plot(expiries, atm, marker="o", color="#1F4E79")
    axes[0].set_title("ATM volatility")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: "{:.1f}%".format(100.0 * value)))
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(expiries, surface.skews, marker="o", label="skew")
    axes[1].plot(expiries, surface.convs, marker="s", label="conv")
    axes[1].set_title("skew / conv (scaled)")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(fontsize=8)

    axes[2].plot(expiries, surface.left_skews_1, marker="o", label="left skew1")
    axes[2].plot(expiries, surface.left_skews_2, marker="s", label="left skew2")
    axes[2].plot(expiries, surface.right_skews_1, marker="^", label="right skew1")
    axes[2].plot(expiries, surface.right_skews_2, marker="d", label="right skew2")
    axes[2].set_title("wing parameters (scaled)")
    axes[2].grid(True, alpha=0.22)
    axes[2].legend(fontsize=8)

    for axis in axes:
        for label in axis.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

    title = "{} SABR term structure".format(fit_result.underlying or "surface")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


__all__ = ["plot_fit_result"]
