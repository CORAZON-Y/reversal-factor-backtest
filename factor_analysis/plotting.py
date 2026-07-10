"""Plot generated analysis results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_analysis.utils import setup_plot_cache

setup_plot_cache()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_cumulative_ic(ic_df: pd.DataFrame, output_dir: Path) -> None:
    if ic_df.empty:
        return
    for metric in ["ic", "rank_ic"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for factor, group in ic_df.groupby("factor", sort=False):
            series = group.sort_values("date").set_index("date")[metric].cumsum()
            ax.plot(series.index, series.values, label=factor, linewidth=1.5)
        ax.set_title(f"Cumulative {metric.upper()}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Cumulative {metric}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"cumulative_{metric}.png", dpi=150)
        plt.close(fig)


def plot_group_backtests(cumulative: pd.DataFrame, output_dir: Path) -> None:
    if cumulative.empty:
        return
    keys = ["factor", "scope", "weight"]
    for (factor, scope, weight), group in cumulative.groupby(keys, sort=False):
        pivot = group.pivot(index="date", columns="portfolio", values="cumulative_return").sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        for column in pivot.columns:
            linewidth = 2.4 if column == "long_short" else 1.3
            linestyle = "--" if column == "long_short" else "-"
            ax.plot(pivot.index, pivot[column], label=column, linewidth=linewidth, linestyle=linestyle)
        ax.set_title(f"{factor} | {scope} | {weight}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative additive return")
        ax.legend(ncol=3)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        file_name = f"group_backtest_{factor}_{scope}_{weight}.png"
        fig.savefig(output_dir / file_name, dpi=150)
        plt.close(fig)
