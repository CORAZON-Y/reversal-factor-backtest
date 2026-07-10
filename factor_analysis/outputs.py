"""Persist pipeline outputs."""

from __future__ import annotations

import pandas as pd

from factor_analysis.config import PipelineConfig


def save_processed_data(processed: pd.DataFrame, config: PipelineConfig) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = processed.set_index(["date", "code"]).sort_index()
    processed.to_parquet(output_dir / "processed_data.parquet")


def save_outputs(
    processed: pd.DataFrame,
    ic_df: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_returns: pd.DataFrame,
    cumulative: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_processed_data(processed, config)
    ic_df.to_csv(output_dir / "ic_series.csv", index=False)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)
    group_returns.to_csv(output_dir / "group_returns.csv", index=False)
    cumulative.to_csv(output_dir / "group_cumulative_returns.csv", index=False)
    backtest_summary.to_csv(output_dir / "backtest_summary.csv", index=False)

    if config.make_plots:
        from factor_analysis.plotting import plot_cumulative_ic, plot_group_backtests

        plot_cumulative_ic(ic_df, output_dir)
        plot_group_backtests(cumulative, output_dir)
