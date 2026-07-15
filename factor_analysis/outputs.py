"""Persist internal cache data and final PNG figures."""

from __future__ import annotations

import pandas as pd

from factor_analysis.config import PipelineConfig


SCOPE_FOLDERS = {
    "all": "直接分层回测",
    "industry_neutral": "行业中性分层回测",
}


def save_processed_data(processed: pd.DataFrame, config: PipelineConfig) -> None:
    """Save the large intermediate panel outside the user-facing output tree."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cached = processed.set_index(["date", "code"]).sort_index()
    cached.to_parquet(config.cache_dir / "processed_data.parquet")


def save_outputs(
    processed: pd.DataFrame,
    ic_df: pd.DataFrame,
    cumulative: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Save only classified PNG figures under ``output/单因子回测``."""
    save_processed_data(processed, config)
    if not config.make_plots:
        return

    from factor_analysis.plotting import plot_cumulative_ic, plot_group_backtests

    ic_dir = config.output_dir / "IC检验"
    ic_dir.mkdir(parents=True, exist_ok=True)
    plot_cumulative_ic(ic_df, ic_dir)

    for scope, folder_name in SCOPE_FOLDERS.items():
        scope_data = cumulative.loc[cumulative["scope"].eq(scope)].copy()
        if scope_data.empty:
            continue
        scope_dir = config.output_dir / folder_name
        scope_dir.mkdir(parents=True, exist_ok=True)
        plot_group_backtests(scope_data, scope_dir)
