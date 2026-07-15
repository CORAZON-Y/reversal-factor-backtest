"""End-to-end factor analysis pipeline orchestration."""

from __future__ import annotations

import pandas as pd

from factor_analysis.backtest import run_group_backtest
from factor_analysis.config import PipelineConfig
from factor_analysis.data_loader import (
    load_daily_data,
    load_industry_data,
    load_st_flags,
    load_suspension_flags,
)
from factor_analysis.dataset import build_factor_dataset
from factor_analysis.factors import factor_columns, standardize_factors
from factor_analysis.ic import calculate_ic, summarize_ic
from factor_analysis.neutralization import neutralize_factors
from factor_analysis.outputs import save_outputs, save_processed_data


def build_processed_data(config: PipelineConfig) -> pd.DataFrame:
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    daily = load_daily_data(config)
    industry = load_industry_data(config)
    st_flags = load_st_flags(config)
    suspension_flags = load_suspension_flags(config)

    processed = build_factor_dataset(daily, industry, st_flags, suspension_flags, config)
    processed = standardize_factors(processed, n_mad=config.winsor_mad_n)
    processed = neutralize_factors(processed, ["factor_zscore", "factor_rank_zscore"], config)
    return processed


def run_factor_processing(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    processed = build_processed_data(config)
    save_processed_data(processed, config)
    return {"processed": processed}


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    processed = build_processed_data(config)

    factors = factor_columns()
    ic_df = calculate_ic(processed, factors)
    ic_summary = summarize_ic(ic_df)
    group_returns, cumulative, backtest_summary = run_group_backtest(processed, factors, config)
    save_outputs(processed, ic_df, cumulative, config)

    return {
        "processed": processed,
        "ic": ic_df,
        "ic_summary": ic_summary,
        "group_returns": group_returns,
        "group_cumulative_returns": cumulative,
        "backtest_summary": backtest_summary,
    }
