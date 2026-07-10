"""
回测说明：
有以下功能
- 截面排序
- 分层组合
- 等权/市值加权
- 多空组合
- 行业内分层
- 累计收益
- 基础评价指标

没有以下功能
- 手续费
- 滑点
- 调仓换手率
- 持仓明细
- 基准指数和超额收益
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from factor_analysis.config import PipelineConfig


def assign_quantile_groups(series: pd.Series, groups: int) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < groups or valid.nunique() < groups:
        return result
    ranks = valid.rank(method="first", ascending=True)
    labels = np.floor((ranks - 1) * groups / len(valid)).astype(int) + 1
    result.loc[valid.index] = labels.astype(float)
    return result


def calculate_weighted_group_returns(
    data: pd.DataFrame,
    weight_method: str,
) -> pd.DataFrame:
    if weight_method == "equal":
        daily_returns = data.groupby(["date", "group"], sort=True)["return"].mean()
    elif weight_method == "float_mv":
        weighted = data.copy()
        weighted["weight"] = weighted["weight_float_mv"].where(weighted["weight_float_mv"].gt(0))
        weighted = weighted.dropna(subset=["weight"])
        weighted["weighted_return"] = weighted["return"] * weighted["weight"]
        grouped = weighted.groupby(["date", "group"], sort=True)
        daily_returns = grouped["weighted_return"].sum() / grouped["weight"].sum()
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    wide = daily_returns.unstack("group").sort_index()
    wide.columns = [f"group_{int(col)}" for col in wide.columns]
    high_col = f"group_{int(data['group'].max())}"
    if "group_1" in wide.columns and high_col in wide.columns:
        wide["long_short"] = wide[high_col] - wide["group_1"]
    return wide


def summarize_backtest(returns: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in returns.columns:
        series = returns[column].dropna()
        if series.empty:
            continue
        curve = series.cumsum()
        drawdown = curve - curve.cummax()
        std = series.std(ddof=1)
        rows.append(
            {
                "portfolio": column,
                "periods": len(series),
                "total_additive_return": series.sum(),
                "annual_return": series.mean() * config.trading_days_per_year,
                "annual_vol": std * np.sqrt(config.trading_days_per_year),
                "sharpe": series.mean() / std * np.sqrt(config.trading_days_per_year)
                if std > 0
                else np.nan,
                "max_drawdown_additive": drawdown.min(),
                "win_rate": series.gt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def run_group_backtest(
    df: pd.DataFrame,
    factors: Iterable[str],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return_frames: list[pd.DataFrame] = []
    cumulative_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    scopes = ["all"]
    if config.include_industry_layer:
        scopes.append("industry_layer")

    for factor_col in factors:
        for scope in scopes:
            use_cols = ["date", "code", factor_col, "return", "weight_float_mv", config.industry_col]
            data = df[use_cols].dropna(subset=[factor_col, "return"]).copy()
            if scope == "industry_layer":
                data["group"] = data.groupby(["date", config.industry_col], sort=False)[factor_col].transform(
                    assign_quantile_groups,
                    groups=config.groups,
                )
            else:
                data["group"] = data.groupby("date", sort=False)[factor_col].transform(
                    assign_quantile_groups,
                    groups=config.groups,
                )
            data = data.dropna(subset=["group"])
            data["group"] = data["group"].astype(int)

            for weight_method in config.weights:
                daily_returns = calculate_weighted_group_returns(data, weight_method)
                cumulative = daily_returns.cumsum()
                summary = summarize_backtest(daily_returns, config)

                daily_long = daily_returns.reset_index().melt(
                    id_vars="date",
                    var_name="portfolio",
                    value_name="return",
                )
                daily_long["factor"] = factor_col
                daily_long["scope"] = scope
                daily_long["weight"] = weight_method

                cumulative_long = cumulative.reset_index().melt(
                    id_vars="date",
                    var_name="portfolio",
                    value_name="cumulative_return",
                )
                cumulative_long["factor"] = factor_col
                cumulative_long["scope"] = scope
                cumulative_long["weight"] = weight_method

                summary["factor"] = factor_col
                summary["scope"] = scope
                summary["weight"] = weight_method

                return_frames.append(daily_long)
                cumulative_frames.append(cumulative_long)
                summary_frames.append(summary)

    returns = pd.concat(return_frames, ignore_index=True)
    cumulative = pd.concat(cumulative_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    return returns, cumulative, summary
