"""
回测说明：
有以下功能
- 截面排序
- 分层组合
- 等权/市值加权
- 行业中性分层
- 累计净值
- 基础评价指标

没有以下功能
- 手续费
- 滑点
- 调仓换手率
- 持仓明细
- 基准指数收益和超额收益
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from factor_analysis.config import PipelineConfig


def assign_quantile_groups(
    series: pd.Series,
    groups: int,
    ascending: bool = False,
) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < groups or valid.nunique() < groups:
        return result
    ranks = valid.rank(method="first", ascending=ascending)
    labels = np.floor((ranks - 1) * groups / len(valid)).astype(int) + 1
    result.loc[valid.index] = labels.astype(float)
    return result


def calculate_net_value(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return (1.0 + returns).cumprod()


def filter_backtest_start(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.Timestamp | pd.NaT]:
    start_day = max(int(config.backtest_start_day), 1)
    unique_dates = pd.Series(df["date"].dropna().unique()).sort_values(ignore_index=True)
    if unique_dates.empty:
        return df.iloc[0:0].copy(), pd.NaT
    if start_day > len(unique_dates):
        return df.iloc[0:0].copy(), pd.NaT
    start_date = unique_dates.iloc[start_day - 1]
    return df.loc[df["date"].ge(start_date)].copy(), start_date


def calculate_weighted_group_returns(
    data: pd.DataFrame,
    weight_method: str,
) -> pd.DataFrame:
    if weight_method == "equal":
        daily_returns = data.groupby(["date", "group"], sort=True)["return"].mean()
    elif weight_method == "float_mv":
        weighted = data.copy()
        weighted["weight"] = weighted["weight_float_mv"].where(
            weighted["weight_float_mv"].gt(0)
        )
        weighted = weighted.dropna(subset=["weight"])
        weighted["weighted_return"] = weighted["return"] * weighted["weight"]
        grouped = weighted.groupby(["date", "group"], sort=True)
        daily_returns = grouped["weighted_return"].sum() / grouped["weight"].sum()
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    if daily_returns.empty:
        return pd.DataFrame()
    wide = daily_returns.unstack("group").sort_index()
    wide.columns = [f"group_{int(col)}" for col in wide.columns]
    return wide


def calculate_industry_weights(
    weight_data: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    if config.benchmark_weight_col is not None:
        if config.benchmark_weight_col not in weight_data.columns:
            raise ValueError(f"Missing benchmark weight column: {config.benchmark_weight_col}")
        weight_col = config.benchmark_weight_col
    else:
        weight_col = "weight_float_mv"

    weights = weight_data[["date", config.industry_col, weight_col]].copy()
    weights["industry_weight_source"] = weights[weight_col].where(
        weights[weight_col].gt(0)
    )
    weights = weights.dropna(subset=[config.industry_col, "industry_weight_source"])
    industry_weight = weights.groupby(["date", config.industry_col], sort=True)[
        "industry_weight_source"
    ].sum()
    date_total = industry_weight.groupby(level="date").transform("sum")
    industry_weight = industry_weight / date_total.replace(0, np.nan)
    industry_weight = industry_weight.dropna()
    return industry_weight.rename("industry_weight").reset_index()


def calculate_industry_neutral_group_returns(
    data: pd.DataFrame,
    industry_weight_data: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    weighted = data.copy()
    weighted["stock_weight"] = weighted["weight_float_mv"].where(
        weighted["weight_float_mv"].gt(0)
    )
    weighted = weighted.dropna(subset=[config.industry_col, "stock_weight"])
    if weighted.empty:
        return pd.DataFrame()
    weighted["weighted_return"] = weighted["return"] * weighted["stock_weight"]

    keys = ["date", config.industry_col, "group"]
    grouped = weighted.groupby(keys, sort=True)
    industry_group_returns = grouped["weighted_return"].sum() / grouped[
        "stock_weight"
    ].sum()
    industry_group_returns = industry_group_returns.rename(
        "industry_group_return"
    ).reset_index()

    industry_weights = calculate_industry_weights(industry_weight_data, config)
    if industry_group_returns.empty or industry_weights.empty:
        return pd.DataFrame()
    combined = industry_group_returns.merge(
        industry_weights,
        on=["date", config.industry_col],
        how="inner",
    )
    if combined.empty:
        return pd.DataFrame()
    group_weight_total = combined.groupby(["date", "group"], sort=False)[
        "industry_weight"
    ].transform("sum")
    valid_weight = group_weight_total.gt(0)
    combined = combined.loc[valid_weight].copy()
    combined["industry_weight"] = (
        combined["industry_weight"] / group_weight_total.loc[valid_weight]
    )
    combined["weighted_group_return"] = (
        combined["industry_group_return"] * combined["industry_weight"]
    )
    daily_returns = combined.groupby(["date", "group"], sort=True)[
        "weighted_group_return"
    ].sum()
    if daily_returns.empty:
        return pd.DataFrame()
    wide = daily_returns.unstack("group").sort_index()
    wide.columns = [f"group_{int(col)}" for col in wide.columns]
    return wide


def summarize_backtest(returns: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in returns.columns:
        series = returns[column].dropna()
        if series.empty:
            continue
        curve = calculate_net_value(series)
        drawdown = curve / curve.cummax() - 1.0
        std = series.std(ddof=1)
        total_return = curve.iloc[-1] - 1.0
        annual_return = (
            curve.iloc[-1] ** (config.trading_days_per_year / len(series)) - 1.0
            if curve.iloc[-1] > 0
            else np.nan
        )
        rows.append(
            {
                "portfolio": column,
                "periods": len(series),
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_vol": std * np.sqrt(config.trading_days_per_year),
                "sharpe": series.mean() / std * np.sqrt(config.trading_days_per_year)
                if std > 0
                else np.nan,
                "max_drawdown": drawdown.min(),
                "win_rate": series.gt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def run_group_backtest(
    df: pd.DataFrame,
    factors: Iterable[str],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, backtest_start_date = filter_backtest_start(df, config)
    return_frames: list[pd.DataFrame] = []
    cumulative_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    scopes = ["all"]
    if config.include_industry_layer:
        scopes.append("industry_neutral")

    for factor_col in factors:
        for scope in scopes:
            use_cols = [
                "date",
                "code",
                factor_col,
                "return",
                "weight_float_mv",
                config.industry_col,
            ]
            industry_weight_cols = ["date", config.industry_col]
            if config.benchmark_weight_col is not None:
                use_cols.append(config.benchmark_weight_col)
                industry_weight_cols.append(config.benchmark_weight_col)
            else:
                industry_weight_cols.append("weight_float_mv")
            missing_cols = sorted(
                set(use_cols + industry_weight_cols) - set(df.columns)
            )
            if missing_cols:
                raise ValueError(
                    f"Missing required columns for group backtest: {missing_cols}"
                )
            data = df[use_cols].dropna(subset=[factor_col, "return"]).copy()
            # ``factor_sign`` is applied once when the raw factor is constructed.
            # Every downstream factor column therefore has the same "higher is better" direction.
            data["_backtest_factor"] = data[factor_col]
            industry_weight_data = df[industry_weight_cols].copy()
            if scope == "industry_neutral":
                data["group"] = data.groupby(
                    ["date", config.industry_col], sort=False
                )["_backtest_factor"].transform(
                    assign_quantile_groups, groups=config.groups, ascending=False
                )
            else:
                data["group"] = data.groupby("date", sort=False)[
                    "_backtest_factor"
                ].transform(assign_quantile_groups, groups=config.groups, ascending=False)
            data = data.dropna(subset=["group"])
            data["group"] = data["group"].astype(int)

            if scope == "industry_neutral":
                weight_methods = ("benchmark_industry_float_mv",)
            else:
                weight_methods = config.weights

            for weight_method in weight_methods:
                if scope == "industry_neutral":
                    daily_returns = calculate_industry_neutral_group_returns(
                        data,
                        industry_weight_data,
                        config,
                    )
                else:
                    daily_returns = calculate_weighted_group_returns(
                        data, weight_method
                    )
                if daily_returns.empty:
                    continue
                cumulative = calculate_net_value(daily_returns)
                summary = summarize_backtest(daily_returns, config)

                daily_long = daily_returns.reset_index().melt(
                    id_vars="date",
                    var_name="portfolio",
                    value_name="return",
                )
                daily_long["factor"] = factor_col
                daily_long["scope"] = scope
                daily_long["weight"] = weight_method
                daily_long["backtest_factor_sign"] = config.factor_sign
                daily_long["winsor_mad_n"] = config.winsor_mad_n
                daily_long["backtest_start_day"] = config.backtest_start_day
                daily_long["backtest_start_date"] = backtest_start_date

                cumulative_long = cumulative.reset_index().melt(
                    id_vars="date",
                    var_name="portfolio",
                    value_name="cumulative_return",
                )
                cumulative_long["factor"] = factor_col
                cumulative_long["scope"] = scope
                cumulative_long["weight"] = weight_method
                cumulative_long["backtest_factor_sign"] = config.factor_sign
                cumulative_long["winsor_mad_n"] = config.winsor_mad_n
                cumulative_long["backtest_start_day"] = config.backtest_start_day
                cumulative_long["backtest_start_date"] = backtest_start_date

                summary["factor"] = factor_col
                summary["scope"] = scope
                summary["weight"] = weight_method
                summary["backtest_factor_sign"] = config.factor_sign
                summary["winsor_mad_n"] = config.winsor_mad_n
                summary["backtest_start_day"] = config.backtest_start_day
                summary["backtest_start_date"] = backtest_start_date

                return_frames.append(daily_long)
                cumulative_frames.append(cumulative_long)
                summary_frames.append(summary)

    if not return_frames:
        return (
            pd.DataFrame(
                columns=[
                    "date",
                    "portfolio",
                    "return",
                    "factor",
                    "scope",
                    "weight",
                    "backtest_factor_sign",
                    "winsor_mad_n",
                    "backtest_start_day",
                    "backtest_start_date",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "date",
                    "portfolio",
                    "cumulative_return",
                    "factor",
                    "scope",
                    "weight",
                    "backtest_factor_sign",
                    "winsor_mad_n",
                    "backtest_start_day",
                    "backtest_start_date",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "portfolio",
                    "periods",
                    "total_return",
                    "annual_return",
                    "annual_vol",
                    "sharpe",
                    "max_drawdown",
                    "win_rate",
                    "factor",
                    "scope",
                    "weight",
                    "backtest_factor_sign",
                    "winsor_mad_n",
                    "backtest_start_day",
                    "backtest_start_date",
                ]
            ),
        )

    returns = pd.concat(return_frames, ignore_index=True)
    cumulative = pd.concat(cumulative_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    return returns, cumulative, summary
