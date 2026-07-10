"""
处理因子、收益率、行业、市值、ST、停牌、涨跌停、上市天数等信息，
对前面清洗过的股票日线数据进行预处理，对股票能否交易等数据进行处理，为回测构建数据集
1. 计算因子值、收益率、T+1、T+2的开盘价等
2.判断是否被限制、停牌、ST、上市天数
3. 构建市值权重列
"""

from __future__ import annotations

import pandas as pd

from factor_analysis.config import PipelineConfig
from factor_analysis.constants import LIMIT_STATUS, SUSPENDED_STATUS
from factor_analysis.utils import finite_mask


def build_factor_dataset(
    daily: pd.DataFrame,
    industry: pd.DataFrame,
    st_flags: pd.DataFrame,
    suspension_flags: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    df = daily.copy()
    grouped = df.groupby("code", sort=False)

    raw_factor = df["close"] / grouped["close"].shift(config.factor_lag) - 1.0
    df["factor"] = config.factor_sign * raw_factor
    df["t1_date"] = grouped["date"].shift(-1)
    df["t2_date"] = grouped["date"].shift(-2)
    df["open_t1"] = grouped["open"].shift(-1)
    df["open_t2"] = grouped["open"].shift(-2)
    df["return"] = df["open_t2"] / df["open_t1"] - 1.0
    df["chg_status_t1"] = grouped["chg_status"].shift(-1)

    first_seen = grouped["date"].transform("min")
    global_start = df["date"].min()
    listed_days = (df["t1_date"] - first_seen).dt.days
    listed_days = listed_days.mask(first_seen.eq(global_start), config.min_listed_days)
    df["listed_days_t1"] = listed_days

    st_t1 = st_flags.rename(columns={"date": "t1_date", "is_st": "is_st_t1"})
    df = df.merge(st_t1, on=["t1_date", "code"], how="left")
    df["is_st_t1"] = df["is_st_t1"].fillna(False).astype(bool)

    suspension_t1 = suspension_flags.rename(
        columns={"date": "t1_date", "is_suspended": "is_suspended_t1"}
    )
    df = df.merge(suspension_t1, on=["t1_date", "code"], how="left")
    df["is_suspended_t1"] = df["is_suspended_t1"].fillna(False).astype(bool)

    df["is_limit_t1"] = df["chg_status_t1"].isin(LIMIT_STATUS)
    df["is_suspended_t1"] |= df["chg_status_t1"].eq(SUSPENDED_STATUS)

    df = df.merge(industry, on=["date", "code"], how="left")

    base_valid = (
        finite_mask(df["factor"], df["return"], df["open_t1"], df["open_t2"], df["close"])
        & df["open_t1"].gt(0)
        & df["open_t2"].gt(0)
        & df["close"].gt(0)
    )
    tradable_valid = (
        ~df["is_limit_t1"]
        & ~df["is_suspended_t1"]
        & ~df["is_st_t1"]
        & df["listed_days_t1"].ge(config.min_listed_days)
    )

    df = df.loc[base_valid & tradable_valid].copy()
    df["weight_float_mv"] = df["float_market_value"].where(df["float_market_value"].gt(0))
    df["weight_float_mv"] = df["weight_float_mv"].fillna(
        df["market_value"].where(df["market_value"].gt(0))
    )
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df
