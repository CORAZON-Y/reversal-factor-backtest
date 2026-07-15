"""
生成标准化因子
一.处理极端值
二。标准化因子（两种方案）
1.zscore
2.rank_zscore
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
    else:
        width = n_mad * mad
        lower = median - width
        upper = median + width
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    cleaned = winsorize_mad(series, n_mad=n_mad)
    std = cleaned.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=series.index)
    return (cleaned - cleaned.mean()) / std


def rank_zscore(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="average", pct=True)
    centered = ranks - ranks.mean()
    std = centered.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=series.index)
    return centered / std


def standardize_factors(df: pd.DataFrame, n_mad: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    by_date = out.groupby("date", sort=False)["factor"]
    out["factor_zscore"] = by_date.transform(zscore, n_mad=n_mad)
    out["factor_rank_zscore"] = by_date.transform(rank_zscore)
    return out


def factor_columns() -> list[str]:
    return [
        "factor_zscore",
        "factor_zscore_neutral",
        "factor_rank_zscore",
        "factor_rank_zscore_neutral",
    ]
