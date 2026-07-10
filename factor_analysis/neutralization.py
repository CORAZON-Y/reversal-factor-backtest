"""
对行业、市值变量进行回归，得到对市值行业中性化后的因子
因子值 = 常数项 + log(流通市值) + 行业虚拟变量 + 残差
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from factor_analysis.config import PipelineConfig
from factor_analysis.utils import finite_mask


def neutralize_one_date(
    group: pd.DataFrame,
    factor_col: str,
    industry_col: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=group.index, dtype=float)
    y = group[factor_col].astype(float)
    size = group["weight_float_mv"].astype(float)
    log_size = np.log(size.where(size > 0))
    valid = finite_mask(y, log_size)

    if valid.sum() < 20 or y.loc[valid].nunique() <= 1:
        return result

    industries = group.loc[valid, industry_col].astype("string").fillna("UNKNOWN")
    industry_dummies = pd.get_dummies(industries, drop_first=True, dtype=float)
    x = pd.DataFrame({"const": 1.0, "log_float_mv": log_size.loc[valid]}, index=industries.index)
    x = pd.concat([x, industry_dummies], axis=1)

    if len(x) <= x.shape[1] + 5:
        x = x[["const", "log_float_mv"]]
    if len(x) <= x.shape[1] + 2:
        residual = y.loc[valid] - y.loc[valid].mean()
    else:
        beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y.loc[valid].to_numpy(dtype=float), rcond=None)
        residual = y.loc[valid] - x.to_numpy(dtype=float) @ beta

    std = residual.std(ddof=0)
    if np.isfinite(std) and std > 0:
        residual = (residual - residual.mean()) / std
    result.loc[valid] = residual
    return result


def neutralize_factors(
    df: pd.DataFrame,
    factor_cols: Iterable[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    out = df.copy()
    date_groups = list(out.groupby("date", sort=False).indices.items())

    for factor_col in factor_cols:
        neutral_col = f"{factor_col}_neutral"
        neutralized = pd.Series(np.nan, index=out.index, dtype=float)
        for _, positions in date_groups:
            group = out.iloc[positions]
            neutralized.iloc[positions] = neutralize_one_date(group, factor_col, config.industry_col)
        out[neutral_col] = neutralized
    return out
