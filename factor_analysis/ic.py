"""
对因子IC值做t检验
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def calculate_ic(df: pd.DataFrame, factors: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_col in factors:
        needed = df[["date", factor_col, "return"]].dropna()
        for date, group in needed.groupby("date", sort=True):
            if len(group) < 20 or group[factor_col].nunique() <= 1 or group["return"].nunique() <= 1:
                continue
            rows.append(
                {
                    "date": date,
                    "factor": factor_col,
                    "ic": group[factor_col].corr(group["return"], method="pearson"),
                    "rank_ic": group[factor_col].corr(group["return"], method="spearman"),
                    "n": len(group),
                }
            )
    return pd.DataFrame(rows, columns=["date", "factor", "ic", "rank_ic", "n"])


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor",
        "metric",
        "count",
        "mean",
        "std",
        "t_stat",
        "p_value",
        "ic_ir_annual",
        "positive_ratio",
        "abs_gt_0.02_ratio",
    ]
    if ic_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for factor, group in ic_df.groupby("factor", sort=False):
        for metric in ["ic", "rank_ic"]:
            values = group[metric].dropna()
            if values.empty:
                continue
            std = values.std(ddof=1)
            t_stat, p_value = stats.ttest_1samp(values, popmean=0.0)
            rows.append(
                {
                    "factor": factor,
                    "metric": metric,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": std,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "ic_ir_annual": values.mean() / std * np.sqrt(252) if std > 0 else np.nan,
                    "positive_ratio": values.gt(0).mean(),
                    "abs_gt_0.02_ratio": values.abs().gt(0.02).mean(),
                }
            )
    return pd.DataFrame(rows, columns=columns)
