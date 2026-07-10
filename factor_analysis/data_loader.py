"""Load and normalize raw data tables."""

from __future__ import annotations

import pandas as pd

from factor_analysis.config import PipelineConfig
from factor_analysis.constants import (
    DAILY_COLUMNS,
    DAILY_DATA_FILE,
    INDUSTRY_FILE,
    RAW_DAILY_RENAME,
    RAW_INDUSTRY_RENAME,
    ST_FILE,
    SUSPENSION_FILE,
)
from factor_analysis.utils import normalize_code


def load_daily_data(config: PipelineConfig) -> pd.DataFrame:
    daily = pd.read_parquet(config.data_dir / DAILY_DATA_FILE, columns=DAILY_COLUMNS)
    daily = daily.rename(columns=RAW_DAILY_RENAME)
    daily = daily.reset_index().rename(columns={"DATE": "date", "CODE": "code"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["code"] = normalize_code(daily["code"])
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    return daily


def load_industry_data(config: PipelineConfig) -> pd.DataFrame:
    industry = pd.read_parquet(config.data_dir / INDUSTRY_FILE)
    industry = industry.reset_index().rename(columns={"DATE": "date", "CODE": "code"})
    industry["date"] = pd.to_datetime(industry["date"].astype(str), format="%Y%m%d")
    industry["code"] = normalize_code(industry["code"])
    industry = industry.rename(columns=RAW_INDUSTRY_RENAME)
    keep_cols = [
        "date",
        "code",
        "industry_id",
        "industry_level1",
        "industry_level2",
        "industry_level3",
    ]
    industry = industry[keep_cols].drop_duplicates(["date", "code"])
    return industry


def load_st_flags(config: PipelineConfig) -> pd.DataFrame:
    st = pd.read_parquet(config.data_dir / ST_FILE)
    st.index = pd.to_datetime(st.index.astype(str), format="%Y%m%d")
    st.columns = pd.Index(st.columns.astype(str)).str.zfill(6)

    stacked = st.stack()
    stacked = stacked[stacked.notna()]
    flags = stacked.rename("st_value").reset_index()
    flags.columns = ["date", "code", "st_value"]
    flags["code"] = normalize_code(flags["code"])
    flags = flags[["date", "code"]].drop_duplicates()
    flags["is_st"] = True
    return flags


def load_suspension_flags(config: PipelineConfig) -> pd.DataFrame:
    suspension = pd.read_parquet(config.data_dir / SUSPENSION_FILE)
    suspension = suspension.rename(
        columns={"股票代码": "code", "日期": "date", "是否停牌": "is_suspended"}
    )
    suspension["date"] = pd.to_datetime(suspension["date"])
    suspension["code"] = normalize_code(suspension["code"])
    suspension["is_suspended"] = suspension["is_suspended"].fillna(0).astype(int).eq(1)
    suspension = suspension.loc[suspension["is_suspended"], ["date", "code", "is_suspended"]]
    suspension = suspension.drop_duplicates(["date", "code"])
    return suspension
