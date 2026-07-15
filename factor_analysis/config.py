"""Runtime configuration for the factor analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from factor_analysis.constants import DATA_DIR, FACTOR_CACHE_DIR, OUTPUT_DIR


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    cache_dir: Path = FACTOR_CACHE_DIR
    factor_lag: int = 5
    factor_sign: float = -1.0
    winsor_mad_n: float = 3.0
    min_listed_days: int = 60
    backtest_start_day: int = 61
    groups: int = 5
    trading_days_per_year: int = 252
    industry_col: str = "industry_level1"
    weights: tuple[str, ...] = ("equal", "float_mv")
    include_industry_layer: bool = False
    benchmark_weight_col: str | None = None
    make_plots: bool = True
