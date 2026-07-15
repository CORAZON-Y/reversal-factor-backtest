"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from factor_analysis.config import PipelineConfig
from factor_analysis.constants import DATA_DIR, FACTOR_CACHE_DIR, OUTPUT_DIR
from factor_analysis.pipeline import run_factor_processing, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-share factor analysis pipeline.")
    parser.add_argument("--groups", type=int, default=5, help="Number of quantile groups.")
    parser.add_argument(
        "--backtest-start-day",
        type=int,
        default=61,
        help="1-based trading-day offset for group backtests; 61 starts after skipping the first 60 dates.",
    )
    parser.add_argument(
        "--winsor-mad-n",
        type=float,
        default=3.0,
        help="MAD winsorization multiplier used before z-score standardization.",
    )
    parser.add_argument(
        "--weight",
        choices=["equal", "float_mv", "both"],
        default="both",
        help="Backtest weighting method.",
    )
    parser.add_argument(
        "--industry-layer",
        "--industry-neutral",
        dest="industry_layer",
        action="store_true",
        help="Also build industry-neutral groups weighted by benchmark industry weights.",
    )
    parser.add_argument(
        "--benchmark-weight-col",
        default=None,
        help=(
            "Column containing benchmark constituent weights. If omitted, "
            "industry-neutral groups use universe float market value as the industry-weight proxy."
        ),
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument(
        "--factors-only",
        action="store_true",
        help="Only build and save the internal .cache/单因子回测/processed_data.parquet.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing daily_data.parquet, industry.parquet, st.parquet, and 停牌.parquet.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated results.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=FACTOR_CACHE_DIR,
        help="Directory for internal intermediate parquet data.",
    )
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    weights = ("equal", "float_mv") if args.weight == "both" else (args.weight,)
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        groups=args.groups,
        backtest_start_day=args.backtest_start_day,
        winsor_mad_n=args.winsor_mad_n,
        weights=weights,
        include_industry_layer=args.industry_layer,
        benchmark_weight_col=args.benchmark_weight_col,
        make_plots=not args.skip_plots,
    )
    if args.factors_only:
        return run_factor_processing(config)
    return run_pipeline(config)


if __name__ == "__main__":
    main()
