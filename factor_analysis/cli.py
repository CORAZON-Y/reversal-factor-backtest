"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from factor_analysis.config import PipelineConfig
from factor_analysis.constants import OUTPUT_DIR
from factor_analysis.pipeline import run_factor_processing, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-share factor analysis pipeline.")
    parser.add_argument("--groups", type=int, default=5, help="Number of quantile groups.")
    parser.add_argument(
        "--weight",
        choices=["equal", "float_mv", "both"],
        default="both",
        help="Backtest weighting method.",
    )
    parser.add_argument(
        "--industry-layer",
        action="store_true",
        help="Also sort stocks into groups within each industry before aggregating.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument(
        "--factors-only",
        action="store_true",
        help="Only build and save output/processed_data.parquet.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated results.")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    weights = ("equal", "float_mv") if args.weight == "both" else (args.weight,)
    config = PipelineConfig(
        output_dir=args.output_dir,
        groups=args.groups,
        weights=weights,
        include_industry_layer=args.industry_layer,
        make_plots=not args.skip_plots,
    )
    if args.factors_only:
        return run_factor_processing(config)
    return run_pipeline(config)


if __name__ == "__main__":
    main()
