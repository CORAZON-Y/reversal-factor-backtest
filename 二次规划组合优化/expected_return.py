from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_ROOT_DIR = ROOT_DIR / ".cache"
FACTOR_CACHE_DIR = CACHE_ROOT_DIR / "单因子回测"
OPTIMIZATION_CACHE_DIR = CACHE_ROOT_DIR / "二次规划组合优化"
DEFAULT_INPUT_FILE = FACTOR_CACHE_DIR / "processed_data.parquet"
DEFAULT_OUTPUT_FILE = OPTIMIZATION_CACHE_DIR / "expected_returns.parquet"


@dataclass(frozen=True)
class ExpectedReturnConfig:
    input_file: Path = DEFAULT_INPUT_FILE
    output_file: Path = DEFAULT_OUTPUT_FILE
    factor_col: str = "factor_rank_zscore"
    return_col: str = "return"
    window: int = 60
    min_periods: int = 20
    return_availability_lag: int = 2


def validate_config(config: ExpectedReturnConfig) -> None:
    if config.window <= 0:
        raise ValueError("window must be positive")
    if config.min_periods <= 0 or config.min_periods > config.window:
        raise ValueError("min_periods must be between 1 and window")
    if config.return_availability_lag < 2:
        raise ValueError(
            "return_availability_lag must be at least 2 for T+1-open to T+2-open returns"
        )


def load_factor_panel(config: ExpectedReturnConfig) -> pd.DataFrame:
    if not config.input_file.exists():
        raise FileNotFoundError(
            f"{config.input_file} does not exist. Run `.venv/bin/python -m factor_analysis` first."
        )

    panel = pd.read_parquet(
        config.input_file,
        columns=[config.factor_col, config.return_col],
    ).reset_index()
    panel = panel.rename(columns={config.return_col: "realized_return"})
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype("string")
    return panel.dropna(subset=[config.factor_col, "realized_return"])


def estimate_daily_factor_returns(
    panel: pd.DataFrame,
    config: ExpectedReturnConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in panel.groupby("date", sort=True):
        x = group[config.factor_col].to_numpy(dtype=float)
        y = group["realized_return"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        denominator = float(np.dot(x, x))
        beta = np.nan if denominator <= 0 else float(np.dot(x, y) / denominator)
        rows.append({"date": date, "realized_beta": beta, "n": int(valid.sum())})

    factor_returns = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    factor_returns["beta_hat"] = (
        factor_returns["realized_beta"]
        # A factor dated T earns the T+1-open to T+2-open return.  When the
        # T signal is formed, the latest fully observed label is therefore T-2.
        .shift(config.return_availability_lag)
        .rolling(config.window, min_periods=config.min_periods)
        .mean()
    )
    return factor_returns


def predict_expected_returns(
    panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    config: ExpectedReturnConfig,
) -> pd.DataFrame:
    beta_by_date = factor_returns.set_index("date")["beta_hat"]
    result = panel[["date", "code", config.factor_col, "realized_return"]].copy()
    result["beta_hat"] = result["date"].map(beta_by_date)
    result["expected_return"] = result["beta_hat"] * result[config.factor_col]
    result = result[
        [
            "date",
            "code",
            config.factor_col,
            "beta_hat",
            "expected_return",
            "realized_return",
        ]
    ]
    return result


def save_expected_return_outputs(
    expected_returns: pd.DataFrame,
    config: ExpectedReturnConfig,
) -> None:
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    expected_returns.to_parquet(config.output_file, index=False)


def run_expected_return_model(config: ExpectedReturnConfig) -> dict[str, pd.DataFrame]:
    validate_config(config)
    panel = load_factor_panel(config)
    factor_returns = estimate_daily_factor_returns(panel, config)
    expected_returns = predict_expected_returns(panel, factor_returns, config)
    save_expected_return_outputs(expected_returns, config)
    return {
        "expected_returns": expected_returns,
        "factor_returns": factor_returns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate expected stock returns from factor_rank_zscore.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--factor-col", default="factor_rank_zscore")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument(
        "--return-availability-lag",
        type=int,
        default=2,
        help="Number of factor-date rows before a realized return is observable.",
    )
    return parser.parse_args()


def main() -> dict[str, pd.DataFrame]:
    args = parse_args()
    config = ExpectedReturnConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        factor_col=args.factor_col,
        window=args.window,
        min_periods=args.min_periods,
        return_availability_lag=args.return_availability_lag,
    )
    return run_expected_return_model(config)


if __name__ == "__main__":
    main()
