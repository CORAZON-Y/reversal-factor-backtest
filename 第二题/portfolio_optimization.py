"""组合优化回测模块。

每个调仓日执行以下流程：

1. 按 `factor_zscore` 从高到低选出前 N 只股票。
2. 使用候选股票过去一段时间的真实收益估计协方差矩阵。
3. 在风险和权重约束下，最大化组合预期收益。
4. 用下一期真实收益计算组合回测表现。

优化模型：

    maximize    mu' w
    subject to  w' Sigma w <= max_variance
                sum(w) = 1
                0 <= w_i <= max_weight
                industry_weight_j <= max_industry_weight

其中：

- `mu` 是股票预期收益。
- `w` 是组合权重。
- `Sigma` 是候选股票协方差矩阵。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_RETURN_FILE = ROOT_DIR / "第二题" / "expected_returns.parquet"
DEFAULT_PROCESSED_DATA_FILE = ROOT_DIR / "output" / "processed_data.parquet"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "第二题" / "optimized_portfolio"


@dataclass(frozen=True)
class OptimizationConfig:
    expected_return_file: Path = DEFAULT_EXPECTED_RETURN_FILE
    processed_data_file: Path = DEFAULT_PROCESSED_DATA_FILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    factor_col: str = "factor_zscore"
    industry_col: str = "industry_level1"
    candidate_count: int = 100
    risk_window: int = 60
    min_history: int = 20
    max_variance: float = 0.0004
    max_weight: float = 0.05
    max_industry_weight: float = 0.20
    shrinkage: float = 0.20
    trading_days_per_year: int = 252
    start_date: str | None = None
    end_date: str | None = None
    max_dates: int | None = None


def load_optimization_panel(config: OptimizationConfig) -> pd.DataFrame:
    if not config.expected_return_file.exists():
        raise FileNotFoundError(
            f"{config.expected_return_file} does not exist. Run `.venv/bin/python 第二题/expected_return.py` first."
        )
    if not config.processed_data_file.exists():
        raise FileNotFoundError(
            f"{config.processed_data_file} does not exist. Run `.venv/bin/python -m factor_analysis` first."
        )

    processed = pd.read_parquet(
        config.processed_data_file,
        columns=[config.factor_col, "return", config.industry_col],
    ).reset_index()
    processed = processed.rename(columns={"return": "realized_return"})
    processed["date"] = pd.to_datetime(processed["date"])
    processed["code"] = processed["code"].astype("string")

    expected = pd.read_parquet(
        config.expected_return_file,
        columns=["date", "code", "expected_return", "beta_hat"],
    )
    expected["date"] = pd.to_datetime(expected["date"])
    expected["code"] = expected["code"].astype("string")

    panel = processed.merge(expected, on=["date", "code"], how="left")
    panel[config.industry_col] = panel[config.industry_col].astype("string").fillna("UNKNOWN")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    return panel


def make_return_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="date", columns="code", values="realized_return").sort_index()


def estimate_covariance_matrix(
    history: pd.DataFrame,
    config: OptimizationConfig,
) -> tuple[np.ndarray | None, list[str]]:
    available_counts = history.notna().sum(axis=0)
    valid_codes = available_counts[available_counts >= config.min_history].index.tolist()
    if len(valid_codes) < max(2, int(np.ceil(1.0 / config.max_weight))):
        return None, []

    hist = history[valid_codes].astype(float)
    centered = hist - hist.mean(axis=0)
    centered = centered.fillna(0.0)
    if len(centered) < 2:
        return None, []

    cov = np.cov(centered.to_numpy(dtype=float), rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    diag = np.diag(cov).copy()
    positive_diag = diag[diag > 0]
    if len(positive_diag) == 0:
        return None, []

    floor = float(np.nanmedian(positive_diag) * 1e-4)
    diag = np.maximum(diag, floor)
    shrunk = (1.0 - config.shrinkage) * cov + config.shrinkage * np.diag(diag)
    shrunk = (shrunk + shrunk.T) / 2.0
    shrunk[np.diag_indices_from(shrunk)] = np.maximum(np.diag(shrunk), floor)
    return shrunk, valid_codes


def build_industry_matrix(industries: pd.Series) -> tuple[np.ndarray, list[str]]:
    names = sorted(industries.astype("string").fillna("UNKNOWN").unique().tolist())
    matrix = np.zeros((len(names), len(industries)), dtype=float)
    industry_values = industries.astype("string").fillna("UNKNOWN").to_numpy()
    for row, name in enumerate(names):
        matrix[row, :] = industry_values == name
    return matrix, names


def find_linear_feasible_weight(
    industry_matrix: np.ndarray,
    config: OptimizationConfig,
) -> np.ndarray | None:
    n = industry_matrix.shape[1]
    result = linprog(
        c=np.zeros(n),
        A_ub=industry_matrix,
        b_ub=np.full(industry_matrix.shape[0], config.max_industry_weight),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, config.max_weight)] * n,
        method="highs",
    )
    if not result.success:
        return None
    return np.asarray(result.x, dtype=float)


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def minimize_variance(
    covariance: np.ndarray,
    industry_matrix: np.ndarray,
    x0: np.ndarray,
    config: OptimizationConfig,
) -> np.ndarray | None:
    n = len(x0)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    for row in industry_matrix:
        constraints.append(
            {"type": "ineq", "fun": lambda w, r=row: config.max_industry_weight - float(r @ w)}
        )

    result = minimize(
        fun=lambda w: portfolio_variance(w, covariance),
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, config.max_weight)] * n,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500, "disp": False},
    )
    if not result.success:
        return None
    return np.asarray(result.x, dtype=float)


def optimize_weights(
    mu: np.ndarray,
    covariance: np.ndarray,
    industries: pd.Series,
    config: OptimizationConfig,
) -> tuple[np.ndarray | None, dict[str, object]]:
    n = len(mu)
    industry_matrix, industry_names = build_industry_matrix(industries)
    x0 = find_linear_feasible_weight(industry_matrix, config)
    if x0 is None:
        return None, {"status": "linear_constraints_infeasible", "industry_count": len(industry_names)}

    min_var_weight = minimize_variance(covariance, industry_matrix, x0, config)
    if min_var_weight is None:
        return None, {"status": "min_variance_failed", "industry_count": len(industry_names)}

    min_var = portfolio_variance(min_var_weight, covariance)
    if min_var > config.max_variance * 1.000001:
        return None, {
            "status": "variance_constraint_infeasible",
            "min_variance": min_var,
            "industry_count": len(industry_names),
        }

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: config.max_variance - portfolio_variance(w, covariance)},
    ]
    for row in industry_matrix:
        constraints.append(
            {"type": "ineq", "fun": lambda w, r=row: config.max_industry_weight - float(r @ w)}
        )

    result = minimize(
        fun=lambda w: -float(mu @ w),
        x0=min_var_weight,
        method="SLSQP",
        bounds=[(0.0, config.max_weight)] * n,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500, "disp": False},
    )
    if not result.success:
        return None, {
            "status": "return_optimization_failed",
            "message": result.message,
            "min_variance": min_var,
            "industry_count": len(industry_names),
        }

    weights = np.asarray(result.x, dtype=float)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights = weights / weights.sum()
    return weights, {
        "status": "ok",
        "min_variance": min_var,
        "optimized_variance": portfolio_variance(weights, covariance),
        "industry_count": len(industry_names),
    }


def select_candidates(
    day_data: pd.DataFrame,
    config: OptimizationConfig,
) -> pd.DataFrame:
    data = day_data.dropna(subset=[config.factor_col, "expected_return", "realized_return"]).copy()
    data = data.sort_values(config.factor_col, ascending=False)
    return data.head(config.candidate_count)


def summarize_portfolio_returns(
    portfolio_returns: pd.DataFrame,
    config: OptimizationConfig,
) -> pd.DataFrame:
    series = portfolio_returns["realized_return"].dropna()
    if series.empty:
        return pd.DataFrame()

    cumulative = series.cumsum()
    drawdown = cumulative - cumulative.cummax()
    std = series.std(ddof=1)
    summary = {
        "periods": len(series),
        "total_additive_return": series.sum(),
        "annual_return": series.mean() * config.trading_days_per_year,
        "annual_vol": std * np.sqrt(config.trading_days_per_year),
        "sharpe": series.mean() / std * np.sqrt(config.trading_days_per_year) if std > 0 else np.nan,
        "max_drawdown_additive": drawdown.min(),
        "win_rate": series.gt(0).mean(),
        "mean_expected_return": portfolio_returns["expected_return"].mean(),
        "mean_optimized_variance": portfolio_returns["optimized_variance"].mean(),
    }
    return pd.DataFrame([summary])


def run_optimization_backtest(config: OptimizationConfig) -> dict[str, pd.DataFrame]:
    panel = load_optimization_panel(config)
    returns_wide = make_return_matrix(panel)
    dates = returns_wide.index

    if config.start_date is not None:
        dates = dates[dates >= pd.Timestamp(config.start_date)]
    if config.end_date is not None:
        dates = dates[dates <= pd.Timestamp(config.end_date)]
    if config.max_dates is not None:
        dates = dates[: config.max_dates]

    panel_by_date = {date: group for date, group in panel.groupby("date", sort=False)}
    return_rows: list[dict[str, object]] = []
    weight_frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []

    for date in dates:
        day_data = panel_by_date.get(date)
        if day_data is None:
            continue

        candidates = select_candidates(day_data, config)
        if len(candidates) < max(2, int(np.ceil(1.0 / config.max_weight))):
            report_rows.append({"date": date, "status": "too_few_candidates", "selected_count": len(candidates)})
            continue

        current_position = returns_wide.index.get_loc(date)
        history = returns_wide.iloc[max(0, current_position - config.risk_window) : current_position]
        history = history.reindex(columns=candidates["code"].tolist())
        covariance, valid_codes = estimate_covariance_matrix(history, config)
        if covariance is None:
            report_rows.append({"date": date, "status": "insufficient_history", "selected_count": len(candidates)})
            continue

        candidates = candidates.set_index("code").loc[valid_codes].reset_index()
        mu = candidates["expected_return"].to_numpy(dtype=float)
        weights, info = optimize_weights(mu, covariance, candidates[config.industry_col], config)
        report_row = {
            "date": date,
            "selected_count": len(candidates),
            **info,
        }
        report_rows.append(report_row)
        if weights is None:
            continue

        expected_return = float(mu @ weights)
        realized_return = float(candidates["realized_return"].to_numpy(dtype=float) @ weights)
        optimized_variance = portfolio_variance(weights, covariance)
        return_rows.append(
            {
                "date": date,
                "expected_return": expected_return,
                "realized_return": realized_return,
                "optimized_variance": optimized_variance,
                "selected_count": len(candidates),
                "holding_count": int(np.sum(weights > 1e-8)),
            }
        )

        weights_df = candidates[
            ["date", "code", config.factor_col, "expected_return", "realized_return", config.industry_col]
        ].copy()
        weights_df["weight"] = weights
        weights_df = weights_df.loc[weights_df["weight"] > 1e-8]
        weight_frames.append(weights_df)

    portfolio_returns = pd.DataFrame(return_rows)
    if not portfolio_returns.empty:
        portfolio_returns["cumulative_return"] = portfolio_returns["realized_return"].cumsum()
    portfolio_weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    optimization_report = pd.DataFrame(report_rows)
    summary = summarize_portfolio_returns(portfolio_returns, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_returns.to_csv(config.output_dir / "portfolio_returns.csv", index=False)
    portfolio_weights.to_parquet(config.output_dir / "portfolio_weights.parquet", index=False)
    optimization_report.to_csv(config.output_dir / "optimization_report.csv", index=False)
    summary.to_csv(config.output_dir / "portfolio_summary.csv", index=False)

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_weights": portfolio_weights,
        "optimization_report": optimization_report,
        "portfolio_summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained portfolio optimization backtest.")
    parser.add_argument("--expected-return-file", type=Path, default=DEFAULT_EXPECTED_RETURN_FILE)
    parser.add_argument("--processed-data-file", type=Path, default=DEFAULT_PROCESSED_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--factor-col", default="factor_zscore")
    parser.add_argument("--industry-col", default="industry_level1")
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--risk-window", type=int, default=60)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--max-variance", type=float, default=0.0004)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--max-industry-weight", type=float, default=0.20)
    parser.add_argument("--shrinkage", type=float, default=0.20)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-dates", type=int)
    return parser.parse_args()


def main() -> dict[str, pd.DataFrame]:
    args = parse_args()
    config = OptimizationConfig(
        expected_return_file=args.expected_return_file,
        processed_data_file=args.processed_data_file,
        output_dir=args.output_dir,
        factor_col=args.factor_col,
        industry_col=args.industry_col,
        candidate_count=args.candidate_count,
        risk_window=args.risk_window,
        min_history=args.min_history,
        max_variance=args.max_variance,
        max_weight=args.max_weight,
        max_industry_weight=args.max_industry_weight,
        shrinkage=args.shrinkage,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=args.max_dates,
    )
    return run_optimization_backtest(config)


if __name__ == "__main__":
    main()
