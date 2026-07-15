from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_analysis.config import PipelineConfig
from factor_analysis.backtest import run_group_backtest
from factor_analysis.constants import FACTOR_CACHE_DIR, OUTPUT_DIR
from factor_analysis.data_loader import resolve_data_file
from factor_analysis.dataset import build_factor_dataset
from 二次规划组合优化.expected_return import (
    ExpectedReturnConfig,
    estimate_daily_factor_returns,
)


portfolio_optimization = importlib.import_module("二次规划组合优化.portfolio_optimization")


class OutputLayoutTests(unittest.TestCase):
    def test_default_outputs_are_classified_under_output_root(self) -> None:
        self.assertEqual(OUTPUT_DIR.name, "单因子回测")
        self.assertEqual(OUTPUT_DIR.parent.name, "output")
        self.assertEqual(
            portfolio_optimization.DEFAULT_OUTPUT_DIR.name,
            "二次规划组合优化",
        )
        self.assertEqual(
            portfolio_optimization.DEFAULT_OUTPUT_DIR.parent.name,
            "output",
        )
        self.assertEqual(FACTOR_CACHE_DIR.parent.name, ".cache")
        self.assertEqual(
            portfolio_optimization.DEFAULT_EXPECTED_RETURN_FILE.parents[1].name,
            ".cache",
        )

    def test_revised_strategy_defaults(self) -> None:
        pipeline_config = PipelineConfig()
        expected_config = ExpectedReturnConfig()
        optimization_config = portfolio_optimization.OptimizationConfig()
        self.assertEqual(pipeline_config.winsor_mad_n, 3.0)
        self.assertEqual(expected_config.factor_col, "factor_rank_zscore")
        self.assertEqual(optimization_config.factor_col, "factor_rank_zscore")
        self.assertEqual(optimization_config.max_weight, 0.03)
        self.assertEqual(optimization_config.max_variance, 0.0004)


class DataPathTests(unittest.TestCase):
    def test_resolve_flat_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            expected = data_dir / "daily_data.parquet"
            expected.touch()
            self.assertEqual(resolve_data_file(data_dir, expected.name), expected)

    def test_resolve_nested_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            nested = data_dir / "basic_data"
            nested.mkdir()
            expected = nested / "daily_data.parquet"
            expected.touch()
            self.assertEqual(resolve_data_file(data_dir, expected.name), expected)


class FactorDirectionTests(unittest.TestCase):
    def test_factor_sign_is_applied_during_construction(self) -> None:
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        daily = pd.DataFrame(
            {
                "date": dates,
                "code": ["000001"] * 4,
                "open": [10.0, 11.0, 12.0, 13.0],
                "close": [10.0, 12.0, 15.0, 18.0],
                "market_value": [100.0] * 4,
                "float_market_value": [80.0] * 4,
                "chg_status": [0] * 4,
            }
        )
        industry = pd.DataFrame(
            {"date": dates, "code": ["000001"] * 4, "industry_level1": ["A"] * 4}
        )
        st_flags = pd.DataFrame(columns=["date", "code", "is_st"])
        suspension_flags = pd.DataFrame(columns=["date", "code", "is_suspended"])
        config = PipelineConfig(factor_lag=1, factor_sign=-1.0, min_listed_days=0)

        result = build_factor_dataset(
            daily,
            industry,
            st_flags,
            suspension_flags,
            config,
        )

        self.assertAlmostEqual(result.iloc[0]["factor"], -(12.0 / 10.0 - 1.0))

    def test_group_one_contains_highest_processed_factor(self) -> None:
        date = pd.Timestamp("2020-01-01")
        panel = pd.DataFrame(
            {
                "date": [date] * 5,
                "code": [f"{code:06d}" for code in range(5)],
                "factor_zscore": [5.0, 4.0, 3.0, 2.0, 1.0],
                "return": [0.05, 0.04, 0.03, 0.02, 0.01],
                "weight_float_mv": [1.0] * 5,
                "industry_level1": ["A"] * 5,
            }
        )
        config = PipelineConfig(
            groups=5,
            backtest_start_day=1,
            weights=("equal",),
        )

        returns, _, _ = run_group_backtest(panel, ["factor_zscore"], config)

        group_one = returns.loc[returns["portfolio"].eq("group_1"), "return"].iloc[0]
        self.assertAlmostEqual(group_one, 0.05)


class TimeAlignmentTests(unittest.TestCase):
    def test_beta_prediction_uses_only_two_period_old_labels(self) -> None:
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        panel = pd.DataFrame(
            {
                "date": np.repeat(dates, 2),
                "factor_rank_zscore": [-1.0, 1.0] * 4,
                "realized_return": [-0.01, 0.01, -0.02, 0.02, -0.03, 0.03, -0.04, 0.04],
            }
        )
        config = ExpectedReturnConfig(window=1, min_periods=1, return_availability_lag=2)

        result = estimate_daily_factor_returns(panel, config)

        self.assertTrue(result.loc[:1, "beta_hat"].isna().all())
        self.assertAlmostEqual(result.loc[2, "beta_hat"], result.loc[0, "realized_beta"])
        self.assertAlmostEqual(result.loc[3, "beta_hat"], result.loc[1, "realized_beta"])

    def test_covariance_history_ends_at_t_minus_two(self) -> None:
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        returns = pd.DataFrame({"000001": range(6)}, index=dates)

        history = portfolio_optimization.select_observable_return_history(
            returns,
            dates[5],
            risk_window=3,
        )

        self.assertEqual(list(history.index), list(dates[1:4]))


if __name__ == "__main__":
    unittest.main()
