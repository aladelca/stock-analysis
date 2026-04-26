from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis.backtest.cashflows import simulate_benchmark_value_path
from stock_analysis.ml.evaluation import (
    EvaluationConfig,
    benchmark_relative_metrics,
    evaluate,
)


def test_evaluate_reports_reproducible_predictive_metrics() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=50)
    prediction = rng.normal(size=len(dates) * 5)
    target = 0.05 * prediction + np.sqrt(1 - 0.05**2) * rng.normal(size=len(prediction))
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, 5),
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"] * len(dates),
            "prediction": prediction,
            "target": target,
        }
    )

    metrics = evaluate(
        frame,
        config=EvaluationConfig(bootstrap_samples=100, random_seed=7),
    )
    repeated = evaluate(
        frame,
        config=EvaluationConfig(bootstrap_samples=100, random_seed=7),
    )

    assert metrics["predictive"]["pearson_ic"] == pytest.approx(
        frame["prediction"].corr(frame["target"])
    )
    assert metrics["predictive"]["pearson_ic_ci_95"] == repeated["predictive"]["pearson_ic_ci_95"]
    assert "rank_ic" in metrics["predictive"]


def test_benchmark_relative_information_ratio_uses_aligned_active_returns() -> None:
    portfolio = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-08", "2026-01-15"],
            "portfolio_return": [0.03, 0.01, -0.02],
        }
    )
    benchmark = pd.DataFrame(
        {
            # Deliberately unordered to prove the function aligns by date, not position.
            "date": ["2026-01-15", "2026-01-01", "2026-01-08"],
            "benchmark_return": [0.00, 0.01, 0.00],
        }
    )

    metrics = benchmark_relative_metrics(
        portfolio,
        benchmark,
        EvaluationConfig(periods_per_year=52),
    )

    active = np.array([0.02, 0.01, -0.02])
    expected_active_return = float(active.mean() * 52)
    expected_tracking_error = float(active.std(ddof=1) * np.sqrt(52))
    assert metrics["observations"] == 3
    assert metrics["active_return"] == pytest.approx(expected_active_return)
    assert metrics["tracking_error"] == pytest.approx(expected_tracking_error)
    assert metrics["information_ratio"] == pytest.approx(
        expected_active_return / expected_tracking_error
    )


def test_benchmark_cashflow_twr_is_net_of_commissions() -> None:
    benchmark = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-08"],
            "benchmark_return": [0.0, 0.0],
        }
    )

    metrics = simulate_benchmark_value_path(
        benchmark,
        initial_value=1000.0,
        contribution_by_date={},
        commission_rate=0.02,
    )

    assert metrics["benchmark_ending_value"] == pytest.approx(980.0)
    assert metrics["benchmark_time_weighted_return"] == pytest.approx(-0.02)
    assert metrics["benchmark_money_weighted_return"] < 0
