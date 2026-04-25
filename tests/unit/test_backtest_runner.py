from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pytest

from stock_analysis.backtest.cashflows import (
    ContributionSchedule,
    contributions_for_rebalance_dates,
    money_weighted_return,
)
from stock_analysis.backtest.runner import BacktestConfig, run_walk_forward_backtest
from stock_analysis.config import OptimizerConfig


@dataclass
class MomentumModel:
    def predict(self, features: pd.DataFrame) -> list[float]:
        return features["momentum_5d"].astype(float).tolist()


def test_walk_forward_backtest_records_weights_and_costs() -> None:
    dates = pd.bdate_range("2025-01-01", periods=45)
    tickers = ["AAA", "BBB", "CCC"]
    panel_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for ticker_idx, ticker in enumerate(tickers):
        for current_date in dates:
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "momentum_5d": 0.1 - ticker_idx * 0.02,
                    "volatility_21d": 0.2,
                    "security": ticker,
                    "gics_sector": "Sector",
                }
            )
            return_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "return_1d": 0.001 + ticker_idx * 0.0001,
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "fwd_return_5d": 0.01 + ticker_idx * 0.001,
                }
            )

    result = run_walk_forward_backtest(
        pd.DataFrame(panel_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(return_rows),
        lambda train: MomentumModel(),
        OptimizerConfig(
            max_weight=0.6,
            risk_aversion=0.1,
            lambda_turnover=0.001,
            commission_rate=0.02,
        ),
        BacktestConfig(
            horizon_days=5,
            embargo_days=10,
            commission_rate=0.02,
            covariance_lookback_days=20,
            max_rebalances=3,
        ),
    )

    assert not result.empty
    assert result.groupby("rebalance_date")["target_weight"].sum().iloc[0] == pytest.approx(1.0)
    assert {
        "portfolio_gross_return",
        "portfolio_net_return",
        "turnover",
        "trade_abs_weight",
        "commission_rate",
    } <= set(result.columns)
    first = result.drop_duplicates("rebalance_date").iloc[0]
    assert first["transaction_cost"] == pytest.approx(0.02 * first["trade_abs_weight"])
    assert first["portfolio_net_return"] == pytest.approx(
        first["portfolio_gross_return"] - first["transaction_cost"]
    )


def test_walk_forward_backtest_tracks_contributions_and_value_path() -> None:
    dates = pd.bdate_range("2025-01-01", periods=65)
    tickers = ["AAA", "BBB", "CCC"]
    panel_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for ticker_idx, ticker in enumerate(tickers):
        for current_date in dates:
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "momentum_5d": 0.2 - ticker_idx * 0.05,
                    "volatility_21d": 0.2,
                    "security": ticker,
                    "gics_sector": "Sector",
                }
            )
            return_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "return_1d": 0.001,
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "fwd_return_5d": 0.01 + ticker_idx * 0.001,
                }
            )

    result = run_walk_forward_backtest(
        pd.DataFrame(panel_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(return_rows),
        lambda train: MomentumModel(),
        OptimizerConfig(max_weight=0.6, risk_aversion=0.1, commission_rate=0.02),
        BacktestConfig(
            horizon_days=5,
            embargo_days=10,
            commission_rate=0.02,
            covariance_lookback_days=20,
            max_rebalances=5,
            initial_portfolio_value=1000,
            monthly_deposit_amount=100,
            deposit_frequency_days=30,
            deposit_start_date=dates[0].date(),
        ),
    )

    periods = result.drop_duplicates("rebalance_date")
    assert periods["external_contribution"].sum() >= 100
    assert periods["strategy_ending_value"].iloc[-1] > 0
    assert periods["total_deposits"].iloc[-1] == pytest.approx(
        periods["external_contribution"].sum()
    )
    assert periods["total_commissions"].iloc[-1] == pytest.approx(periods["commission_paid"].sum())
    assert "money_weighted_return" in result.columns


def test_contributions_map_to_next_rebalance_date() -> None:
    rebalance_dates = pd.DatetimeIndex(["2025-01-03", "2025-01-17", "2025-02-03"])

    mapped = contributions_for_rebalance_dates(
        rebalance_dates,
        ContributionSchedule(amount=100, frequency_days=30, start_date=date(2025, 1, 4)),
    )

    assert mapped[pd.Timestamp("2025-01-03")] == 0
    assert mapped[pd.Timestamp("2025-01-17")] == 100
    assert mapped[pd.Timestamp("2025-02-03")] == 100


def test_money_weighted_return_solves_simple_cashflows() -> None:
    value = money_weighted_return([(date(2025, 1, 1), -1000), (date(2026, 1, 1), 1100)])

    assert value == pytest.approx(0.0997, abs=0.001)
