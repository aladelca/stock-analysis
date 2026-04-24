from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

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
        OptimizerConfig(max_weight=0.6, risk_aversion=0.1, lambda_turnover=0.001),
        BacktestConfig(
            horizon_days=5,
            embargo_days=10,
            covariance_lookback_days=20,
            max_rebalances=3,
        ),
    )

    assert not result.empty
    assert result.groupby("rebalance_date")["target_weight"].sum().iloc[0] == pytest.approx(1.0)
    assert {"portfolio_gross_return", "portfolio_net_return", "turnover"} <= set(result.columns)
