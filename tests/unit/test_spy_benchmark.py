from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.benchmarks.spy import (
    benchmark_return_horizon,
    build_benchmark_returns,
    build_spy_daily,
)


def test_build_spy_daily_and_benchmark_returns() -> None:
    dates = pd.bdate_range("2026-01-01", periods=10)
    prices = pd.DataFrame(
        {
            "ticker": ["SPY"] * len(dates),
            "date": [date.date().isoformat() for date in dates],
            "adj_close": [100 + i for i in range(len(dates))],
            "as_of_date": "2026-01-15",
        }
    )

    spy_daily = build_spy_daily(prices)
    benchmark_returns = build_benchmark_returns(spy_daily, horizons=[5])

    assert spy_daily.loc[1, "return_1d"] == pytest.approx(101 / 100 - 1)
    expected = 105 / 100 - 1
    assert benchmark_returns.loc[0, "spy_return"] == pytest.approx(expected)
    assert benchmark_return_horizon(benchmark_returns, "2026-01-01", 5) == pytest.approx(expected)
