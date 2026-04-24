from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.ml.labels import build_forward_return_labels


def test_forward_return_labels_are_shifted_per_ticker() -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": date.date().isoformat(),
                "adj_close": 100 + offset + day,
            }
            for offset, ticker in enumerate(["AAA", "BBB"])
            for day, date in enumerate(dates)
        ]
    )
    panel = prices[["ticker", "date", "adj_close"]].copy()
    benchmark_returns = pd.DataFrame(
        {
            "date": [date.date().isoformat() for date in dates],
            "horizon_days": 5,
            "spy_return": [0.01] * len(dates),
        }
    )

    labels = build_forward_return_labels(
        prices,
        panel,
        benchmark_returns=benchmark_returns,
        horizons=[5],
    )

    aaa_first = labels[(labels["ticker"] == "AAA") & (labels["date"] == "2026-01-01")].iloc[0]
    expected_return = 105 / 100 - 1
    assert aaa_first["fwd_return_5d"] == pytest.approx(expected_return)
    assert aaa_first["fwd_excess_return_5d"] == pytest.approx(expected_return - 0.01)
    assert labels["fwd_rank_5d"].dropna().between(1, 2).all()
    assert labels["fwd_is_top_tercile_5d"].dropna().isin([0, 1]).all()
