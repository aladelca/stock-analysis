from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.forecasting.outcomes import attach_forecast_outcomes


def test_attach_forecast_outcomes_marks_realized_after_horizon() -> None:
    recommendations = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "as_of_date": "2026-01-02",
                "expected_return": 0.02,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            *[
                {
                    "ticker": "AAA",
                    "date": date.date().isoformat(),
                    "adj_close": 100 + index,
                }
                for index, date in enumerate(pd.bdate_range("2026-01-02", periods=7))
            ],
            *[
                {
                    "ticker": "SPY",
                    "date": date.date().isoformat(),
                    "adj_close": 200 + index,
                }
                for index, date in enumerate(pd.bdate_range("2026-01-02", periods=7))
            ],
        ]
    )

    result = attach_forecast_outcomes(
        recommendations,
        prices,
        horizon_days=5,
        run_data_as_of_date="2026-01-02",
    )

    row = result.iloc[0]
    assert row["forecast_horizon_days"] == 5
    assert row["forecast_start_date"] == "2026-01-02"
    assert row["forecast_end_date"] == "2026-01-09"
    assert row["realized_return"] == pytest.approx(0.05)
    assert row["realized_spy_return"] == pytest.approx(0.025)
    assert row["realized_active_return"] == pytest.approx(0.025)
    assert row["forecast_error"] == pytest.approx(0.03)
    assert bool(row["forecast_hit"]) is True
    assert row["outcome_status"] == "realized"


def test_attach_forecast_outcomes_marks_pending_without_future_prices() -> None:
    recommendations = pd.DataFrame(
        [{"ticker": "AAA", "as_of_date": "2026-01-02", "expected_return": 0.02}]
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2026-01-02", "adj_close": 100.0},
            {"ticker": "SPY", "date": "2026-01-02", "adj_close": 200.0},
        ]
    )

    result = attach_forecast_outcomes(
        recommendations,
        prices,
        horizon_days=5,
        run_data_as_of_date="2026-01-02",
    )

    row = result.iloc[0]
    assert pd.isna(row["forecast_end_date"])
    assert pd.isna(row["realized_return"])
    assert row["outcome_status"] == "pending"
