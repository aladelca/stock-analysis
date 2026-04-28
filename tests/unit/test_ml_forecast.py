from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analysis.config import ForecastConfig
from stock_analysis.forecasting.ml_forecast import (
    build_ml_optimizer_inputs,
    build_ml_optimizer_inputs_with_artifacts,
)


def test_build_ml_optimizer_inputs_uses_phase2_blend_and_liquidity_filter() -> None:
    dates = pd.bdate_range("2025-01-01", periods=80)
    tickers = [f"T{i}" for i in range(6)]
    panel_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []

    for date_idx, current_date in enumerate(dates):
        market_cycle = np.sin(date_idx / 6)
        for ticker_idx, ticker in enumerate(tickers):
            signal = ticker_idx * 0.04 + market_cycle * 0.01
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "security": ticker,
                    "gics_sector": "Sector",
                    "momentum_21d": signal,
                    "volatility_63d": 0.15 + ticker_idx * 0.01,
                    "return_5d": signal / 5,
                    "dollar_volume_21d": 1_000_000 + ticker_idx * 100_000,
                    "history_days": date_idx + 1,
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "fwd_return_5d": (
                        signal * 0.03 + ticker_idx * 0.001 if date_idx < len(dates) - 5 else np.nan
                    ),
                }
            )
            return_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "adj_close": 100 + date_idx + ticker_idx,
                    "return_1d": 0.0005 + ticker_idx * 0.0001 + market_cycle * 0.0001,
                }
            )

    raw_optimizer_input, _ = build_ml_optimizer_inputs(
        pd.DataFrame(panel_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(return_rows),
        ForecastConfig(
            engine="ml",
            covariance_lookback_days=40,
            ml_max_assets=4,
            ml_feature_columns=["momentum_21d", "volatility_63d", "return_5d"],
            ml_score_scale=1.0,
            ml_lightgbm_nested_cv=False,
        ),
    )
    optimizer_input, covariance = build_ml_optimizer_inputs(
        pd.DataFrame(panel_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(return_rows),
        ForecastConfig(
            engine="ml",
            covariance_lookback_days=40,
            ml_max_assets=4,
            ml_feature_columns=["momentum_21d", "volatility_63d", "return_5d"],
            ml_score_scale=0.5,
            ml_lightgbm_nested_cv=False,
        ),
    )

    assert len(optimizer_input) == 4
    assert optimizer_input["forecast_engine"].eq("ml").all()
    assert optimizer_input["forecast_model_version"].eq("lightgbm_return_zscore").all()
    assert not optimizer_input["expected_return_is_calibrated"].any()
    assert optimizer_input["expected_return"].notna().all()
    np.testing.assert_allclose(
        raw_optimizer_input["forecast_score"].to_numpy() * 0.5,
        optimizer_input["forecast_score"].to_numpy(),
    )
    assert optimizer_input["eligible_for_optimization"].all()
    assert set(covariance.columns) == set(optimizer_input["ticker"])
    assert set(optimizer_input["ticker"]) == {"T2", "T3", "T4", "T5"}

    calibrated = build_ml_optimizer_inputs_with_artifacts(
        pd.DataFrame(panel_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(return_rows),
        ForecastConfig(
            engine="ml",
            covariance_lookback_days=40,
            ml_max_assets=4,
            ml_feature_columns=["momentum_21d", "volatility_63d", "return_5d"],
            ml_score_scale=0.5,
            ml_lightgbm_nested_cv=False,
            ml_calibration_enabled=True,
            ml_calibration_min_observations=20,
            ml_calibration_min_validation_observations=10,
            ml_calibration_validation_fraction=0.25,
            ml_calibration_min_rank_ic=-1.0,
            ml_calibration_splits=3,
            ml_calibration_embargo_days=1,
            ml_calibration_shrinkage=0.0,
        ),
    )

    assert calibrated.optimizer_input["expected_return_is_calibrated"].all()
    assert calibrated.optimizer_input["calibrated_expected_return"].notna().all()
    assert calibrated.optimizer_input["expected_return"].notna().all()
    assert calibrated.calibration_diagnostics["calibration_status"].iat[0] == "calibrated"
    assert calibrated.calibration_diagnostics["calibration_observations"].iat[0] >= 20
    assert not calibrated.calibration_predictions.empty
