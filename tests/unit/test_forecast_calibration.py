from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analysis.forecasting.calibration import (
    build_oos_prediction_frame,
    calibrate_forecast_scores,
)


class SignalModel:
    def __init__(self, train: pd.DataFrame) -> None:
        self.train_max_date = pd.to_datetime(train["date"]).max()

    def predict(self, features: pd.DataFrame) -> list[float]:
        return pd.to_numeric(features["signal"], errors="coerce").astype(float).tolist()


def test_calibrate_forecast_scores_maps_scores_to_expected_returns() -> None:
    panel, labels = _panel_and_labels()
    latest = panel.loc[panel["date"].eq(panel["date"].max())].copy()
    latest["forecast_score"] = latest["signal"]

    result = calibrate_forecast_scores(
        panel=panel,
        labels=labels,
        latest_features=latest,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        method="isotonic",
        min_observations=20,
        min_validation_observations=5,
        validation_fraction=0.25,
        min_rank_ic=0.0,
        max_mae=None,
        max_rmse=None,
        splits=4,
        embargo_days=1,
        shrinkage=0.0,
    )

    assert result.is_calibrated
    assert result.diagnostics["calibration_status"].iat[0] == "calibrated"
    assert result.diagnostics["calibration_observations"].iat[0] >= 20
    assert result.calibrated_latest.notna().all()
    ordered = result.calibrated_latest.loc[latest.sort_values("signal").index]
    assert ordered.is_monotonic_increasing


def test_calibration_shrinkage_pulls_predictions_toward_target_mean() -> None:
    panel, labels = _panel_and_labels()
    latest = panel.loc[panel["date"].eq(panel["date"].max())].copy()
    latest["forecast_score"] = latest["signal"]

    unshrunk = calibrate_forecast_scores(
        panel=panel,
        labels=labels,
        latest_features=latest,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        method="isotonic",
        min_observations=20,
        min_validation_observations=5,
        validation_fraction=0.25,
        min_rank_ic=0.0,
        max_mae=None,
        max_rmse=None,
        splits=4,
        embargo_days=1,
        shrinkage=0.0,
    )
    shrunk = calibrate_forecast_scores(
        panel=panel,
        labels=labels,
        latest_features=latest,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        method="isotonic",
        min_observations=20,
        min_validation_observations=5,
        validation_fraction=0.25,
        min_rank_ic=0.0,
        max_mae=None,
        max_rmse=None,
        splits=4,
        embargo_days=1,
        shrinkage=1.0,
    )

    assert shrunk.calibrated_latest.nunique() == 1
    assert np.isfinite(shrunk.calibrated_latest.iat[0])
    assert unshrunk.calibrated_latest.nunique() > 1


def test_calibration_falls_back_when_observations_are_insufficient() -> None:
    panel, labels = _panel_and_labels()
    latest = panel.loc[panel["date"].eq(panel["date"].max())].copy()
    latest["forecast_score"] = latest["signal"]

    result = calibrate_forecast_scores(
        panel=panel,
        labels=labels,
        latest_features=latest,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        method="isotonic",
        min_observations=10_000,
        min_validation_observations=5,
        validation_fraction=0.25,
        min_rank_ic=0.0,
        max_mae=None,
        max_rmse=None,
        splits=4,
        embargo_days=1,
        shrinkage=0.0,
    )

    assert not result.is_calibrated
    assert result.diagnostics["calibration_status"].iat[0] == "insufficient_observations"
    assert result.calibrated_latest.isna().all()


def test_calibration_falls_back_when_quality_gate_fails() -> None:
    panel, labels = _panel_and_labels()
    latest = panel.loc[panel["date"].eq(panel["date"].max())].copy()
    latest["forecast_score"] = latest["signal"]

    result = calibrate_forecast_scores(
        panel=panel,
        labels=labels,
        latest_features=latest,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        method="isotonic",
        min_observations=20,
        min_validation_observations=5,
        validation_fraction=0.25,
        min_rank_ic=1.01,
        max_mae=None,
        max_rmse=None,
        splits=4,
        embargo_days=1,
        shrinkage=0.0,
    )

    assert not result.is_calibrated
    assert result.diagnostics["calibration_status"].iat[0] == "failed_quality_gate"
    assert result.calibrated_latest.isna().all()
    assert set(result.predictions["calibration_sample"].dropna()) == {"fit", "validation"}


def test_oos_prediction_frame_respects_embargoed_train_cutoff() -> None:
    panel, labels = _panel_and_labels()

    predictions = build_oos_prediction_frame(
        panel=panel,
        labels=labels,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        score_scale=1.0,
        splits=4,
        embargo_days=2,
    )

    assert not predictions.empty
    validation_dates = pd.to_datetime(predictions["date"])
    train_cutoffs = pd.to_datetime(predictions["train_cutoff_date"])
    assert (train_cutoffs < validation_dates).all()


def test_oos_prediction_frame_uses_point_in_time_liquidity_universe() -> None:
    panel, labels = _panel_and_labels()

    predictions = build_oos_prediction_frame(
        panel=panel,
        labels=labels,
        model_factory=SignalModel,
        feature_columns=("signal",),
        target_column="fwd_return_5d",
        score_scale=1.0,
        splits=4,
        embargo_days=2,
        max_assets_per_date=2,
    )

    assert not predictions.empty
    assert predictions.groupby("date")["ticker"].nunique().max() <= 2


def _panel_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-01", periods=40)
    tickers = [f"T{index}" for index in range(5)]
    panel_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for date_idx, current_date in enumerate(dates):
        for ticker_idx, ticker in enumerate(tickers):
            signal = ticker_idx - 2 + np.sin(date_idx / 5)
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date,
                    "signal": signal,
                    "dollar_volume_21d": ticker_idx * 1_000_000 + date_idx,
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date,
                    "fwd_return_5d": 0.01 * signal if date_idx < len(dates) - 5 else np.nan,
                }
            )
    return pd.DataFrame(panel_rows), pd.DataFrame(label_rows)
