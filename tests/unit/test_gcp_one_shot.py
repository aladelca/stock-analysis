from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_analysis.config import ForecastConfig, GcpConfig, PortfolioConfig
from stock_analysis.forecasting.ml_forecast import MLForecastModelArtifact
from stock_analysis.pipeline.gcp_one_shot import _validate_model_artifact_contract


def test_validate_model_artifact_rejects_future_trained_model() -> None:
    artifact = _artifact(trained_through_date="2026-05-02")
    config = PortfolioConfig(gcp=GcpConfig(enabled=True, bucket="bucket"))

    with pytest.raises(ValueError, match="after inference data_as_of_date"):
        _validate_model_artifact_contract(
            config,
            "gs://bucket/models/production/current.json",
            artifact,
            data_as_of_date=date(2026, 5, 1),
            feature_panel=_feature_panel(),
        )


def test_validate_model_artifact_rejects_uncalibrated_when_config_requires_calibration() -> None:
    artifact = _artifact(expected_return_is_calibrated=False)
    config = PortfolioConfig(
        forecast=ForecastConfig(
            engine="ml",
            ml_calibration_enabled=True,
            ml_use_calibrated_expected_return=True,
        ),
        gcp=GcpConfig(enabled=True, bucket="bucket"),
    )

    with pytest.raises(ValueError, match="not calibrated"):
        _validate_model_artifact_contract(
            config,
            "gs://bucket/models/production/current.json",
            artifact,
            data_as_of_date=date(2026, 5, 1),
            feature_panel=_feature_panel(),
        )


class ConstantModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def _artifact(
    *,
    trained_through_date: str = "2026-05-01",
    expected_return_is_calibrated: bool = True,
) -> MLForecastModelArtifact:
    return MLForecastModelArtifact(
        model=ConstantModel(),
        model_version="lightgbm_return_zscore",
        feature_columns=("momentum_21d",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        trained_through_date=trained_through_date,
        expected_return_is_calibrated=expected_return_is_calibrated,
        calibration_status="calibrated" if expected_return_is_calibrated else "disabled",
        calibration_method="isotonic",
        calibration_target="return",
        calibration_shrinkage=0.25,
        calibrator=object() if expected_return_is_calibrated else None,
        calibration_target_mean=0.0 if expected_return_is_calibrated else None,
        calibration_predictions=pd.DataFrame(),
        calibration_diagnostics=pd.DataFrame(),
        created_at_utc="2026-05-01T00:00:00+00:00",
    )


def _feature_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["SPY"],
            "date": ["2026-05-01"],
            "momentum_21d": [0.01],
        }
    )
