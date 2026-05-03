from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analysis.config import GcpConfig
from stock_analysis.forecasting.ml_forecast import MLForecastModelArtifact
from stock_analysis.gcp.model_registry import GcsModelRegistry


def test_gcs_model_registry_writes_versioned_and_production_artifacts() -> None:
    client = FakeStorageClient()
    artifact = MLForecastModelArtifact(
        model=ConstantModel(0.01),
        model_version="lightgbm_return_zscore",
        feature_columns=("momentum_21d",),
        target_column="fwd_return_5d",
        horizon_days=5,
        score_scale=1.0,
        trained_through_date="2026-04-24",
        expected_return_is_calibrated=False,
        calibration_status="disabled",
        calibration_method="isotonic",
        calibration_target="return",
        calibration_shrinkage=0.0,
        calibrator=None,
        calibration_target_mean=None,
        calibration_predictions=pd.DataFrame({"ticker": []}),
        calibration_diagnostics=pd.DataFrame({"calibration_status": ["disabled"]}),
        created_at_utc="2026-04-24T00:00:00+00:00",
    )
    registry = GcsModelRegistry(
        GcpConfig(bucket="stock-analysis-medallion-prod", model_registry_prefix="models"),
        storage_client=client,
    )

    uris = registry.write_artifact(
        artifact,
        run_id="train-run-1",
        config_hash="abc123",
        promote=True,
    )

    assert uris["model"] == (
        "gs://stock-analysis-medallion-prod/models/runs/train-run-1/model.cloudpickle"
    )
    assert uris["production_model"] == (
        "gs://stock-analysis-medallion-prod/models/production/model.cloudpickle"
    )
    assert registry.exists(registry.default_model_uri())
    loaded = registry.load_artifact(registry.default_model_uri())
    assert loaded.model_version == artifact.model_version
    np.testing.assert_allclose(
        loaded.model.predict(pd.DataFrame({"momentum_21d": [1, 2]})), [0.01, 0.01]
    )


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        bucket = self.buckets.get(name)
        if bucket is None:
            bucket = FakeBucket()
            self.buckets[name] = bucket
        return bucket


class FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        blob = self.blobs.get(name)
        if blob is None:
            blob = FakeBlob()
            self.blobs[name] = blob
        return blob


class FakeBlob:
    def __init__(self) -> None:
        self.content = b""

    def upload_from_string(self, content: str | bytes, content_type: str | None = None) -> None:
        del content_type
        self.content = content if isinstance(content, bytes) else content.encode("utf-8")

    def upload_from_file(self, file_obj, content_type: str | None = None) -> None:
        del content_type
        self.content = file_obj.read()

    def download_as_bytes(self) -> bytes:
        return self.content

    def exists(self) -> bool:
        return bool(self.content)
