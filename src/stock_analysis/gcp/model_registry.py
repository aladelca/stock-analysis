from __future__ import annotations

from dataclasses import fields
from io import BytesIO
from typing import Any

import cloudpickle
import pandas as pd

from stock_analysis.config import GcpConfig
from stock_analysis.forecasting.ml_forecast import MLForecastModelArtifact
from stock_analysis.gcp.gcs_store import _parse_gcs_uri

MODEL_FILENAME = "model.cloudpickle"
METADATA_FILENAME = "metadata.json"
CALIBRATION_DIAGNOSTICS_FILENAME = "calibration_diagnostics.parquet"
CALIBRATION_PREDICTIONS_FILENAME = "calibration_predictions.parquet"


class GcsModelRegistry:
    """GCS-backed model registry with a production pointer path."""

    def __init__(
        self,
        config: GcpConfig,
        *,
        storage_client: Any | None = None,
    ) -> None:
        if not config.bucket:
            msg = "Set gcp.bucket before using the GCS model registry."
            raise ValueError(msg)
        self.bucket_name = config.bucket
        self.prefix = config.model_registry_prefix.strip("/")
        self._client = storage_client or _storage_client()
        self._bucket = self._client.bucket(self.bucket_name)

    def run_root_uri(self, run_id: str) -> str:
        return self._root_uri(f"runs/{run_id}")

    def production_root_uri(self) -> str:
        return self._root_uri("production")

    def default_model_uri(self) -> str:
        return f"{self.production_root_uri()}/{MODEL_FILENAME}"

    def write_artifact(
        self,
        artifact: MLForecastModelArtifact,
        *,
        run_id: str,
        config_hash: str,
        promote: bool,
    ) -> dict[str, str]:
        run_root = self.run_root_uri(run_id)
        uris = self._write_bundle(
            artifact,
            root_uri=run_root,
            metadata={
                **_artifact_metadata(artifact),
                "run_id": run_id,
                "config_hash": config_hash,
                "promoted_to_production": promote,
            },
        )
        if promote:
            production_uris = self._write_bundle(
                artifact,
                root_uri=self.production_root_uri(),
                metadata={
                    **_artifact_metadata(artifact),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "promoted_to_production": True,
                    "source_model_uri": uris["model"],
                },
            )
            uris.update({f"production_{key}": value for key, value in production_uris.items()})
        return uris

    def load_artifact(self, model_uri: str) -> MLForecastModelArtifact:
        uri = _model_uri(model_uri)
        payload = self._blob(uri).download_as_bytes()
        artifact = cloudpickle.loads(payload)
        if not isinstance(artifact, MLForecastModelArtifact):
            msg = f"GCS object is not an ML forecast model artifact: {uri}"
            raise ValueError(msg)
        return artifact

    def exists(self, uri: str) -> bool:
        return bool(self._blob(_model_uri(uri)).exists())

    def _write_bundle(
        self,
        artifact: MLForecastModelArtifact,
        *,
        root_uri: str,
        metadata: dict[str, object],
    ) -> dict[str, str]:
        model_uri = f"{root_uri}/{MODEL_FILENAME}"
        metadata_uri = f"{root_uri}/{METADATA_FILENAME}"
        diagnostics_uri = f"{root_uri}/{CALIBRATION_DIAGNOSTICS_FILENAME}"
        predictions_uri = f"{root_uri}/{CALIBRATION_PREDICTIONS_FILENAME}"

        self._blob(model_uri).upload_from_string(
            cloudpickle.dumps(artifact),
            content_type="application/octet-stream",
        )
        self._blob(metadata_uri).upload_from_string(
            _json_dumps(metadata),
            content_type="application/json",
        )
        _upload_parquet(self._blob(diagnostics_uri), artifact.calibration_diagnostics)
        _upload_parquet(self._blob(predictions_uri), artifact.calibration_predictions)
        return {
            "model": model_uri,
            "metadata": metadata_uri,
            "calibration_diagnostics": diagnostics_uri,
            "calibration_predictions": predictions_uri,
        }

    def _root_uri(self, suffix: str) -> str:
        prefix = f"{self.prefix}/" if self.prefix else ""
        return f"gs://{self.bucket_name}/{prefix}{suffix.strip('/')}"

    def _blob(self, uri: str) -> Any:
        bucket_name, object_name = _parse_gcs_uri(uri)
        if bucket_name != self.bucket_name:
            msg = f"URI bucket {bucket_name!r} does not match model registry bucket."
            raise ValueError(msg)
        return self._bucket.blob(object_name)


def _model_uri(uri: str) -> str:
    return uri if uri.endswith(f"/{MODEL_FILENAME}") else f"{uri.rstrip('/')}/{MODEL_FILENAME}"


def _artifact_metadata(artifact: MLForecastModelArtifact) -> dict[str, object]:
    excluded = {
        "model",
        "calibrator",
        "calibration_predictions",
        "calibration_diagnostics",
    }
    return {
        field.name: getattr(artifact, field.name)
        for field in fields(artifact)
        if field.name not in excluded
    }


def _upload_parquet(blob: Any, frame: pd.DataFrame) -> None:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type="application/octet-stream")


def _json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _storage_client() -> Any:
    try:
        from google.cloud import storage
    except ImportError as exc:
        msg = (
            "GCS model registry requires the optional gcp extra. "
            "Run `uv sync --extra gcp` or prefix the command with `uv run --extra gcp`."
        )
        raise RuntimeError(msg) from exc
    return storage.Client()
