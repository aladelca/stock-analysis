from __future__ import annotations

import json
from collections.abc import Mapping
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
MANIFEST_FILENAME = "manifest.json"
CURRENT_POINTER_FILENAME = "current.json"
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
        return f"{self.production_root_uri()}/{CURRENT_POINTER_FILENAME}"

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
            current_uri = self._write_current_pointer(
                manifest_uri=uris["manifest"],
                run_id=run_id,
                config_hash=config_hash,
            )
            uris["production_current"] = current_uri
        return uris

    def load_artifact(self, model_uri: str) -> MLForecastModelArtifact:
        uri = self._resolve_model_uri(model_uri)
        payload = self._blob(uri).download_as_bytes()
        artifact = cloudpickle.loads(payload)
        if not isinstance(artifact, MLForecastModelArtifact):
            msg = f"GCS object is not an ML forecast model artifact: {uri}"
            raise ValueError(msg)
        return artifact

    def exists(self, uri: str) -> bool:
        if uri.endswith(f"/{CURRENT_POINTER_FILENAME}") or uri.endswith(f"/{MANIFEST_FILENAME}"):
            try:
                self._resolve_model_uri(uri)
            except ValueError:
                return False
            return True
        pointer_uri = f"{uri.rstrip('/')}/{CURRENT_POINTER_FILENAME}"
        if self._blob(pointer_uri).exists():
            try:
                self._resolve_model_uri(pointer_uri)
            except ValueError:
                return False
            return True
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
        manifest_uri = f"{root_uri}/{MANIFEST_FILENAME}"

        _upload_string(
            self._blob(model_uri),
            cloudpickle.dumps(artifact),
            content_type="application/octet-stream",
            if_generation_match=0,
        )
        _upload_string(
            self._blob(metadata_uri),
            _json_dumps(metadata),
            content_type="application/json",
            if_generation_match=0,
        )
        _upload_parquet(
            self._blob(diagnostics_uri),
            artifact.calibration_diagnostics,
            if_generation_match=0,
        )
        _upload_parquet(
            self._blob(predictions_uri),
            artifact.calibration_predictions,
            if_generation_match=0,
        )
        uris = {
            "model": model_uri,
            "metadata": metadata_uri,
            "calibration_diagnostics": diagnostics_uri,
            "calibration_predictions": predictions_uri,
        }
        _upload_string(
            self._blob(manifest_uri),
            _json_dumps(
                _manifest_payload(root_uri=root_uri, artifact_uris=uris, metadata=metadata)
            ),
            content_type="application/json",
            if_generation_match=0,
        )
        return {**uris, "manifest": manifest_uri}

    def _write_current_pointer(
        self,
        *,
        manifest_uri: str,
        run_id: str,
        config_hash: str,
    ) -> str:
        current_uri = self.default_model_uri()
        payload: dict[str, object] = {
            "manifest_uri": manifest_uri,
            "run_id": run_id,
            "config_hash": config_hash,
        }
        _upload_string(
            self._blob(current_uri),
            _json_dumps(payload),
            content_type="application/json",
        )
        return current_uri

    def _resolve_model_uri(self, uri: str) -> str:
        if uri.endswith(f"/{CURRENT_POINTER_FILENAME}"):
            pointer = self._download_json(uri)
            manifest_uri = _required_uri(pointer, "manifest_uri", uri)
            return self._model_uri_from_manifest(manifest_uri)
        if uri.endswith(f"/{MANIFEST_FILENAME}"):
            return self._model_uri_from_manifest(uri)
        pointer_uri = f"{uri.rstrip('/')}/{CURRENT_POINTER_FILENAME}"
        if self._blob(pointer_uri).exists():
            return self._resolve_model_uri(pointer_uri)
        return _model_uri(uri)

    def _model_uri_from_manifest(self, manifest_uri: str) -> str:
        manifest = self._download_json(manifest_uri)
        artifact_uris = manifest.get("artifact_uris")
        if not isinstance(artifact_uris, dict):
            msg = f"Model manifest does not include artifact_uris: {manifest_uri}"
            raise ValueError(msg)
        required_keys = {
            "model",
            "metadata",
            "calibration_diagnostics",
            "calibration_predictions",
        }
        missing_keys = sorted(required_keys - set(artifact_uris))
        if missing_keys:
            msg = f"Model manifest {manifest_uri} is missing artifact URIs: {missing_keys}"
            raise ValueError(msg)
        for key in sorted(required_keys):
            artifact_uri = artifact_uris[key]
            if not isinstance(artifact_uri, str) or not self._blob(artifact_uri).exists():
                msg = (
                    f"Model manifest {manifest_uri} references missing artifact "
                    f"{key}: {artifact_uri}"
                )
                raise ValueError(msg)
        return str(artifact_uris["model"])

    def _download_json(self, uri: str) -> dict[str, Any]:
        blob = self._blob(uri)
        if not blob.exists():
            msg = f"GCS model registry object does not exist: {uri}"
            raise ValueError(msg)
        payload = blob.download_as_bytes().decode("utf-8")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            msg = f"GCS model registry JSON object is not a mapping: {uri}"
            raise ValueError(msg)
        return parsed

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


def _manifest_payload(
    *,
    root_uri: str,
    artifact_uris: dict[str, str],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "root_uri": root_uri,
        "artifact_uris": artifact_uris,
        "metadata": metadata,
    }


def _required_uri(payload: dict[str, Any], field_name: str, source_uri: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        msg = f"Model registry pointer {source_uri} is missing {field_name}."
        raise ValueError(msg)
    return value


def _upload_parquet(
    blob: Any,
    frame: pd.DataFrame,
    *,
    if_generation_match: int | None = None,
) -> None:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    buffer.seek(0)
    kwargs: dict[str, object] = {"content_type": "application/octet-stream"}
    if if_generation_match is not None:
        kwargs["if_generation_match"] = if_generation_match
    try:
        blob.upload_from_file(buffer, **kwargs)
    except TypeError:
        blob.upload_from_file(buffer, content_type="application/octet-stream")


def _upload_string(
    blob: Any,
    content: str | bytes,
    *,
    content_type: str,
    if_generation_match: int | None = None,
) -> None:
    kwargs: dict[str, object] = {"content_type": content_type}
    if if_generation_match is not None:
        kwargs["if_generation_match"] = if_generation_match
    try:
        blob.upload_from_string(content, **kwargs)
    except TypeError:
        blob.upload_from_string(content, content_type=content_type)


def _json_dumps(payload: Mapping[str, object]) -> str:
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
