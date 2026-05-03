from __future__ import annotations

import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


class GcsArtifactStore:
    """Medallion artifact store backed directly by Google Cloud Storage."""

    def __init__(
        self,
        *,
        bucket: str,
        run_id: str,
        prefix: str = "runs",
        storage_client: Any | None = None,
    ) -> None:
        cleaned_bucket = bucket.removeprefix("gs://").strip("/")
        if not cleaned_bucket:
            msg = "GCS artifact store requires a bucket name."
            raise ValueError(msg)
        cleaned_prefix = prefix.strip("/")
        self.bucket_name = cleaned_bucket
        self.prefix = cleaned_prefix
        self.run_id = run_id
        self._client = storage_client or _storage_client()
        self._bucket = self._client.bucket(self.bucket_name)

    @property
    def run_root_uri(self) -> str:
        return f"gs://{self.bucket_name}/{self._run_prefix}"

    @property
    def _run_prefix(self) -> str:
        if self.prefix:
            return f"{self.prefix}/{self.run_id}"
        return self.run_id

    def raw_uri(self, source: str, filename: str) -> str:
        return self._uri(f"raw/{source.strip('/')}/{filename.strip('/')}")

    def table_uri(self, layer: str, name: str, suffix: str = "parquet") -> str:
        if layer not in {"bronze", "silver", "gold"}:
            msg = f"Unsupported medallion layer: {layer}"
            raise ValueError(msg)
        return self._uri(f"{layer}/{name}.{suffix}")

    def csv_uri(self, layer: str, name: str) -> str:
        if layer not in {"bronze", "silver", "gold"}:
            msg = f"Unsupported medallion layer: {layer}"
            raise ValueError(msg)
        return self._uri(f"{layer}/csv/{name}.csv")

    def write_text(self, uri: str, content: str) -> str:
        blob = self._blob_for_uri(uri)
        blob.upload_from_string(content, content_type="text/plain; charset=utf-8")
        return uri

    def write_json(self, uri: str, payload: Mapping[str, Any]) -> str:
        blob = self._blob_for_uri(uri)
        blob.upload_from_string(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            content_type="application/json",
        )
        return uri

    def write_parquet(self, uri: str, frame: pd.DataFrame, *, index: bool = False) -> str:
        buffer = BytesIO()
        frame.to_parquet(buffer, index=index)
        buffer.seek(0)
        blob = self._blob_for_uri(uri)
        blob.upload_from_file(buffer, content_type="application/octet-stream")
        return uri

    def write_csv(self, uri: str, frame: pd.DataFrame) -> str:
        blob = self._blob_for_uri(uri)
        blob.upload_from_string(frame.to_csv(index=False), content_type="text/csv")
        return uri

    def read_parquet(self, uri: str) -> pd.DataFrame:
        blob = self._blob_for_uri(uri)
        buffer = BytesIO(blob.download_as_bytes())
        return pd.read_parquet(buffer)

    def exists(self, uri: str) -> bool:
        return bool(self._blob_for_uri(uri).exists())

    def local_path(self, uri: str) -> Path | None:
        del uri
        return None

    def _uri(self, relative_path: str) -> str:
        return f"gs://{self.bucket_name}/{self._run_prefix}/{relative_path.strip('/')}"

    def _blob_for_uri(self, uri: str) -> Any:
        bucket_name, object_name = _parse_gcs_uri(uri)
        if bucket_name != self.bucket_name:
            msg = f"URI bucket {bucket_name!r} does not match store bucket {self.bucket_name!r}."
            raise ValueError(msg)
        return self._bucket.blob(object_name)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        msg = f"Expected a gs:// URI, got: {uri}"
        raise ValueError(msg)
    without_scheme = uri.removeprefix("gs://")
    bucket, _, object_name = without_scheme.partition("/")
    if not bucket or not object_name:
        msg = f"Invalid GCS URI: {uri}"
        raise ValueError(msg)
    return bucket, object_name


def _storage_client() -> Any:
    try:
        from google.cloud import storage
    except ImportError as exc:
        msg = (
            "GCS artifact writes require the optional gcp extra. "
            "Run `uv sync --extra gcp` or prefix the command with `uv run --extra gcp`."
        )
        raise RuntimeError(msg) from exc
    return storage.Client()
