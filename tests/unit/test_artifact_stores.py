from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_analysis.artifacts.local_store import LocalArtifactStore
from stock_analysis.gcp.gcs_store import GcsArtifactStore


def test_local_artifact_store_preserves_run_layout(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "data", "run-1")
    frame = pd.DataFrame({"ticker": ["SPY"], "weight": [1.0]})

    assert store.run_root == tmp_path / "data" / "runs" / "run-1"
    assert store.table_uri("gold", "portfolio_recommendations").endswith(
        "data/runs/run-1/gold/portfolio_recommendations.parquet"
    )

    parquet_uri = store.write_parquet(store.table_uri("gold", "portfolio_recommendations"), frame)
    csv_uri = store.write_csv(store.csv_uri("gold", "portfolio_recommendations"), frame)
    text_uri = store.write_text(store.raw_uri("prices", "metadata.json"), "{}")

    assert Path(parquet_uri).exists()
    assert Path(csv_uri).exists()
    assert Path(text_uri).exists()
    assert store.exists(parquet_uri)
    assert store.read_parquet(parquet_uri).equals(frame)
    assert store.local_path(parquet_uri) == Path(parquet_uri)


def test_gcs_artifact_store_builds_run_scoped_uris_and_writes_objects() -> None:
    client = FakeStorageClient()
    store = GcsArtifactStore(
        bucket="gs://stock-analysis-medallion-prod/",
        prefix="runs",
        run_id="run-1",
        storage_client=client,
    )
    frame = pd.DataFrame({"ticker": ["SPY"], "weight": [1.0]})

    assert store.run_root_uri == "gs://stock-analysis-medallion-prod/runs/run-1"
    assert (
        store.raw_uri("prices", "metadata.json")
        == "gs://stock-analysis-medallion-prod/runs/run-1/raw/prices/metadata.json"
    )
    assert (
        store.table_uri("gold", "portfolio_recommendations")
        == "gs://stock-analysis-medallion-prod/runs/run-1/gold/portfolio_recommendations.parquet"
    )
    assert (
        store.csv_uri("gold", "portfolio_recommendations")
        == "gs://stock-analysis-medallion-prod/runs/run-1/gold/csv/portfolio_recommendations.csv"
    )

    store.write_text(store.raw_uri("prices", "metadata.json"), "{}")
    store.write_json(store.raw_uri("sp500_constituents", "metadata.json"), {"run_id": "run-1"})
    parquet_uri = store.write_parquet(store.table_uri("gold", "portfolio_recommendations"), frame)
    store.write_csv(store.csv_uri("gold", "portfolio_recommendations"), frame)

    assert client.bucket_names == ["stock-analysis-medallion-prod"]
    assert store.exists(parquet_uri)
    assert store.read_parquet(parquet_uri).equals(frame)
    assert store.local_path(parquet_uri) is None


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}
        self.bucket_names: list[str] = []

    def bucket(self, name: str) -> FakeBucket:
        self.bucket_names.append(name)
        bucket = self.buckets.get(name)
        if bucket is None:
            bucket = FakeBucket(name)
            self.buckets[name] = bucket
        return bucket


class FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        blob = self.blobs.get(name)
        if blob is None:
            blob = FakeBlob(name)
            self.blobs[name] = blob
        return blob


class FakeBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.content = b""

    def upload_from_string(self, content: str, content_type: str | None = None) -> None:
        del content_type
        self.content = content.encode("utf-8")

    def upload_from_file(self, file_obj, content_type: str | None = None) -> None:
        del content_type
        self.content = file_obj.read()

    def download_as_bytes(self) -> bytes:
        return self.content

    def exists(self) -> bool:
        return bool(self.content)
