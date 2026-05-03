from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class RunArtifactStore(Protocol):
    """Write and locate medallion artifacts for one pipeline run."""

    run_id: str

    @property
    def run_root_uri(self) -> str:
        """Return the root URI for this run."""

    def raw_uri(self, source: str, filename: str) -> str:
        """Return the URI for a raw artifact."""

    def table_uri(self, layer: str, name: str, suffix: str = "parquet") -> str:
        """Return the URI for a medallion table artifact."""

    def csv_uri(self, layer: str, name: str) -> str:
        """Return the URI for a CSV mirror artifact."""

    def write_text(self, uri: str, content: str) -> str:
        """Write UTF-8 text and return the written URI."""

    def write_json(self, uri: str, payload: Mapping[str, Any]) -> str:
        """Write JSON and return the written URI."""

    def write_parquet(self, uri: str, frame: pd.DataFrame, *, index: bool = False) -> str:
        """Write a DataFrame as Parquet and return the written URI."""

    def write_csv(self, uri: str, frame: pd.DataFrame) -> str:
        """Write a DataFrame as CSV and return the written URI."""

    def read_parquet(self, uri: str) -> pd.DataFrame:
        """Read a Parquet artifact."""

    def exists(self, uri: str) -> bool:
        """Return whether an artifact exists."""

    def local_path(self, uri: str) -> Path | None:
        """Return a local path for a URI when the store is local."""


def write_table_with_csv(
    store: RunArtifactStore,
    layer: str,
    name: str,
    frame: pd.DataFrame,
    *,
    parquet_index: bool = False,
) -> list[str]:
    """Write a medallion table and CSV mirror."""

    return [
        store.write_parquet(
            store.table_uri(layer, name),
            frame,
            index=parquet_index,
        ),
        store.write_csv(store.csv_uri(layer, name), frame),
    ]
