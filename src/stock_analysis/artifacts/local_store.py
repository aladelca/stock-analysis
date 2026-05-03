from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from stock_analysis.paths import ProjectPaths, ensure_parent


class LocalArtifactStore:
    """Medallion artifact store backed by the local filesystem."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self._paths = ProjectPaths(output_root, run_id)
        self.run_id = run_id

    @property
    def run_root_uri(self) -> str:
        return str(self._paths.run_root)

    @property
    def run_root(self) -> Path:
        return self._paths.run_root

    def raw_uri(self, source: str, filename: str) -> str:
        return str(self._paths.raw_dir(source) / filename)

    def table_uri(self, layer: str, name: str, suffix: str = "parquet") -> str:
        if layer == "bronze":
            return str(self._paths.bronze_path(name, suffix))
        if layer == "silver":
            return str(self._paths.silver_path(name, suffix))
        if layer == "gold":
            return str(self._paths.gold_path(name, suffix))
        msg = f"Unsupported medallion layer: {layer}"
        raise ValueError(msg)

    def csv_uri(self, layer: str, name: str) -> str:
        return str(self._paths.csv_mirror_path(layer, name))

    def write_text(self, uri: str, content: str) -> str:
        path = Path(uri)
        ensure_parent(path)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_json(self, uri: str, payload: Mapping[str, Any]) -> str:
        return self.write_text(uri, json.dumps(payload, indent=2, sort_keys=True, default=str))

    def write_parquet(self, uri: str, frame: pd.DataFrame, *, index: bool = False) -> str:
        path = Path(uri)
        ensure_parent(path)
        frame.to_parquet(path, index=index)
        return str(path)

    def write_csv(self, uri: str, frame: pd.DataFrame) -> str:
        path = Path(uri)
        ensure_parent(path)
        frame.to_csv(path, index=False)
        return str(path)

    def write_bytes(self, uri: str, content: bytes, *, content_type: str | None = None) -> str:
        del content_type
        path = Path(uri)
        ensure_parent(path)
        path.write_bytes(content)
        return str(path)

    def read_bytes(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

    def read_parquet(self, uri: str) -> pd.DataFrame:
        return pd.read_parquet(Path(uri))

    def exists(self, uri: str) -> bool:
        return Path(uri).exists()

    def local_path(self, uri: str) -> Path | None:
        return Path(uri)
