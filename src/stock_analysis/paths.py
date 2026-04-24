from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    output_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.output_root / "runs" / self.run_id

    def raw_dir(self, source: str) -> Path:
        return self.run_root / "raw" / source

    def bronze_path(self, name: str, suffix: str = "parquet") -> Path:
        return self.run_root / "bronze" / f"{name}.{suffix}"

    def silver_path(self, name: str, suffix: str = "parquet") -> Path:
        return self.run_root / "silver" / f"{name}.{suffix}"

    def gold_path(self, name: str, suffix: str = "parquet") -> Path:
        return self.run_root / "gold" / f"{name}.{suffix}"

    def csv_mirror_path(self, layer: str, name: str) -> Path:
        return self.run_root / layer / "csv" / f"{name}.csv"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
