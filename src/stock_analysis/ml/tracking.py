from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    run_dir: Path


class ExperimentTracker:
    def __init__(self, root: Path = Path("data/experiments")) -> None:
        self.root = root
        self.index_path = root / "experiments_index.parquet"

    def start_run(
        self,
        experiment_id: str,
        config: dict[str, Any],
        *,
        seed: int = 42,
        force: bool = False,
    ) -> ExperimentRun:
        run_dir = self.root / experiment_id
        if run_dir.exists() and not force:
            msg = f"experiment_id already exists: {experiment_id}; pass force=True to overwrite"
            raise FileExistsError(msg)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "code_hash.txt").write_text(_code_hash(), encoding="utf-8")
        (run_dir / "seed.txt").write_text(str(seed), encoding="utf-8")
        return ExperimentRun(experiment_id=experiment_id, run_dir=run_dir)

    def write_run(
        self,
        run: ExperimentRun,
        *,
        predictions: pd.DataFrame | None = None,
        backtest: pd.DataFrame | None = None,
        metrics: dict[str, Any] | None = None,
        feature_importance: pd.DataFrame | None = None,
    ) -> None:
        if predictions is not None:
            predictions.to_parquet(run.run_dir / "predictions.parquet", index=False)
        if backtest is not None:
            backtest.to_parquet(run.run_dir / "backtest.parquet", index=False)
        if metrics is not None:
            (run.run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if feature_importance is not None:
            feature_importance.to_parquet(run.run_dir / "feature_importance.parquet", index=False)
        self._append_index(run, metrics or {})

    def _append_index(self, run: ExperimentRun, metrics: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": run.experiment_id,
            "run_dir": str(run.run_dir),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "code_hash": (run.run_dir / "code_hash.txt").read_text(encoding="utf-8").strip(),
            "metrics_json": json.dumps(metrics, sort_keys=True),
        }
        existing = pd.read_parquet(self.index_path) if self.index_path.exists() else pd.DataFrame()
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated = updated.drop_duplicates(subset=["experiment_id"], keep="last")
        updated.to_parquet(self.index_path, index=False)


def _code_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
