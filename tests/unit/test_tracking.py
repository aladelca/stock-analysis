from __future__ import annotations

import json

import pandas as pd
import pytest

from stock_analysis.ml.tracking import ExperimentTracker


def test_experiment_tracker_writes_artifacts_and_index(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    run = tracker.start_run("e0_heuristic", {"model": "heuristic"}, seed=123)

    tracker.write_run(
        run,
        predictions=pd.DataFrame({"date": ["2026-01-01"], "ticker": ["AAA"], "prediction": [0.1]}),
        backtest=pd.DataFrame({"rebalance_date": ["2026-01-01"], "portfolio_net_return": [0.01]}),
        metrics={"predictive": {"pearson_ic": 0.1}},
    )

    assert (run.run_dir / "config.yaml").exists()
    assert (run.run_dir / "predictions.parquet").exists()
    assert json.loads((run.run_dir / "metrics.json").read_text(encoding="utf-8"))["predictive"][
        "pearson_ic"
    ] == pytest.approx(0.1)
    index = pd.read_parquet(tmp_path / "experiments" / "experiments_index.parquet")
    assert index.loc[0, "experiment_id"] == "e0_heuristic"


def test_experiment_tracker_rejects_existing_run_without_force(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    tracker.start_run("duplicate", {}, seed=1)

    with pytest.raises(FileExistsError):
        tracker.start_run("duplicate", {}, seed=1)
