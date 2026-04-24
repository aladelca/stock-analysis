from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stock_analysis.config import OptimizerConfig, PortfolioConfig
from stock_analysis.ml.mlflow_tracking import (
    log_autoresearch_result,
    log_portfolio_run,
    resolve_tracking_uri,
)


class _FakeRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="run-123")

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


class _FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.run_name = None
        self.tags = {}
        self.params = {}
        self.metrics = {}
        self.dict_artifacts = {}
        self.file_artifacts = []
        self.created_experiment = None
        self.artifact_location = None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def get_experiment_by_name(self, experiment_name: str) -> None:
        del experiment_name
        return None

    def create_experiment(self, experiment_name: str, *, artifact_location: str) -> str:
        self.created_experiment = experiment_name
        self.artifact_location = artifact_location
        return "1"

    def start_run(self, *, run_name: str) -> _FakeRun:
        self.run_name = run_name
        return _FakeRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_dict(self, value: dict[str, object], artifact_file: str) -> None:
        self.dict_artifacts[artifact_file] = value

    def log_artifact(self, path: str) -> None:
        self.file_artifacts.append(path)


def test_resolve_tracking_uri_converts_local_paths_to_file_uris(tmp_path: Path) -> None:
    assert resolve_tracking_uri(None) == "sqlite:///data/mlflow/mlflow.db"
    assert resolve_tracking_uri(str(tmp_path / "mlflow.db")).startswith("sqlite:////")
    assert resolve_tracking_uri(str(tmp_path)).startswith("file://")
    assert resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"


def test_log_autoresearch_result_logs_params_metrics_tags_and_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact = tmp_path / "result.tsv"
    artifact.write_text("candidate_id\tstatus\ncandidate\tgo\n", encoding="utf-8")

    run_id = log_autoresearch_result(
        _result(),
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="experiment",
        artifacts=[artifact, tmp_path / "missing.tsv"],
    )

    assert run_id == "run-123"
    assert fake_mlflow.tracking_uri.startswith("file://")
    assert fake_mlflow.experiment_name == "experiment"
    assert fake_mlflow.created_experiment is None
    assert fake_mlflow.run_name == "candidate-iter-1"
    assert fake_mlflow.tags["candidate_id"] == "candidate"
    assert fake_mlflow.tags["decision_status"] == "go"
    assert fake_mlflow.params["candidate.model_kind"] == "ridge"
    assert fake_mlflow.params["config.max_assets"] == 100
    assert fake_mlflow.metrics["comparison.candidate_sharpe"] == 3.0
    assert fake_mlflow.metrics["baseline.information_ratio"] == 2.2
    assert "result.json" in fake_mlflow.dict_artifacts
    assert fake_mlflow.file_artifacts == [str(artifact)]


def test_log_autoresearch_result_sets_local_artifact_location_for_sqlite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    log_autoresearch_result(
        _result(),
        tracking_uri=str(tmp_path / "tracking.db"),
        experiment_name="experiment",
    )

    assert fake_mlflow.tracking_uri.startswith("sqlite:////")
    assert fake_mlflow.created_experiment == "experiment"
    assert fake_mlflow.artifact_location.startswith("file://")


def test_log_portfolio_run_logs_trade_metrics_and_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact = tmp_path / "portfolio_recommendations.parquet"
    artifact.write_text("placeholder", encoding="utf-8")

    mlflow_run_id = log_portfolio_run(
        PortfolioConfig(optimizer=OptimizerConfig(commission_rate=0.02)),
        run_id="run-1",
        data_as_of_date="2026-04-23",
        recommendations=pd.DataFrame(
            {
                "action": ["BUY", "SELL", "HOLD", "EXCLUDE"],
                "target_weight": [0.2, 0.0, 0.3, 0.0],
                "current_weight": [0.0, 0.1, 0.301, 0.0],
                "trade_abs_weight": [0.2, 0.1, 0.001, 0.0],
                "estimated_commission_weight": [0.004, 0.002, 0.0, 0.0],
                "cash_required_weight": [0.204, 0.0, 0.0, 0.0],
                "cash_released_weight": [0.0, 0.098, 0.0, 0.0],
            }
        ),
        risk_metrics=pd.DataFrame(
            {
                "metric": ["expected_return", "expected_volatility"],
                "value": [0.12, 0.2],
            }
        ),
        run_metadata=pd.DataFrame(
            {
                "config_hash": ["abc123"],
                "universe_count": [500],
                "price_row_count": [1000],
                "expected_return_is_calibrated": [False],
            }
        ),
        artifacts=[artifact, tmp_path / "missing.parquet"],
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="portfolio",
    )

    assert mlflow_run_id == "run-123"
    assert fake_mlflow.run_name == "portfolio-run-1"
    assert fake_mlflow.tags["stock_analysis.workflow"] == "one_shot_portfolio"
    assert fake_mlflow.params["optimizer.commission_rate"] == 0.02
    assert fake_mlflow.metrics["recommendations.num_buy"] == 1.0
    assert fake_mlflow.metrics["recommendations.num_sell"] == 1.0
    assert fake_mlflow.metrics["recommendations.estimated_commission_weight"] == 0.006
    assert fake_mlflow.metrics["portfolio.expected_return"] == 0.12
    assert "portfolio_run.json" in fake_mlflow.dict_artifacts
    assert fake_mlflow.file_artifacts == [str(artifact)]


def _result() -> dict[str, object]:
    return {
        "iteration_id": "iter-1",
        "git_commit": "abc123",
        "candidate": {
            "candidate_id": "candidate",
            "description": "Candidate",
            "feature_columns": ["momentum_21d", "volatility_21d"],
            "model_kind": "ridge",
            "training_target_column": None,
        },
        "config": {
            "input_run_root": "data/runs/source",
            "max_assets": 100,
            "max_rebalances": 48,
        },
        "metrics": {
            "comparison": {
                "candidate_sharpe": 3.0,
                "spy_sharpe": 1.0,
                "information_ratio": 2.5,
            },
            "portfolio": {"annualized_return": 1.1},
            "spy": {"annualized_return": 0.2},
            "benchmark_relative": {"active_return": 0.5},
        },
        "baseline": {
            "candidate_id": "phase2_e8_final",
            "information_ratio": 2.2,
            "candidate_sharpe": 2.9,
        },
        "decision": {
            "status": "go",
            "passed_spy_gate": True,
            "objective_improved": True,
        },
    }
