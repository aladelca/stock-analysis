from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_analysis.config import PortfolioConfig

DEFAULT_MLFLOW_EXPERIMENT_NAME = "stock-analysis-autoresearch"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///data/mlflow/mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_DIR = Path("data/mlflow/artifacts")


def log_autoresearch_result(
    result: dict[str, Any],
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
    artifacts: Iterable[Path] = (),
) -> str:
    """Log an autoresearch evaluator result to MLflow and return the run id."""

    mlflow = _import_mlflow()
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_tracking_uri)
    _set_experiment(mlflow, experiment_name, resolved_tracking_uri)

    candidate = result.get("candidate", {})
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    iteration_id = str(result.get("iteration_id", "unknown"))
    run_name = f"{candidate_id}-{iteration_id}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(_tags_for_result(result))
        params = _params_for_result(result)
        if params:
            mlflow.log_params(params)
        metrics = _metrics_for_result(result)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_dict(result, "result.json")
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
        return str(run.info.run_id)


def log_portfolio_run(
    config: PortfolioConfig,
    *,
    run_id: str,
    data_as_of_date: str,
    recommendations: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    run_metadata: pd.DataFrame,
    artifacts: Iterable[Path] = (),
    tracking_uri: str | None = None,
    experiment_name: str = "stock-analysis-portfolio",
) -> str:
    """Log a production portfolio run to MLflow and return the MLflow run id."""

    mlflow = _import_mlflow()
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_tracking_uri)
    _set_experiment(mlflow, experiment_name, resolved_tracking_uri)

    with mlflow.start_run(run_name=f"portfolio-{run_id}") as run:
        mlflow.set_tags(
            {
                "stock_analysis.workflow": "one_shot_portfolio",
                "run_id": run_id,
                "data_as_of_date": data_as_of_date,
                "forecast_engine": config.forecast.engine,
                "model_version": (
                    config.forecast.ml_model_version
                    if config.forecast.engine == "ml"
                    else "heuristic"
                ),
            }
        )
        params = _params_for_portfolio_run(config, run_metadata)
        if params:
            mlflow.log_params(params)
        metrics = _metrics_for_portfolio_run(recommendations, risk_metrics)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "run_id": run_id,
                "data_as_of_date": data_as_of_date,
                "optimizer": config.optimizer.model_dump(mode="json"),
                "forecast": config.forecast.model_dump(mode="json"),
                "tableau": config.tableau.model_dump(mode="json"),
                "metrics": metrics,
            },
            "portfolio_run.json",
        )
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
        return str(run.info.run_id)


def resolve_tracking_uri(tracking_uri: str | None) -> str:
    if not tracking_uri:
        return _ensure_sqlite_parent(DEFAULT_MLFLOW_TRACKING_URI)
    if tracking_uri.startswith("sqlite:///"):
        return _ensure_sqlite_parent(tracking_uri)
    if "://" in tracking_uri or tracking_uri.startswith("databricks"):
        return tracking_uri
    path = Path(tracking_uri)
    if path.suffix in {".db", ".sqlite", ".sqlite3"}:
        return _ensure_sqlite_parent(f"sqlite:///{path.resolve()}")
    return path.resolve().as_uri()


def _ensure_sqlite_parent(tracking_uri: str) -> str:
    db_path = tracking_uri.removeprefix("sqlite:///")
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return tracking_uri


def _set_experiment(mlflow: Any, experiment_name: str, tracking_uri: str) -> None:
    if tracking_uri.startswith("sqlite:///"):
        existing = mlflow.get_experiment_by_name(experiment_name)
        if existing is None:
            artifact_dir = DEFAULT_MLFLOW_ARTIFACT_DIR.resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_dir.as_uri(),
            )
    mlflow.set_experiment(experiment_name)


def _import_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ImportError as exc:
        msg = (
            "MLflow tracking requires the optional mlflow extra. "
            "Run `uv sync --extra mlflow` or prefix the command with `uv run --extra mlflow`."
        )
        raise RuntimeError(msg) from exc


def _tags_for_result(result: dict[str, Any]) -> dict[str, str]:
    candidate = result.get("candidate", {})
    decision = result.get("decision", {})
    return {
        "stock_analysis.workflow": "autoresearch",
        "candidate_id": str(candidate.get("candidate_id", "")),
        "decision_status": str(decision.get("status", "")),
        "passed_spy_gate": str(decision.get("passed_spy_gate", "")),
        "objective_improved": str(decision.get("objective_improved", "")),
        "git_commit": str(result.get("git_commit", "")),
    }


def _params_for_result(result: dict[str, Any]) -> dict[str, str | int | float | bool]:
    raw: dict[str, Any] = {
        "candidate": result.get("candidate", {}),
        "config": result.get("config", {}),
        "baseline": {"candidate_id": result.get("baseline", {}).get("candidate_id")},
    }
    flattened = _flatten(raw)
    params: dict[str, str | int | float | bool] = {}
    for key, value in flattened.items():
        if value is None:
            continue
        params[key] = _param_value(value)
    return params


def _metrics_for_result(result: dict[str, Any]) -> dict[str, float]:
    raw: dict[str, Any] = {
        "portfolio": result.get("metrics", {}).get("portfolio", {}),
        "spy": result.get("metrics", {}).get("spy", {}),
        "benchmark_relative": result.get("metrics", {}).get("benchmark_relative", {}),
        "comparison": result.get("metrics", {}).get("comparison", {}),
        "cashflow": result.get("metrics", {}).get("cashflow", {}),
        "baseline": result.get("baseline", {}),
    }
    metrics: dict[str, float] = {}
    for key, value in _flatten(raw).items():
        number = _finite_float(value)
        if number is not None:
            metrics[key] = number
    return metrics


def _params_for_portfolio_run(
    config: PortfolioConfig,
    run_metadata: pd.DataFrame,
) -> dict[str, str | int | float | bool]:
    metadata = run_metadata.iloc[0].to_dict() if not run_metadata.empty else {}
    raw: dict[str, Any] = {
        "optimizer": config.optimizer.model_dump(mode="json"),
        "forecast": config.forecast.model_dump(mode="json"),
        "prices": config.prices.model_dump(mode="json"),
        "portfolio_state": {
            "has_current_holdings": config.portfolio_state.current_holdings_path is not None,
            "current_holdings_path": (
                str(config.portfolio_state.current_holdings_path)
                if config.portfolio_state.current_holdings_path is not None
                else None
            ),
            "portfolio_value": config.portfolio_state.portfolio_value,
        },
        "contributions": config.contributions.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "run_metadata": {
            "config_hash": metadata.get("config_hash"),
            "universe_count": metadata.get("universe_count"),
            "price_row_count": metadata.get("price_row_count"),
            "expected_return_is_calibrated": metadata.get("expected_return_is_calibrated"),
        },
    }
    params: dict[str, str | int | float | bool] = {}
    for key, value in _flatten(raw).items():
        if value is None:
            continue
        params[key] = _param_value(value)
    return params


def _metrics_for_portfolio_run(
    recommendations: pd.DataFrame,
    risk_metrics: pd.DataFrame,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not risk_metrics.empty:
        for _, row in risk_metrics.iterrows():
            metric_name = str(row["metric"])
            value = _finite_float(row["value"])
            if value is not None:
                metrics[f"portfolio.{_metric_key(metric_name)}"] = value

    if not recommendations.empty:
        actions = recommendations["action"].astype(str).str.lower().value_counts()
        for action in ("buy", "sell", "hold", "exclude"):
            metrics[f"recommendations.num_{action}"] = float(actions.get(action, 0))
        planned = recommendations["action"].astype(str).isin(["BUY", "SELL"])
        metrics["recommendations.target_weight_sum"] = _sum_metric(
            recommendations,
            "target_weight",
        )
        metrics["recommendations.current_weight_sum"] = _sum_metric(
            recommendations,
            "current_weight",
        )
        metrics["recommendations.total_trade_abs_weight"] = _sum_metric(
            recommendations.loc[planned],
            "trade_abs_weight",
        )
        metrics["recommendations.estimated_commission_weight"] = _sum_metric(
            recommendations.loc[planned],
            "estimated_commission_weight",
        )
        metrics["recommendations.cash_required_weight"] = _sum_metric(
            recommendations.loc[planned],
            "cash_required_weight",
        )
        metrics["recommendations.cash_released_weight"] = _sum_metric(
            recommendations.loc[planned],
            "cash_released_weight",
        )
        metrics["recommendations.trade_notional"] = _sum_metric(
            recommendations.loc[planned],
            "trade_notional",
        )
        metrics["recommendations.commission_amount"] = _sum_metric(
            recommendations.loc[planned],
            "commission_amount",
        )
        metrics["recommendations.deposit_used_amount"] = _sum_metric(
            recommendations.loc[planned],
            "deposit_used_amount",
        )
    return {key: value for key, value in metrics.items() if np.isfinite(value)}


def _sum_metric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            normalized_key = _metric_key(str(key))
            nested_prefix = f"{prefix}.{normalized_key}" if prefix else normalized_key
            result.update(_flatten(nested, nested_prefix))
        return result
    return {prefix: value}


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, sort_keys=True)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _metric_key(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")
