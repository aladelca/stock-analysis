from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stock_analysis.backtest.runner import BacktestConfig, run_walk_forward_backtest
from stock_analysis.config import OptimizerConfig
from stock_analysis.ml.evaluation import (
    EvaluationConfig,
    benchmark_relative_metrics,
    evaluate,
    portfolio_metrics,
)
from stock_analysis.ml.tracking import ExperimentTracker


def run_experiment_from_config(config_path: Path, *, force: bool = False) -> Path:
    """Run a filesystem-tracked experiment from a small YAML config."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment_id = str(config.get("experiment_id") or config_path.stem)
    seed = int(config.get("seed", 42))
    input_run_root = _resolve_input_run_root(config)
    if input_run_root is None and _requires_phase1_artifacts(config):
        msg = (
            "No Phase 1 run root found. Provide input_run_root or run the one-shot pipeline "
            "so data/runs/<run_id> contains the ML panel and labels artifacts."
        )
        raise FileNotFoundError(msg)
    tracker = ExperimentTracker(Path(config.get("tracking_root", "data/experiments")))
    run = tracker.start_run(experiment_id, config, seed=seed, force=force)

    predictions = _build_predictions(config, input_run_root=input_run_root)
    metrics: dict[str, Any] = {"status": "created"}
    if predictions is not None:
        targets = _load_targets(config, input_run_root=input_run_root)
        if targets is not None and not targets.empty:
            horizon = int(config.get("horizon_days", 5))
            target_col = str(config.get("target_column", f"fwd_return_{horizon}d"))
            metrics = evaluate(
                predictions.rename(columns={"forecast_score": "prediction"}),
                targets.rename(columns={target_col: "target"}),
                config=EvaluationConfig(
                    target_column="target",
                    bootstrap_samples=int(config.get("bootstrap_samples", 200)),
                    random_seed=seed,
                ),
            )

    backtest = _load_optional_frame(config, "backtest_path")
    if backtest is None and input_run_root is not None and config.get("run_backtest", True):
        backtest = _run_backtest(config, input_run_root)
        if backtest is not None and not backtest.empty:
            metrics["portfolio"] = portfolio_metrics(
                backtest.drop_duplicates("rebalance_date")["portfolio_net_return"]
            )
            benchmark = _load_benchmark_returns(input_run_root, int(config.get("horizon_days", 5)))
            if benchmark is not None and not benchmark.empty:
                portfolio_returns = backtest.drop_duplicates("rebalance_date")[
                    ["rebalance_date", "portfolio_net_return"]
                ].rename(
                    columns={
                        "rebalance_date": "date",
                        "portfolio_net_return": "portfolio_return",
                    }
                )
                metrics["benchmark_relative"] = benchmark_relative_metrics(
                    portfolio_returns,
                    benchmark,
                )

    feature_importance = _load_optional_frame(config, "feature_importance_path")
    tracker.write_run(
        run,
        predictions=predictions,
        backtest=backtest if backtest is not None else pd.DataFrame(),
        metrics=metrics,
        feature_importance=feature_importance,
    )
    return run.run_dir


class HeuristicForecastModel:
    def __init__(
        self,
        *,
        momentum_column: str = "momentum_252d",
        volatility_column: str = "volatility_63d",
        volatility_penalty: float = 0.25,
    ) -> None:
        self.momentum_column = momentum_column
        self.volatility_column = volatility_column
        self.volatility_penalty = volatility_penalty

    def predict(self, features: pd.DataFrame) -> list[float]:
        missing = [
            column
            for column in [self.momentum_column, self.volatility_column]
            if column not in features.columns
        ]
        if missing:
            msg = f"feature frame missing heuristic columns: {missing}"
            raise ValueError(msg)
        scores = pd.to_numeric(features[self.momentum_column], errors="coerce").fillna(
            0
        ) - self.volatility_penalty * pd.to_numeric(
            features[self.volatility_column], errors="coerce"
        ).fillna(0)
        return scores.astype(float).tolist()


def _build_predictions(
    config: dict[str, Any],
    *,
    input_run_root: Path | None,
) -> pd.DataFrame | None:
    explicit = _load_optional_frame(config, "predictions_path")
    if explicit is not None:
        return explicit

    if input_run_root is None:
        return None

    panel_path = input_run_root / "silver" / "asset_daily_features_panel.parquet"
    if not panel_path.exists():
        msg = f"asset feature panel not found: {panel_path}"
        raise FileNotFoundError(msg)
    panel = pd.read_parquet(panel_path)
    model = str(config.get("model", "heuristic"))
    if model != "heuristic":
        msg = f"unsupported experiment model for this runner: {model}"
        raise ValueError(msg)

    predictor = _heuristic_model_from_config(config)

    predictions = panel[["ticker", "date"]].copy()
    predictions["forecast_score"] = predictor.predict(panel)
    return predictions


def _load_targets(
    config: dict[str, Any],
    *,
    input_run_root: Path | None,
) -> pd.DataFrame | None:
    explicit = _load_optional_frame(config, "targets_path")
    if explicit is not None:
        return explicit
    if input_run_root is None:
        return None
    labels_path = input_run_root / "gold" / "labels_panel.parquet"
    if not labels_path.exists():
        return None
    return pd.read_parquet(labels_path)


def _run_backtest(config: dict[str, Any], input_run_root: Path) -> pd.DataFrame | None:
    panel_path = input_run_root / "silver" / "asset_daily_features_panel.parquet"
    labels_path = input_run_root / "gold" / "labels_panel.parquet"
    returns_path = input_run_root / "silver" / "asset_daily_returns.parquet"
    required = [panel_path, labels_path, returns_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        msg = f"cannot run backtest; missing required artifacts: {missing}"
        raise FileNotFoundError(msg)

    panel = pd.read_parquet(panel_path)
    labels = pd.read_parquet(labels_path)
    returns = pd.read_parquet(returns_path)
    model = _heuristic_model_from_config(config)
    backtest_config = _backtest_config_from_dict(config.get("backtest", {}))
    optimizer_config = OptimizerConfig.model_validate(config.get("optimizer", {}))

    return run_walk_forward_backtest(
        panel,
        labels,
        returns,
        lambda _train: model,
        optimizer_config,
        backtest_config,
    )


def _heuristic_model_from_config(config: dict[str, Any]) -> HeuristicForecastModel:
    return HeuristicForecastModel(
        momentum_column=str(config.get("momentum_column", "momentum_252d")),
        volatility_column=str(config.get("volatility_column", "volatility_63d")),
        volatility_penalty=float(config.get("volatility_penalty", 0.25)),
    )


def _backtest_config_from_dict(raw: dict[str, Any]) -> BacktestConfig:
    data = dict(raw)
    if "feature_columns" in data:
        data["feature_columns"] = tuple(data["feature_columns"])
    return BacktestConfig(**data)


def _load_benchmark_returns(input_run_root: Path, horizon_days: int) -> pd.DataFrame | None:
    path = input_run_root / "silver" / "benchmark_returns.parquet"
    if not path.exists():
        return None
    benchmark = pd.read_parquet(path)
    benchmark = benchmark.loc[benchmark["horizon_days"].astype(int) == horizon_days].copy()
    return benchmark.rename(columns={"spy_return": "benchmark_return"})


def _resolve_input_run_root(config: dict[str, Any]) -> Path | None:
    explicit = config.get("input_run_root")
    if explicit:
        return Path(explicit)

    data_root = Path(config.get("data_root", "data"))
    runs_root = data_root / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        [path for path in runs_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "silver" / "asset_daily_features_panel.parquet").exists() and (
            candidate / "gold" / "labels_panel.parquet"
        ).exists():
            return candidate
    return None


def _requires_phase1_artifacts(config: dict[str, Any]) -> bool:
    if config.get("predictions_path") is not None and not config.get("run_backtest", False):
        return False
    return str(config.get("model", "heuristic")) == "heuristic" or bool(
        config.get("run_backtest", True)
    )


def _load_optional_frame(config: dict[str, Any], key: str) -> pd.DataFrame | None:
    path_value = config.get(key)
    if path_value is None:
        return None
    path = Path(path_value)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    msg = f"unsupported tabular artifact format for {key}: {path}"
    raise ValueError(msg)
