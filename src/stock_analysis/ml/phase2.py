from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_analysis.backtest.runner import BacktestConfig, run_walk_forward_backtest
from stock_analysis.config import OptimizerConfig
from stock_analysis.ml.evaluation import (
    EvaluationConfig,
    benchmark_relative_metrics,
    evaluate,
    portfolio_metrics,
)
from stock_analysis.ml.experiments import HeuristicForecastModel
from stock_analysis.ml.tracking import ExperimentTracker

SURVIVORSHIP_BANNER = "Uses current S&P 500 constituents; survivorship bias present."

DEFAULT_FEATURE_CANDIDATES = (
    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "momentum_252d",
    "momentum_21d_rank",
    "momentum_63d_rank",
    "momentum_126d_rank",
    "momentum_252d_rank",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
    "max_drawdown_63d",
    "max_drawdown_252d",
    "ma_ratio_50d",
    "ma_ratio_200d",
    "return_5d",
    "return_21d",
    "dollar_volume_21d",
    "volume_21d_zscore",
    "return_21d_excess",
)


@dataclass(frozen=True)
class Phase2Config:
    input_run_root: Path
    output_dir: Path = Path("docs/experiments")
    tracking_root: Path = Path("data/experiments")
    experiments: tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4", "E8")
    horizon_days: int = 5
    bootstrap_samples: int = 500
    force: bool = False
    feature_columns: tuple[str, ...] = ()
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


class RidgeForecastModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_column: str,
        alpha: float = 1.0,
        rank_normalize_features: bool = False,
    ) -> None:
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.alpha = alpha
        self.rank_normalize_features = rank_normalize_features
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.coef_: np.ndarray | None = None
        self._fit(train_df)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            msg = "model has not been fit"
            raise ValueError(msg)
        x = self._transform_features(features, fit=False)
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        return design @ self.coef_

    def _fit(self, train_df: pd.DataFrame) -> None:
        target = pd.to_numeric(train_df[self.target_column], errors="coerce")
        valid = target.notna()
        if not valid.any():
            msg = f"no non-null training target values for {self.target_column}"
            raise ValueError(msg)
        x = self._transform_features(train_df.loc[valid], fit=True)
        y = target.loc[valid].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y

    def _transform_features(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            msg = f"missing model feature columns: {missing}"
            raise ValueError(msg)
        x = frame.loc[:, list(self.feature_columns)].apply(pd.to_numeric, errors="coerce")
        if self.rank_normalize_features:
            x = x.rank(method="average", pct=True).fillna(0.5)

        if fit:
            self.medians = x.median().fillna(0)
            filled = x.fillna(self.medians)
            self.means = filled.mean()
            self.stds = filled.std(ddof=0).replace(0, 1).fillna(1)
        if self.medians is None or self.means is None or self.stds is None:
            msg = "preprocessing statistics are unavailable"
            raise ValueError(msg)
        filled = x.fillna(self.medians)
        return (filled - self.means) / self.stds


class BlendedForecastModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        return_target_column: str,
        rank_target_column: str,
    ) -> None:
        self.return_model = RidgeForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=return_target_column,
        )
        self.rank_model = RidgeForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=rank_target_column,
            rank_normalize_features=True,
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return _zscore(self.return_model.predict(features)) + _zscore(
            self.rank_model.predict(features)
        )


def run_phase2(config: Phase2Config) -> pd.DataFrame:
    """Run the first Phase 2 experiment batch and write reports/artifacts."""

    artifacts = _load_phase1_artifacts(config.input_run_root)
    feature_columns = config.feature_columns or _default_feature_columns(artifacts["panel"])
    if not feature_columns:
        msg = "no Phase 2 feature columns are available in the panel"
        raise ValueError(msg)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for experiment_id in config.experiments:
        result = _run_single_experiment(
            experiment_id,
            artifacts,
            feature_columns,
            config,
        )
        results.append(result)
        _write_experiment_report(result, config.output_dir / f"{experiment_id.lower()}.md")

    summary = pd.DataFrame(results)
    _write_phase2_report(summary, config.output_dir / "phase2-report.md")
    return summary


def _run_single_experiment(
    experiment_id: str,
    artifacts: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    config: Phase2Config,
) -> dict[str, Any]:
    if experiment_id == "E0":
        heuristic_columns = _existing_columns(
            artifacts["panel"],
            ("momentum_252d", "volatility_63d"),
        )
        return _run_model_experiment(
            experiment_id,
            "Current heuristic",
            lambda train: HeuristicForecastModel(),
            artifacts,
            heuristic_columns,
            config,
        )
    if experiment_id == "E1":
        return _run_spy_benchmark(experiment_id, artifacts, config)
    if experiment_id == "E2":
        return _run_equal_weight_benchmark(experiment_id, artifacts, config)
    if experiment_id == "E3":
        return _run_model_experiment(
            experiment_id,
            "Ridge regression",
            lambda train: RidgeForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=f"fwd_return_{config.horizon_days}d",
            ),
            artifacts,
            feature_columns,
            config,
        )
    if experiment_id == "E4":
        return _run_model_experiment(
            experiment_id,
            "Ridge on rank-normalized features",
            lambda train: RidgeForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=f"fwd_rank_{config.horizon_days}d",
                rank_normalize_features=True,
            ),
            artifacts,
            feature_columns,
            config,
            training_target_column=f"fwd_rank_{config.horizon_days}d",
        )
    if experiment_id == "E8":
        return _run_model_experiment(
            experiment_id,
            "Linear blend of ridge return + ridge rank",
            lambda train: BlendedForecastModel(
                train,
                feature_columns=feature_columns,
                return_target_column=f"fwd_return_{config.horizon_days}d",
                rank_target_column=f"fwd_rank_{config.horizon_days}d",
            ),
            artifacts,
            feature_columns,
            config,
            training_target_column=f"fwd_rank_{config.horizon_days}d",
        )
    if experiment_id in {"E5", "E6", "E7"}:
        return _skipped_result(
            experiment_id,
            "LightGBM experiments are registered but require the LightGBM dependency "
            "and nested-CV tuning grid.",
        )
    return _skipped_result(experiment_id, f"Unknown Phase 2 experiment id: {experiment_id}")


def _run_model_experiment(
    experiment_id: str,
    model_name: str,
    predict_fn,
    artifacts: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    config: Phase2Config,
    *,
    training_target_column: str | None = None,
) -> dict[str, Any]:
    backtest_config = BacktestConfig(
        horizon_days=config.horizon_days,
        training_target_column=training_target_column,
        embargo_days=config.backtest.embargo_days,
        cost_bps=config.backtest.cost_bps,
        covariance_lookback_days=config.backtest.covariance_lookback_days,
        feature_columns=feature_columns,
        max_rebalances=config.backtest.max_rebalances,
    )
    backtest = run_walk_forward_backtest(
        artifacts["panel"],
        artifacts["labels"],
        artifacts["returns"],
        predict_fn,
        config.optimizer,
        backtest_config,
    )
    metrics = _metrics_from_backtest(backtest, artifacts["benchmark"], config)
    predictions = _predictions_from_backtest(backtest)
    _track_phase2_run(experiment_id, model_name, predictions, backtest, metrics, config)
    return _summary_row(experiment_id, model_name, metrics, "completed")


def _run_spy_benchmark(
    experiment_id: str,
    artifacts: dict[str, pd.DataFrame],
    config: Phase2Config,
) -> dict[str, Any]:
    benchmark = _benchmark_for_horizon(artifacts["benchmark"], config.horizon_days)
    backtest = benchmark.rename(
        columns={"date": "rebalance_date", "spy_return": "portfolio_net_return"}
    )
    backtest["portfolio_gross_return"] = backtest["portfolio_net_return"]
    backtest["turnover"] = 0.0
    metrics = {"portfolio": portfolio_metrics(backtest["portfolio_net_return"])}
    _track_phase2_run(
        experiment_id,
        "SPY buy-and-hold",
        pd.DataFrame(),
        backtest,
        metrics,
        config,
    )
    return _summary_row(experiment_id, "SPY buy-and-hold", metrics, "completed")


def _run_equal_weight_benchmark(
    experiment_id: str,
    artifacts: dict[str, pd.DataFrame],
    config: Phase2Config,
) -> dict[str, Any]:
    target_col = f"fwd_return_{config.horizon_days}d"
    backtest = (
        artifacts["labels"]
        .dropna(subset=[target_col])
        .groupby("date", as_index=False)
        .agg(portfolio_net_return=(target_col, "mean"))
        .rename(columns={"date": "rebalance_date"})
    )
    backtest["portfolio_gross_return"] = backtest["portfolio_net_return"]
    backtest["turnover"] = np.nan
    metrics = {"portfolio": portfolio_metrics(backtest["portfolio_net_return"])}
    benchmark = _benchmark_for_horizon(artifacts["benchmark"], config.horizon_days)
    if not benchmark.empty:
        portfolio_returns = backtest.rename(
            columns={"rebalance_date": "date", "portfolio_net_return": "portfolio_return"}
        )
        metrics["benchmark_relative"] = benchmark_relative_metrics(
            portfolio_returns[["date", "portfolio_return"]],
            benchmark.rename(columns={"spy_return": "benchmark_return"}),
        )
    _track_phase2_run(
        experiment_id,
        "Equal-weight S&P 500",
        pd.DataFrame(),
        backtest,
        metrics,
        config,
    )
    return _summary_row(experiment_id, "Equal-weight S&P 500", metrics, "completed")


def _metrics_from_backtest(
    backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: Phase2Config,
) -> dict[str, Any]:
    if backtest.empty:
        return {"predictive": {}, "portfolio": {}, "benchmark_relative": {}}
    predictions = _predictions_from_backtest(backtest)
    metrics = evaluate(
        predictions,
        config=EvaluationConfig(
            prediction_column="prediction",
            target_column="target",
            bootstrap_samples=config.bootstrap_samples,
        ),
    )
    portfolio_series = backtest.drop_duplicates("rebalance_date")["portfolio_net_return"]
    metrics["portfolio"] = portfolio_metrics(portfolio_series)
    benchmark_h = _benchmark_for_horizon(benchmark, config.horizon_days)
    if not benchmark_h.empty:
        portfolio_returns = backtest.drop_duplicates("rebalance_date")[
            ["rebalance_date", "portfolio_net_return"]
        ].rename(columns={"rebalance_date": "date", "portfolio_net_return": "portfolio_return"})
        metrics["benchmark_relative"] = benchmark_relative_metrics(
            portfolio_returns,
            benchmark_h.rename(columns={"spy_return": "benchmark_return"}),
        )
    return metrics


def _predictions_from_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame(columns=["date", "ticker", "prediction", "target"])
    return backtest[["rebalance_date", "ticker", "forecast_score", "realized_return"]].rename(
        columns={
            "rebalance_date": "date",
            "forecast_score": "prediction",
            "realized_return": "target",
        }
    )


def _track_phase2_run(
    experiment_id: str,
    model_name: str,
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    metrics: dict[str, Any],
    config: Phase2Config,
) -> None:
    tracker = ExperimentTracker(config.tracking_root)
    run = tracker.start_run(
        experiment_id.lower(),
        {
            "phase": 2,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "input_run_root": str(config.input_run_root),
            "horizon_days": config.horizon_days,
        },
        force=config.force,
    )
    tracker.write_run(run, predictions=predictions, backtest=backtest, metrics=metrics)


def _summary_row(
    experiment_id: str,
    model_name: str,
    metrics: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    predictive = metrics.get("predictive", {})
    portfolio = metrics.get("portfolio", {})
    benchmark = metrics.get("benchmark_relative", {})
    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "status": status,
        "pearson_ic": predictive.get("pearson_ic"),
        "rank_ic": predictive.get("rank_ic"),
        "sharpe": portfolio.get("sharpe"),
        "annualized_return": portfolio.get("annualized_return"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "information_ratio": benchmark.get("information_ratio"),
        "metrics": metrics,
    }


def _skipped_result(experiment_id: str, reason: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "model_name": experiment_id,
        "status": "skipped",
        "pearson_ic": None,
        "rank_ic": None,
        "sharpe": None,
        "annualized_return": None,
        "max_drawdown": None,
        "information_ratio": None,
        "metrics": {"skip_reason": reason},
    }


def _load_phase1_artifacts(run_root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "panel": run_root / "silver" / "asset_daily_features_panel.parquet",
        "labels": run_root / "gold" / "labels_panel.parquet",
        "returns": run_root / "silver" / "asset_daily_returns.parquet",
        "benchmark": run_root / "silver" / "benchmark_returns.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        msg = f"missing Phase 1 artifacts: {missing}"
        raise FileNotFoundError(msg)
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def _default_feature_columns(panel: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in DEFAULT_FEATURE_CANDIDATES if column in panel.columns)


def _existing_columns(panel: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        msg = f"panel is missing required feature columns: {missing}"
        raise ValueError(msg)
    return columns


def _benchmark_for_horizon(benchmark: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    if benchmark.empty:
        return benchmark.copy()
    result = benchmark.loc[benchmark["horizon_days"].astype(int) == horizon_days].copy()
    return result.dropna(subset=["spy_return"])


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0 or not np.isfinite(std):
        return np.zeros(len(values))
    return (values - float(np.mean(values))) / std


def _write_experiment_report(result: dict[str, Any], path: Path) -> None:
    metrics = result.get("metrics", {})
    path.write_text(
        "\n".join(
            [
                f"# {result['experiment_id']} - {result['model_name']}",
                "",
                f"**Status:** {result['status']}",
                "",
                f"> {SURVIVORSHIP_BANNER}",
                "",
                "## Metrics",
                "",
                "```json",
                json.dumps(metrics, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_phase2_report(summary: pd.DataFrame, path: Path) -> None:
    display = summary.drop(columns=["metrics"], errors="ignore")
    gating = _gating_decision(summary)
    path.write_text(
        "\n".join(
            [
                "# Phase 2 Experiment Report",
                "",
                f"> {SURVIVORSHIP_BANNER}",
                "",
                "## Results",
                "",
                _markdown_table(display),
                "",
                "## Gating Decision",
                "",
                gating,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _gating_decision(summary: pd.DataFrame) -> str:
    completed = summary.loc[summary["status"] == "completed"].copy()
    if completed.empty or "E0" not in set(completed["experiment_id"]):
        return "NO-GO: Phase 2 has not produced a completed heuristic baseline yet."
    e0 = completed.loc[completed["experiment_id"] == "E0"].iloc[0]
    contenders = completed.loc[
        completed["experiment_id"].isin(["E3", "E4", "E5", "E6", "E7", "E8"])
    ]
    winners = contenders.loc[
        (contenders["pearson_ic"].fillna(-np.inf) > (e0["pearson_ic"] or -np.inf))
        & (contenders["information_ratio"].fillna(-np.inf) > 0)
    ]
    if winners.empty:
        return (
            "NO-GO: no Phase 2 ML model currently beats E0 on OOS IC and SPY-relative IR. "
            "Do not start Phase 3."
        )
    best = winners.sort_values(["sharpe", "information_ratio"], ascending=False).iloc[0]
    return (
        f"PROVISIONAL GO: {best['experiment_id']} beats E0 on IC and has positive IR vs SPY. "
        "Confirm with full LightGBM nested-CV sweeps and Sharpe-difference bootstrap "
        "before Phase 3."
    )
