from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

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

LIGHTGBM_PARAM_GRID: tuple[dict[str, Any], ...] = (
    {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 120,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    },
)

CATBOOST_PARAM_GRID: tuple[dict[str, Any], ...] = (
    {
        "iterations": 120,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 3.0,
    },
    {
        "iterations": 180,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 5.0,
    },
)


class ForecastModel(Protocol):
    def predict(self, features: pd.DataFrame) -> Sequence[float]:
        """Return a forecast score for each feature row."""


ModelFactory = Callable[[pd.DataFrame], ForecastModel]


@dataclass(frozen=True)
class Phase2Config:
    input_run_root: Path
    output_dir: Path = Path("docs/experiments")
    tracking_root: Path = Path("data/experiments")
    experiments: tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")
    horizon_days: int = 5
    bootstrap_samples: int = 500
    force: bool = False
    feature_columns: tuple[str, ...] = ()
    max_assets: int | None = None
    liquidity_column: str = "dollar_volume_21d"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    run_sweeps: bool = True
    sweep_horizons: tuple[int, ...] = (5, 21, 63)
    sweep_cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    sweep_turnover_penalties: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005)
    sweep_max_weights: tuple[float, ...] = (0.02, 0.05, 0.10)
    sweep_cadences: tuple[tuple[str, int], ...] = (
        ("weekly", 5),
        ("monthly", 21),
        ("quarterly", 63),
    )
    random_seed: int = 42
    lightgbm_nested_cv: bool = True
    lightgbm_inner_folds: int = 2


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_name: str
    model_factory: ModelFactory | None
    feature_columns: tuple[str, ...]
    training_target_column: str | None = None
    benchmark_kind: Literal["model", "spy", "equal_weight"] = "model"


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

    def predict(self, features: pd.DataFrame) -> list[float]:
        if self.coef_ is None:
            msg = "model has not been fit"
            raise ValueError(msg)
        x = self._transform_features(features, fit=False)
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        return (design @ self.coef_).astype(float).tolist()

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


class LightGBMForecastModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_column: str,
        task: Literal["regression", "rank", "classification"],
        score_column: str,
        random_seed: int,
        nested_cv: bool,
        inner_folds: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.task = task
        self.score_column = score_column
        self.random_seed = random_seed
        self.medians: pd.Series | None = None
        self.constant_prediction: float | None = None
        self.model: Any | None = None
        self.params = (
            self._select_params(train_df, inner_folds) if nested_cv else LIGHTGBM_PARAM_GRID[0]
        )
        self._fit(train_df, self.params)

    def predict(self, features: pd.DataFrame) -> list[float]:
        x = self._transform_features(features, fit=False)
        if self.constant_prediction is not None:
            return [self.constant_prediction] * len(x)
        if self.model is None:
            msg = "LightGBM model has not been fit"
            raise ValueError(msg)
        if self.task == "classification":
            probabilities = self.model.predict_proba(x)[:, 1]
            return probabilities.astype(float).tolist()
        return np.asarray(self.model.predict(x), dtype=float).tolist()

    def _fit(self, train_df: pd.DataFrame, params: dict[str, Any]) -> None:
        frame = train_df.copy()
        target = self._target(frame)
        valid = target.notna()
        if not valid.any():
            self.constant_prediction = 0.0
            return
        frame = frame.loc[valid].copy()
        target = target.loc[valid]
        frame["_target"] = target.to_numpy()
        frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
        target = frame.pop("_target")
        x = self._transform_features(frame, fit=True)
        y = self._coerce_target(frame, target)
        if self.task == "classification" and len(set(y.tolist())) < 2:
            self.constant_prediction = float(np.mean(y))
            return
        self.model = self._make_estimator(params)
        if self.task == "rank":
            group = _group_sizes(frame)
            self.model.fit(x, y, group=group)
        else:
            self.model.fit(x, y)

    def _select_params(self, train_df: pd.DataFrame, inner_folds: int) -> dict[str, Any]:
        dates = pd.DatetimeIndex(pd.to_datetime(train_df["date"]).dropna().unique()).sort_values()
        if len(dates) < 80:
            return LIGHTGBM_PARAM_GRID[0]
        folds = _inner_cv_date_folds(dates, max_folds=inner_folds)
        if not folds:
            return LIGHTGBM_PARAM_GRID[0]

        scored: list[tuple[float, dict[str, Any]]] = []
        for params in LIGHTGBM_PARAM_GRID:
            fold_scores: list[float] = []
            for train_dates, val_dates in folds:
                inner_train = train_df.loc[pd.to_datetime(train_df["date"]).isin(train_dates)]
                inner_val = train_df.loc[pd.to_datetime(train_df["date"]).isin(val_dates)]
                if inner_train.empty or inner_val.empty:
                    continue
                candidate = LightGBMForecastModel(
                    inner_train,
                    feature_columns=self.feature_columns,
                    target_column=self.target_column,
                    task=self.task,
                    score_column=self.score_column,
                    random_seed=self.random_seed,
                    nested_cv=False,
                    inner_folds=0,
                )
                candidate.params = params
                candidate._fit(inner_train, params)
                score = _rank_ic(
                    candidate.predict(inner_val),
                    pd.to_numeric(inner_val[self.score_column], errors="coerce"),
                )
                if np.isfinite(score):
                    fold_scores.append(score)
            if fold_scores:
                scored.append((float(np.mean(fold_scores)), params))
        if not scored:
            return LIGHTGBM_PARAM_GRID[0]
        return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]

    def _transform_features(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            msg = f"missing LightGBM feature columns: {missing}"
            raise ValueError(msg)
        x = frame.loc[:, list(self.feature_columns)].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.medians = x.median().fillna(0)
        if self.medians is None:
            msg = "LightGBM preprocessing medians are unavailable"
            raise ValueError(msg)
        return x.fillna(self.medians)

    def _target(self, frame: pd.DataFrame) -> pd.Series:
        if self.target_column not in frame.columns:
            msg = f"missing LightGBM target column: {self.target_column}"
            raise ValueError(msg)
        return pd.to_numeric(frame[self.target_column], errors="coerce")

    def _coerce_target(self, frame: pd.DataFrame, target: pd.Series) -> np.ndarray:
        if self.task == "classification":
            return target.fillna(0).astype(int).to_numpy()
        if self.task == "rank":
            pct = target.groupby(pd.to_datetime(frame["date"])).rank(pct=True)
            return np.maximum(np.floor(pct.fillna(0.5).to_numpy(dtype=float) * 10), 0).astype(int)
        return target.to_numpy(dtype=float)

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        import lightgbm as lgb

        common = {
            **params,
            "random_state": self.random_seed,
            "verbose": -1,
            "force_col_wise": True,
        }
        if self.task == "classification":
            return lgb.LGBMClassifier(objective="binary", **common)
        if self.task == "rank":
            return lgb.LGBMRanker(objective="lambdarank", **common)
        return lgb.LGBMRegressor(objective="regression", **common)


class CatBoostForecastModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_column: str,
        task: Literal["regression", "classification"],
        score_column: str,
        random_seed: int,
        params: dict[str, Any] | None = None,
        nested_cv: bool = False,
        inner_folds: int = 2,
    ) -> None:
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.task = task
        self.score_column = score_column
        self.random_seed = random_seed
        self.medians: pd.Series | None = None
        self.constant_prediction: float | None = None
        self.model: Any | None = None
        self.params = (
            params
            if params is not None
            else (
                self._select_params(train_df, inner_folds) if nested_cv else CATBOOST_PARAM_GRID[0]
            )
        )
        self._fit(train_df, self.params)

    def predict(self, features: pd.DataFrame) -> list[float]:
        x = self._transform_features(features, fit=False)
        if self.constant_prediction is not None:
            return [self.constant_prediction] * len(x)
        if self.model is None:
            msg = "CatBoost model has not been fit"
            raise ValueError(msg)
        if self.task == "classification":
            probabilities = self.model.predict_proba(x)[:, 1]
            return np.asarray(probabilities, dtype=float).tolist()
        return np.asarray(self.model.predict(x), dtype=float).tolist()

    def _fit(self, train_df: pd.DataFrame, params: dict[str, Any]) -> None:
        frame = train_df.copy()
        target = self._target(frame)
        valid = target.notna()
        if not valid.any():
            self.constant_prediction = 0.0
            return
        frame = frame.loc[valid].copy()
        target = target.loc[valid]
        frame["_target"] = target.to_numpy()
        frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
        target = frame.pop("_target")
        x = self._transform_features(frame, fit=True)
        y = self._coerce_target(target)
        if self.task == "classification" and len(set(y.tolist())) < 2:
            self.constant_prediction = float(np.mean(y))
            return
        self.model = self._make_estimator(params)
        self.model.fit(x, y, verbose=False)

    def _select_params(self, train_df: pd.DataFrame, inner_folds: int) -> dict[str, Any]:
        dates = pd.DatetimeIndex(pd.to_datetime(train_df["date"]).dropna().unique()).sort_values()
        if len(dates) < 80:
            return CATBOOST_PARAM_GRID[0]
        folds = _inner_cv_date_folds(dates, max_folds=inner_folds)
        if not folds:
            return CATBOOST_PARAM_GRID[0]

        scored: list[tuple[float, dict[str, Any]]] = []
        for params in CATBOOST_PARAM_GRID:
            fold_scores: list[float] = []
            for train_dates, val_dates in folds:
                inner_train = train_df.loc[pd.to_datetime(train_df["date"]).isin(train_dates)]
                inner_val = train_df.loc[pd.to_datetime(train_df["date"]).isin(val_dates)]
                if inner_train.empty or inner_val.empty:
                    continue
                candidate = CatBoostForecastModel(
                    inner_train,
                    feature_columns=self.feature_columns,
                    target_column=self.target_column,
                    task=self.task,
                    score_column=self.score_column,
                    random_seed=self.random_seed,
                    params=params,
                    nested_cv=False,
                    inner_folds=0,
                )
                score = _rank_ic(
                    candidate.predict(inner_val),
                    pd.to_numeric(inner_val[self.score_column], errors="coerce"),
                )
                if np.isfinite(score):
                    fold_scores.append(score)
            if fold_scores:
                scored.append((float(np.mean(fold_scores)), params))
        if not scored:
            return CATBOOST_PARAM_GRID[0]
        return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]

    def _transform_features(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            msg = f"missing CatBoost feature columns: {missing}"
            raise ValueError(msg)
        x = frame.loc[:, list(self.feature_columns)].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.medians = x.median().fillna(0)
        if self.medians is None:
            msg = "CatBoost preprocessing medians are unavailable"
            raise ValueError(msg)
        return x.fillna(self.medians)

    def _target(self, frame: pd.DataFrame) -> pd.Series:
        if self.target_column not in frame.columns:
            msg = f"missing CatBoost target column: {self.target_column}"
            raise ValueError(msg)
        return pd.to_numeric(frame[self.target_column], errors="coerce")

    def _coerce_target(self, target: pd.Series) -> np.ndarray:
        if self.task == "classification":
            return target.fillna(0).astype(int).to_numpy()
        return target.to_numpy(dtype=float)

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        import catboost as cb

        common = {
            **params,
            "random_seed": self.random_seed,
            "allow_writing_files": False,
            "thread_count": 1,
            "verbose": False,
        }
        if self.task == "classification":
            return cb.CatBoostClassifier(loss_function="Logloss", **common)
        return cb.CatBoostRegressor(loss_function="RMSE", **common)


class BlendedForecastModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        return_target_column: str,
        random_seed: int,
        nested_cv: bool,
        inner_folds: int,
    ) -> None:
        self.return_model = RidgeForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=return_target_column,
        )
        self.lightgbm_model = LightGBMForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=return_target_column,
            task="regression",
            score_column=return_target_column,
            random_seed=random_seed,
            nested_cv=nested_cv,
            inner_folds=inner_folds,
        )

    def predict(self, features: pd.DataFrame) -> list[float]:
        blended = _zscore(np.asarray(self.return_model.predict(features), dtype=float)) + _zscore(
            np.asarray(self.lightgbm_model.predict(features), dtype=float)
        )
        return blended.astype(float).tolist()


def run_phase2(config: Phase2Config) -> pd.DataFrame:
    """Run Phase 2 experiments, sweeps, tracking artifacts, and markdown reports."""

    artifacts = _load_phase1_artifacts(config.input_run_root)
    artifacts = _filter_artifacts_by_liquidity(artifacts, config)
    feature_columns = config.feature_columns or _default_feature_columns(artifacts["panel"])
    if not feature_columns:
        msg = "no Phase 2 feature columns are available in the panel"
        raise ValueError(msg)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_by_id: dict[str, dict[str, Any]] = {}
    backtests: dict[str, pd.DataFrame] = {}
    deferred_benchmarks: list[str] = []

    for experiment_id in config.experiments:
        if experiment_id in {"E1", "E2"}:
            deferred_benchmarks.append(experiment_id)
            continue
        result = _run_single_experiment(experiment_id, artifacts, feature_columns, config)
        _record_phase2_result(result, results_by_id, backtests)
        _write_experiment_report(result, config.output_dir / f"{experiment_id.lower()}.md")

    reference_dates = _reference_rebalance_dates(backtests)
    for experiment_id in deferred_benchmarks:
        result = _run_single_experiment(
            experiment_id,
            artifacts,
            feature_columns,
            config,
            reference_dates=reference_dates,
        )
        _record_phase2_result(result, results_by_id, backtests)
        _write_experiment_report(result, config.output_dir / f"{experiment_id.lower()}.md")

    results = [
        results_by_id[experiment_id]
        for experiment_id in config.experiments
        if experiment_id in results_by_id
    ]
    summary = pd.DataFrame(results)
    sweeps = (
        _run_winner_sweeps(summary, artifacts, feature_columns, config)
        if config.run_sweeps
        else pd.DataFrame()
    )
    _write_phase2_report(
        summary,
        sweeps,
        backtests,
        config.output_dir / "phase2-report.md",
        config=config,
        artifacts=artifacts,
    )
    _write_phase2_report(
        summary,
        sweeps,
        backtests,
        config.output_dir / "phase2-detailed-summary.md",
        detailed=True,
        config=config,
        artifacts=artifacts,
    )
    return summary


def _run_single_experiment(
    experiment_id: str,
    artifacts: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    config: Phase2Config,
    *,
    reference_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    spec = _experiment_spec(experiment_id, artifacts, feature_columns, config)
    if spec.benchmark_kind == "spy":
        return _run_spy_benchmark(spec, artifacts, config, reference_dates=reference_dates)
    if spec.benchmark_kind == "equal_weight":
        return _run_equal_weight_benchmark(
            spec,
            artifacts,
            config,
            reference_dates=reference_dates,
        )
    return _run_model_experiment(spec, artifacts, config)


def _record_phase2_result(
    result: dict[str, Any],
    results_by_id: dict[str, dict[str, Any]],
    backtests: dict[str, pd.DataFrame],
) -> None:
    experiment_id = str(result["experiment_id"])
    backtests[experiment_id] = result.pop("_backtest", pd.DataFrame())
    results_by_id[experiment_id] = result


def _reference_rebalance_dates(backtests: dict[str, pd.DataFrame]) -> list[str]:
    for experiment_id in ("E0", "E3", "E4", "E5", "E6", "E7", "E8"):
        frame = backtests.get(experiment_id, pd.DataFrame())
        if frame.empty or "rebalance_date" not in frame.columns:
            continue
        dates = frame["rebalance_date"].dropna().astype(str).drop_duplicates().tolist()
        if dates:
            return dates
    return []


def _experiment_spec(
    experiment_id: str,
    artifacts: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    config: Phase2Config,
) -> ExperimentSpec:
    return_target = f"fwd_return_{config.horizon_days}d"
    rank_target = f"fwd_rank_{config.horizon_days}d"
    top_target = f"fwd_is_top_tercile_{config.horizon_days}d"
    if experiment_id == "E0":
        heuristic_columns = _existing_columns(
            artifacts["panel"],
            ("momentum_252d", "volatility_63d"),
        )
        return ExperimentSpec(
            experiment_id,
            "Current heuristic",
            lambda _train: HeuristicForecastModel(),
            heuristic_columns,
        )
    if experiment_id == "E1":
        return ExperimentSpec(experiment_id, "SPY buy-and-hold", None, (), benchmark_kind="spy")
    if experiment_id == "E2":
        return ExperimentSpec(
            experiment_id,
            "Equal-weight S&P 500",
            None,
            (),
            benchmark_kind="equal_weight",
        )
    if experiment_id == "E3":
        return ExperimentSpec(
            experiment_id,
            "Ridge regression",
            lambda train: RidgeForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=return_target,
            ),
            feature_columns,
        )
    if experiment_id == "E4":
        return ExperimentSpec(
            experiment_id,
            "Ridge on rank-normalized features",
            lambda train: RidgeForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=rank_target,
                rank_normalize_features=True,
            ),
            feature_columns,
            training_target_column=rank_target,
        )
    if experiment_id == "E5":
        return ExperimentSpec(
            experiment_id,
            "LightGBM regression",
            lambda train: LightGBMForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=return_target,
                task="regression",
                score_column=return_target,
                random_seed=config.random_seed,
                nested_cv=config.lightgbm_nested_cv,
                inner_folds=config.lightgbm_inner_folds,
            ),
            feature_columns,
        )
    if experiment_id == "E6":
        return ExperimentSpec(
            experiment_id,
            "LightGBM LambdaRank",
            lambda train: LightGBMForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=rank_target,
                task="rank",
                score_column=return_target,
                random_seed=config.random_seed,
                nested_cv=config.lightgbm_nested_cv,
                inner_folds=config.lightgbm_inner_folds,
            ),
            feature_columns,
            training_target_column=rank_target,
        )
    if experiment_id == "E7":
        return ExperimentSpec(
            experiment_id,
            "LightGBM top-tercile classifier",
            lambda train: LightGBMForecastModel(
                train,
                feature_columns=feature_columns,
                target_column=top_target,
                task="classification",
                score_column=return_target,
                random_seed=config.random_seed,
                nested_cv=config.lightgbm_nested_cv,
                inner_folds=config.lightgbm_inner_folds,
            ),
            feature_columns,
            training_target_column=top_target,
        )
    if experiment_id == "E8":
        return ExperimentSpec(
            experiment_id,
            "Linear blend of ridge + LightGBM regression",
            lambda train: BlendedForecastModel(
                train,
                feature_columns=feature_columns,
                return_target_column=return_target,
                random_seed=config.random_seed,
                nested_cv=config.lightgbm_nested_cv,
                inner_folds=config.lightgbm_inner_folds,
            ),
            feature_columns,
        )
    msg = f"Unknown Phase 2 experiment id: {experiment_id}"
    raise ValueError(msg)


def _run_model_experiment(
    spec: ExperimentSpec,
    artifacts: dict[str, pd.DataFrame],
    config: Phase2Config,
) -> dict[str, Any]:
    if spec.model_factory is None:
        msg = f"{spec.experiment_id} does not have a model factory"
        raise ValueError(msg)
    backtest_config = _backtest_config_for_spec(spec, config)
    backtest = run_walk_forward_backtest(
        artifacts["panel"],
        artifacts["labels"],
        artifacts["returns"],
        spec.model_factory,
        config.optimizer,
        backtest_config,
    )
    metrics = _metrics_from_backtest(backtest, artifacts["benchmark"], config)
    predictions = _predictions_from_backtest(backtest)
    _track_phase2_run(spec.experiment_id, spec.model_name, predictions, backtest, metrics, config)
    return {
        **_summary_row(spec.experiment_id, spec.model_name, metrics, "completed"),
        "_backtest": backtest,
    }


def _run_spy_benchmark(
    spec: ExperimentSpec,
    artifacts: dict[str, pd.DataFrame],
    config: Phase2Config,
    *,
    reference_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    benchmark = _benchmark_for_horizon(artifacts["benchmark"], config.horizon_days)
    benchmark = _sample_rebalance_rows(
        benchmark,
        "date",
        config.backtest.rebalance_step_days,
        config.backtest.max_rebalances,
        reference_dates=reference_dates,
    )
    backtest = benchmark.rename(
        columns={"date": "rebalance_date", "spy_return": "portfolio_net_return"}
    )
    backtest["portfolio_gross_return"] = backtest["portfolio_net_return"]
    backtest["turnover"] = 0.0
    metrics = {
        "portfolio": portfolio_metrics(
            backtest["portfolio_net_return"],
            _periods_per_year(config),
        )
    }
    _track_phase2_run(
        spec.experiment_id,
        spec.model_name,
        pd.DataFrame(),
        backtest,
        metrics,
        config,
    )
    return {
        **_summary_row(spec.experiment_id, spec.model_name, metrics, "completed"),
        "_backtest": backtest,
    }


def _run_equal_weight_benchmark(
    spec: ExperimentSpec,
    artifacts: dict[str, pd.DataFrame],
    config: Phase2Config,
    *,
    reference_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    target_col = f"fwd_return_{config.horizon_days}d"
    backtest = (
        artifacts["labels"]
        .dropna(subset=[target_col])
        .groupby("date", as_index=False)
        .agg(portfolio_net_return=(target_col, "mean"))
        .rename(columns={"date": "rebalance_date"})
    )
    backtest = _sample_rebalance_rows(
        backtest,
        "rebalance_date",
        config.backtest.rebalance_step_days,
        config.backtest.max_rebalances,
        reference_dates=reference_dates,
    )
    backtest["portfolio_gross_return"] = backtest["portfolio_net_return"]
    backtest["turnover"] = np.nan
    metrics = {
        "portfolio": portfolio_metrics(
            backtest["portfolio_net_return"],
            _periods_per_year(config),
        )
    }
    benchmark = _benchmark_for_horizon(artifacts["benchmark"], config.horizon_days)
    if not benchmark.empty:
        portfolio_returns = backtest.rename(
            columns={"rebalance_date": "date", "portfolio_net_return": "portfolio_return"}
        )
        metrics["benchmark_relative"] = benchmark_relative_metrics(
            portfolio_returns[["date", "portfolio_return"]],
            benchmark.rename(columns={"spy_return": "benchmark_return"}),
            EvaluationConfig(periods_per_year=_periods_per_year(config)),
        )
    _track_phase2_run(
        spec.experiment_id,
        spec.model_name,
        pd.DataFrame(),
        backtest,
        metrics,
        config,
    )
    return {
        **_summary_row(spec.experiment_id, spec.model_name, metrics, "completed"),
        "_backtest": backtest,
    }


def _run_winner_sweeps(
    summary: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    config: Phase2Config,
) -> pd.DataFrame:
    winner_id = _select_sweep_model(summary)
    if winner_id is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []

    for horizon in config.sweep_horizons:
        if f"fwd_return_{horizon}d" not in artifacts["labels"].columns:
            continue
        row = _run_sweep_case(
            "horizon",
            f"h={horizon}",
            winner_id,
            artifacts,
            feature_columns,
            _replace_config(
                config,
                horizon_days=horizon,
                backtest=_replace_backtest(
                    config.backtest,
                    horizon_days=horizon,
                    training_target_column=None,
                    rebalance_step_days=horizon,
                    embargo_days=max(horizon + 10, config.backtest.embargo_days),
                ),
            ),
        )
        rows.append(row)

    for cost_bps in config.sweep_cost_bps:
        rows.append(
            _run_sweep_case(
                "cost_bps",
                f"{cost_bps:g}",
                winner_id,
                artifacts,
                feature_columns,
                _replace_config(
                    config,
                    backtest=_replace_backtest(config.backtest, cost_bps=cost_bps),
                ),
            )
        )

    for penalty in config.sweep_turnover_penalties:
        rows.append(
            _run_sweep_case(
                "lambda_turnover",
                f"{penalty:g}",
                winner_id,
                artifacts,
                feature_columns,
                _replace_config(
                    config,
                    optimizer=config.optimizer.model_copy(update={"lambda_turnover": penalty}),
                ),
            )
        )

    for max_weight in config.sweep_max_weights:
        rows.append(
            _run_sweep_case(
                "max_weight",
                f"{max_weight:g}",
                winner_id,
                artifacts,
                feature_columns,
                _replace_config(
                    config,
                    optimizer=config.optimizer.model_copy(update={"max_weight": max_weight}),
                ),
            )
        )

    for cadence_name, step_days in config.sweep_cadences:
        horizon = (
            step_days
            if f"fwd_return_{step_days}d" in artifacts["labels"].columns
            else config.horizon_days
        )
        rows.append(
            _run_sweep_case(
                "cadence",
                cadence_name,
                winner_id,
                artifacts,
                feature_columns,
                _replace_config(
                    config,
                    horizon_days=horizon,
                    backtest=_replace_backtest(
                        config.backtest,
                        horizon_days=horizon,
                        training_target_column=None,
                        rebalance_step_days=step_days,
                        embargo_days=max(horizon + 10, config.backtest.embargo_days),
                    ),
                ),
            )
        )

    return pd.DataFrame(rows)


def _run_sweep_case(
    sweep_name: str,
    sweep_value: str,
    winner_id: str,
    artifacts: dict[str, pd.DataFrame],
    feature_columns: tuple[str, ...],
    config: Phase2Config,
) -> dict[str, Any]:
    spec = _experiment_spec(winner_id, artifacts, feature_columns, config)
    if spec.benchmark_kind != "model":
        return {"sweep": sweep_name, "value": sweep_value, "status": "skipped"}
    try:
        result = _run_model_experiment(
            ExperimentSpec(
                experiment_id=f"sweep_{sweep_name}_{_slug_value(sweep_value)}",
                model_name=f"{spec.model_name} sweep",
                model_factory=spec.model_factory,
                feature_columns=spec.feature_columns,
                training_target_column=spec.training_target_column,
            ),
            artifacts,
            config,
        )
    except Exception as exc:
        return {
            "sweep": sweep_name,
            "value": sweep_value,
            "winner_model": winner_id,
            "status": "failed",
            "error": str(exc),
        }
    return {
        "sweep": sweep_name,
        "value": sweep_value,
        "winner_model": winner_id,
        "sharpe": result.get("sharpe"),
        "annualized_return": result.get("annualized_return"),
        "max_drawdown": result.get("max_drawdown"),
        "information_ratio": result.get("information_ratio"),
        "status": result.get("status"),
    }


def _backtest_config_for_spec(spec: ExperimentSpec, config: Phase2Config) -> BacktestConfig:
    return BacktestConfig(
        horizon_days=config.horizon_days,
        training_target_column=spec.training_target_column,
        rebalance_step_days=config.backtest.rebalance_step_days,
        embargo_days=config.backtest.embargo_days,
        cost_bps=config.backtest.cost_bps,
        covariance_lookback_days=config.backtest.covariance_lookback_days,
        feature_columns=spec.feature_columns,
        max_rebalances=config.backtest.max_rebalances,
    )


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
            random_seed=config.random_seed,
        ),
    )
    portfolio_series = backtest.drop_duplicates("rebalance_date")["portfolio_net_return"]
    metrics["portfolio"] = portfolio_metrics(portfolio_series, _periods_per_year(config))
    benchmark_h = _benchmark_for_horizon(benchmark, config.horizon_days)
    if not benchmark_h.empty:
        portfolio_returns = backtest.drop_duplicates("rebalance_date")[
            ["rebalance_date", "portfolio_net_return"]
        ].rename(columns={"rebalance_date": "date", "portfolio_net_return": "portfolio_return"})
        metrics["benchmark_relative"] = benchmark_relative_metrics(
            portfolio_returns,
            benchmark_h.rename(columns={"spy_return": "benchmark_return"}),
            EvaluationConfig(periods_per_year=_periods_per_year(config)),
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
            "max_assets": config.max_assets,
            "optimizer": config.optimizer.model_dump(mode="json"),
            "backtest": {
                "horizon_days": config.backtest.horizon_days,
                "rebalance_step_days": config.backtest.rebalance_step_days,
                "embargo_days": config.backtest.embargo_days,
                "cost_bps": config.backtest.cost_bps,
            },
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
        "active_return": benchmark.get("active_return"),
        "tracking_error": benchmark.get("tracking_error"),
        "information_ratio": benchmark.get("information_ratio"),
        "ir_observations": benchmark.get("observations"),
        "metrics": metrics,
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


def _filter_artifacts_by_liquidity(
    artifacts: dict[str, pd.DataFrame],
    config: Phase2Config,
) -> dict[str, pd.DataFrame]:
    if config.max_assets is None:
        return artifacts
    panel = artifacts["panel"]
    if config.liquidity_column not in panel.columns:
        msg = f"cannot apply max_assets; missing liquidity column {config.liquidity_column}"
        raise ValueError(msg)
    latest_date = panel["date"].max()
    top_tickers = (
        panel.loc[panel["date"] == latest_date]
        .sort_values(config.liquidity_column, ascending=False)
        .head(config.max_assets)["ticker"]
        .astype(str)
        .tolist()
    )
    filtered = artifacts.copy()
    for key in ["panel", "labels", "returns"]:
        filtered[key] = (
            artifacts[key].loc[artifacts[key]["ticker"].astype(str).isin(top_tickers)].copy()
        )
    return filtered


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


def _sample_rebalance_rows(
    frame: pd.DataFrame,
    date_column: str,
    rebalance_step_days: int,
    max_rebalances: int | None = None,
    *,
    reference_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.copy()
    sorted_frame[date_column] = pd.to_datetime(sorted_frame[date_column]).dt.normalize()
    if reference_dates:
        aligned_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(reference_dates))).normalize()
        sampled = sorted_frame.loc[sorted_frame[date_column].isin(aligned_dates)].copy()
    else:
        step = max(rebalance_step_days, 1)
        sampled = sorted_frame.sort_values(date_column).iloc[::step].copy()
        if max_rebalances is not None and max_rebalances <= 0:
            sampled = sampled.iloc[0:0].copy()
        elif max_rebalances is not None and len(sampled) > max_rebalances:
            sample_positions = np.linspace(0, len(sampled) - 1, max_rebalances).round()
            sampled = sampled.iloc[np.unique(sample_positions.astype(int))].copy()
    sampled = sampled.sort_values(date_column)
    sampled[date_column] = sampled[date_column].dt.date.astype(str)
    return sampled.reset_index(drop=True)


def _periods_per_year(config: Phase2Config) -> int:
    period_days = max(config.backtest.rebalance_step_days, 1)
    return max(1, round(252 / period_days))


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    ordered_dates = pd.to_datetime(frame["date"])
    return frame.assign(_date=ordered_dates).groupby("_date", sort=True).size().astype(int).tolist()


def _inner_cv_date_folds(
    dates: pd.DatetimeIndex,
    *,
    max_folds: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    if max_folds <= 0:
        return []
    fold_count = min(max_folds, 3)
    boundaries = np.linspace(0.55, 0.85, fold_count)
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for boundary in boundaries:
        train_end = int(len(dates) * boundary)
        val_end = min(int(len(dates) * (boundary + 0.1)), len(dates))
        if train_end <= 0 or val_end <= train_end:
            continue
        folds.append((dates[:train_end], dates[train_end:val_end]))
    return folds


def _rank_ic(predictions: Sequence[float], target: pd.Series) -> float:
    frame = pd.DataFrame({"prediction": predictions, "target": target}).dropna()
    if len(frame) < 3:
        return float("nan")
    return float(frame["prediction"].corr(frame["target"], method="spearman"))


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0 or not np.isfinite(std):
        return np.zeros(len(values))
    return (values - float(np.mean(values))) / std


def _select_sweep_model(summary: pd.DataFrame) -> str | None:
    candidates = summary.loc[
        (summary["status"] == "completed")
        & summary["experiment_id"].isin(["E3", "E4", "E5", "E6", "E7", "E8"])
    ].copy()
    if candidates.empty:
        return None
    candidates["_rank_value"] = candidates["sharpe"].fillna(-np.inf)
    return str(candidates.sort_values("_rank_value", ascending=False).iloc[0]["experiment_id"])


def _replace_config(config: Phase2Config, **updates: Any) -> Phase2Config:
    return Phase2Config(
        input_run_root=updates.get("input_run_root", config.input_run_root),
        output_dir=updates.get("output_dir", config.output_dir),
        tracking_root=updates.get("tracking_root", config.tracking_root),
        experiments=updates.get("experiments", config.experiments),
        horizon_days=updates.get("horizon_days", config.horizon_days),
        bootstrap_samples=updates.get("bootstrap_samples", config.bootstrap_samples),
        force=True,
        feature_columns=updates.get("feature_columns", config.feature_columns),
        max_assets=updates.get("max_assets", config.max_assets),
        liquidity_column=updates.get("liquidity_column", config.liquidity_column),
        optimizer=updates.get("optimizer", config.optimizer),
        backtest=updates.get("backtest", config.backtest),
        run_sweeps=False,
        sweep_horizons=config.sweep_horizons,
        sweep_cost_bps=config.sweep_cost_bps,
        sweep_turnover_penalties=config.sweep_turnover_penalties,
        sweep_max_weights=config.sweep_max_weights,
        sweep_cadences=config.sweep_cadences,
        random_seed=updates.get("random_seed", config.random_seed),
        lightgbm_nested_cv=updates.get("lightgbm_nested_cv", config.lightgbm_nested_cv),
        lightgbm_inner_folds=updates.get("lightgbm_inner_folds", config.lightgbm_inner_folds),
    )


def _replace_backtest(backtest: BacktestConfig, **updates: Any) -> BacktestConfig:
    return BacktestConfig(
        horizon_days=updates.get("horizon_days", backtest.horizon_days),
        training_target_column=updates.get(
            "training_target_column", backtest.training_target_column
        ),
        rebalance_step_days=updates.get("rebalance_step_days", backtest.rebalance_step_days),
        embargo_days=updates.get("embargo_days", backtest.embargo_days),
        cost_bps=updates.get("cost_bps", backtest.cost_bps),
        covariance_lookback_days=updates.get(
            "covariance_lookback_days", backtest.covariance_lookback_days
        ),
        feature_columns=updates.get("feature_columns", backtest.feature_columns),
        max_rebalances=updates.get("max_rebalances", backtest.max_rebalances),
    )


def _slug_value(value: str) -> str:
    return value.lower().replace(".", "p").replace("=", "").replace(" ", "_").replace("-", "_")


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


def _write_phase2_report(
    summary: pd.DataFrame,
    sweeps: pd.DataFrame,
    backtests: dict[str, pd.DataFrame],
    path: Path,
    *,
    detailed: bool = False,
    config: Phase2Config | None = None,
    artifacts: dict[str, pd.DataFrame] | None = None,
) -> None:
    display = summary.drop(columns=["metrics"], errors="ignore")
    gating = _gating_decision(summary, backtests)
    sections = [
        "# Phase 2 Experimentation and Backtesting Summary" if detailed else "# Phase 2 Report",
        "",
        f"> {SURVIVORSHIP_BANNER}",
        "",
    ]
    if detailed:
        sections.extend(
            [
                "## Experiment Design",
                "",
                (
                    "Phase 2 compares the current heuristic forecast against market, "
                    "equal-weight, linear, LightGBM, ranking, classification, and blended ML "
                    "models. All model experiments feed forecast scores into the same frozen "
                    "long-only optimizer so model differences are attributable to `mu` quality. "
                    "The continuation objective for this report is SPY-relative: find an ML "
                    "portfolio with higher Sharpe than SPY and positive information ratio."
                ),
                (
                    "When `max_assets` is configured, experiments run on the most liquid assets "
                    "by latest `dollar_volume_21d`; this keeps optimizer runtimes manageable and "
                    "is explicitly part of the experiment scope."
                ),
                "",
                *_run_configuration_section(config, artifacts, backtests),
                *_model_result_section(summary, backtests),
                "## SPY-Relative IR Calculation",
                "",
                (
                    "Information ratio is calculated only on exact rebalance-date matches between "
                    "portfolio period returns and SPY forward returns for the same horizon. "
                    "`active_return = mean(portfolio_return - spy_return) * periods_per_year`, "
                    "`tracking_error = std(portfolio_return - spy_return) * "
                    "sqrt(periods_per_year)`, "
                    "and `IR = active_return / tracking_error`. The results table includes the "
                    "annualized active return, annualized tracking error, and aligned observation "
                    "count used in that calculation."
                ),
                "",
                "## Optimization Model",
                "",
                (
                    "The optimizer maximizes `mu^T w - gamma * w^T Sigma w - "
                    "lambda_turnover * ||w - w_prev||_1`, with long-only weights, a max-weight "
                    "constraint, and configurable transaction costs applied during backtesting. "
                    "This is the turnover-aware configuration required for weekly rebalancing."
                ),
                "",
                "## ML Models",
                "",
                (
                    "E3/E4 use train-only median imputation and standardization. E5 uses "
                    "LightGBM regression, E6 uses LightGBM LambdaRank, E7 uses a LightGBM "
                    "top-tercile classifier, and E8 blends Ridge and LightGBM regression after "
                    "z-scoring predictions."
                ),
                "",
            ]
        )
    sections.extend(
        [
            "## Results",
            "",
            _markdown_table(display),
            "",
            "## Winner Sweeps",
            "",
            _markdown_table(sweeps) if not sweeps.empty else "_No sweeps were run._",
            "",
            "## Gating Decision",
            "",
            gating,
            "",
            "## Caveats",
            "",
            "- Current-constituent S&P 500 universe creates survivorship bias.",
            "- Transaction-cost model is flat and excludes market impact.",
            "- LightGBM tuning grid is intentionally small for repeatable local execution.",
            "",
        ]
    )
    path.write_text("\n".join(sections), encoding="utf-8")


def _run_configuration_section(
    config: Phase2Config | None,
    artifacts: dict[str, pd.DataFrame] | None,
    backtests: dict[str, pd.DataFrame],
) -> list[str]:
    if config is None:
        return []

    rows = ["## Run Configuration", ""]
    rows.append(f"- Source artifacts: `{config.input_run_root}`.")
    if artifacts:
        panel = artifacts.get("panel", pd.DataFrame())
        labels = artifacts.get("labels", pd.DataFrame())
        rows.append(f"- Feature panel window: {_date_window(panel, 'date')}.")
        rows.append(f"- Label panel window: {_date_window(labels, 'date')}.")
        ticker_count = (
            int(panel["ticker"].astype(str).nunique())
            if not panel.empty and "ticker" in panel.columns
            else 0
        )
        universe = (
            f"top {config.max_assets} assets by latest `{config.liquidity_column}`"
            if config.max_assets is not None
            else "all available assets"
        )
        rows.append(f"- Experiment universe: {universe}; {ticker_count} tickers in scope.")

    reference_dates = _reference_rebalance_dates(backtests)
    if reference_dates:
        rows.append(
            "- Completed rebalance observations: "
            f"{len(reference_dates)} dates from {reference_dates[0]} to {reference_dates[-1]}."
        )
    requested = (
        str(config.backtest.max_rebalances)
        if config.backtest.max_rebalances is not None
        else "all available"
    )
    lightgbm_mode = (
        "nested inner-CV enabled" if config.lightgbm_nested_cv else "fixed grid, no nested CV"
    )
    rows.extend(
        [
            (
                "- Backtest setup: "
                f"{config.horizon_days}-trading-day target, "
                f"{config.backtest.rebalance_step_days}-business-day rebalance step, "
                f"{requested} requested rebalance samples, "
                f"{config.backtest.cost_bps:g} bps transaction cost."
            ),
            (
                "- Optimizer setup: "
                f"max_weight={config.optimizer.max_weight:g}, "
                f"risk_aversion={config.optimizer.risk_aversion:g}, "
                f"lambda_turnover={config.optimizer.lambda_turnover:g}."
            ),
            (f"- LightGBM execution: {lightgbm_mode} with random_seed={config.random_seed}."),
            "",
        ]
    )
    return rows


def _model_result_section(
    summary: pd.DataFrame,
    backtests: dict[str, pd.DataFrame],
) -> list[str]:
    if summary.empty:
        return []

    best_ic = _best_completed_row(summary, ["E3", "E4", "E5", "E6", "E7", "E8"], "pearson_ic")
    best_rank_ic = _best_completed_row(
        summary,
        ["E3", "E4", "E5", "E6", "E7", "E8"],
        "rank_ic",
    )
    best_portfolio = _best_completed_row(
        summary,
        ["E3", "E4", "E5", "E6", "E7", "E8"],
        "sharpe",
    )

    rows = ["## Experimentation Outcome", ""]
    if best_ic is not None:
        rows.append(
            "- Best ML model by Pearson IC: "
            f"{best_ic['experiment_id']} ({best_ic['model_name']}), "
            f"IC {_format_table_value(best_ic['pearson_ic'])}."
        )
    if best_rank_ic is not None:
        rows.append(
            "- Best ML model by rank IC: "
            f"{best_rank_ic['experiment_id']} ({best_rank_ic['model_name']}), "
            f"rank IC {_format_table_value(best_rank_ic['rank_ic'])}."
        )
    if best_portfolio is not None:
        turnover = _mean_turnover(backtests.get(str(best_portfolio["experiment_id"])))
        rows.append(
            "- Best optimized ML portfolio: "
            f"{best_portfolio['experiment_id']} ({best_portfolio['model_name']}), "
            f"Sharpe {_format_table_value(best_portfolio['sharpe'])}, "
            f"annualized return {_format_table_value(best_portfolio['annualized_return'])}, "
            f"SPY-relative IR {_format_table_value(best_portfolio['information_ratio'])}, "
            f"annualized active return {_format_table_value(best_portfolio['active_return'])}, "
            f"mean turnover {_format_table_value(turnover)}."
        )
    rows.extend(
        [
            (
                "- Interpretation: predictive IC and optimized portfolio quality can diverge; "
                "the promotion decision is therefore based on the portfolio backtest versus SPY, "
                "not on predictive IC alone."
            ),
            "",
        ]
    )
    return rows


def _best_completed_row(
    summary: pd.DataFrame,
    experiment_ids: Sequence[str],
    metric_column: str,
) -> pd.Series | None:
    if metric_column not in summary.columns:
        return None
    candidates = summary.loc[
        (summary["status"] == "completed") & summary["experiment_id"].isin(experiment_ids)
    ].copy()
    if candidates.empty:
        return None
    candidates["_rank_value"] = pd.to_numeric(candidates[metric_column], errors="coerce")
    candidates = candidates.dropna(subset=["_rank_value"])
    if candidates.empty:
        return None
    return candidates.sort_values("_rank_value", ascending=False).iloc[0]


def _mean_turnover(backtest: pd.DataFrame | None) -> float | None:
    if backtest is None or backtest.empty or "turnover" not in backtest.columns:
        return None
    by_date = backtest.drop_duplicates("rebalance_date")
    value = pd.to_numeric(by_date["turnover"], errors="coerce").mean()
    return float(value) if pd.notna(value) else None


def _date_window(frame: pd.DataFrame, date_column: str) -> str:
    if frame.empty or date_column not in frame.columns:
        return "unavailable"
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if dates.empty:
        return "unavailable"
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_table_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_table_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _gating_decision(summary: pd.DataFrame, backtests: dict[str, pd.DataFrame]) -> str:
    completed = summary.loc[summary["status"] == "completed"].copy()
    if completed.empty or "E1" not in set(completed["experiment_id"]):
        return "NO-GO: Phase 2 has not produced a completed SPY benchmark yet."
    spy = completed.loc[completed["experiment_id"] == "E1"].iloc[0]
    contenders = completed.loc[
        completed["experiment_id"].isin(["E3", "E4", "E5", "E6", "E7", "E8"])
    ]
    winners = contenders.loc[
        (contenders["sharpe"].fillna(-np.inf) > (spy["sharpe"] or -np.inf))
        & (contenders["information_ratio"].fillna(-np.inf) > 0)
        & (contenders["active_return"].fillna(-np.inf) > 0)
    ]
    if winners.empty:
        return (
            "NO-GO: no Phase 2 ML portfolio currently beats SPY on Sharpe while also "
            "delivering positive annualized active return and positive SPY-relative IR."
        )
    best = winners.sort_values(["sharpe", "information_ratio"], ascending=False).iloc[0]
    ci = _sharpe_difference_ci(
        backtests.get(str(best["experiment_id"]), pd.DataFrame()),
        backtests.get("E1", pd.DataFrame()),
    )
    ci_text = f" Sharpe-difference 95% CI vs SPY: [{ci[0]:.3f}, {ci[1]:.3f}]." if ci else ""
    if ci and ci[0] <= 0:
        return (
            f"PROVISIONAL: {best['experiment_id']} beats SPY on Sharpe and has positive IR, "
            f"but the Sharpe-difference CI includes zero.{ci_text}"
        )
    return (
        f"GO: {best['experiment_id']} beats SPY with Sharpe {best['sharpe']:.3f}, "
        f"annualized active return {best['active_return']:.3f}, and IR "
        f"{best['information_ratio']:.3f}.{ci_text}"
    )


def _sharpe_difference_ci(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    samples: int = 500,
) -> tuple[float, float] | None:
    if candidate.empty or baseline.empty:
        return None
    left = candidate.drop_duplicates("rebalance_date")[
        ["rebalance_date", "portfolio_net_return"]
    ].rename(columns={"portfolio_net_return": "candidate_return"})
    right = baseline.drop_duplicates("rebalance_date")[
        ["rebalance_date", "portfolio_net_return"]
    ].rename(columns={"portfolio_net_return": "baseline_return"})
    merged = left.merge(right, on="rebalance_date", how="inner").dropna()
    if len(merged) < 5:
        return None
    periods_per_year = _infer_periods_per_year(merged["rebalance_date"])
    rng = np.random.default_rng(42)
    diffs: list[float] = []
    for _ in range(samples):
        sample = merged.iloc[rng.choice(len(merged), size=len(merged), replace=True)]
        candidate_sharpe = portfolio_metrics(
            sample["candidate_return"],
            periods_per_year,
        ).get("sharpe", 0.0)
        baseline_sharpe = portfolio_metrics(
            sample["baseline_return"],
            periods_per_year,
        ).get("sharpe", 0.0)
        diffs.append(float(candidate_sharpe) - float(baseline_sharpe))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def _infer_periods_per_year(dates: pd.Series) -> int:
    parsed = pd.to_datetime(dates).sort_values()
    if len(parsed) < 2:
        return 252
    median_days = max(float(parsed.diff().dt.days.dropna().median()), 1.0)
    return max(1, round(365.25 / median_days))
