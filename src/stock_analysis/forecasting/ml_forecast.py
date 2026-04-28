from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from stock_analysis.config import ForecastConfig
from stock_analysis.domain.schemas import validate_columns
from stock_analysis.forecasting.calibration import (
    ForecastCalibrationResult,
    ModelFactory,
    calibrate_forecast_scores,
    disabled_calibration_diagnostics,
    empty_calibration_predictions,
)
from stock_analysis.ml.autoresearch_candidate import (
    CandidateSpec,
    build_model_factory,
    get_candidate,
    resolve_feature_columns,
)
from stock_analysis.ml.phase2 import DEFAULT_FEATURE_CANDIDATES, BlendedForecastModel


@dataclass(frozen=True)
class MLOptimizerInputResult:
    optimizer_input: pd.DataFrame
    covariance: pd.DataFrame
    calibration_predictions: pd.DataFrame
    calibration_diagnostics: pd.DataFrame


def build_ml_optimizer_inputs(
    feature_panel: pd.DataFrame,
    labels_panel: pd.DataFrame,
    returns: pd.DataFrame,
    config: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the selected Phase 2 blend and build optimizer inputs for the latest date."""

    result = build_ml_optimizer_inputs_with_artifacts(feature_panel, labels_panel, returns, config)
    return result.optimizer_input, result.covariance


def build_ml_optimizer_inputs_with_artifacts(
    feature_panel: pd.DataFrame,
    labels_panel: pd.DataFrame,
    returns: pd.DataFrame,
    config: ForecastConfig,
) -> MLOptimizerInputResult:
    """Train the selected Phase 2 blend and return optimizer inputs plus calibration artifacts."""

    if feature_panel.empty:
        msg = "Cannot build ML optimizer inputs from an empty feature panel"
        raise ValueError(msg)

    panel = _prepare_panel(feature_panel)
    labels = _prepare_labels(labels_panel)
    candidate = _candidate_from_model_version(config.ml_model_version)
    target_column = _target_column(config, candidate)
    if target_column not in labels.columns:
        msg = f"labels panel is missing required ML target column: {target_column}"
        raise ValueError(msg)

    latest_date = panel["date"].max()
    latest_features = panel.loc[panel["date"] == latest_date].copy()
    if latest_features.empty:
        msg = "No latest feature rows are available for ML inference"
        raise ValueError(msg)
    calibration_panel = panel.copy()

    top_tickers = _top_liquidity_tickers(latest_features, config.ml_max_assets)
    if top_tickers is not None:
        panel = panel.loc[panel["ticker"].astype(str).isin(top_tickers)].copy()
        latest_features = latest_features.loc[
            latest_features["ticker"].astype(str).isin(top_tickers)
        ].copy()

    feature_columns = (
        resolve_feature_columns(panel, candidate)
        if candidate is not None
        else _feature_columns(panel, config)
    )
    train = panel.merge(labels[["ticker", "date", target_column]], on=["ticker", "date"])
    train = train.loc[(train["date"] < latest_date) & train[target_column].notna()].copy()
    if train.empty:
        msg = "No labeled training rows are available for ML optimizer input generation"
        raise ValueError(msg)

    model_factory = _model_factory(candidate, feature_columns, config, target_column)
    model = model_factory(train)
    forecast_score = np.asarray(model.predict(latest_features), dtype=float) * float(
        config.ml_score_scale
    )

    optimizer_input = latest_features.copy()
    optimizer_input["forecast_score"] = forecast_score
    calibration = _calibration_result(
        panel=calibration_panel,
        labels=labels,
        latest_features=optimizer_input,
        model_factory=model_factory,
        feature_columns=feature_columns,
        target_column=target_column,
        horizon_days=_horizon_days(config, candidate),
        config=config,
    )
    calibration_enabled = (
        config.ml_calibration_enabled
        and config.ml_use_calibrated_expected_return
        and calibration.is_calibrated
    )
    if calibration_enabled:
        calibrated = calibration.calibrated_latest.reindex(optimizer_input.index)
        optimizer_input["calibrated_expected_return"] = calibrated
        optimizer_input["expected_return"] = calibrated
        optimizer_input["expected_return_is_calibrated"] = True
    else:
        optimizer_input["calibrated_expected_return"] = np.nan
        optimizer_input["expected_return"] = optimizer_input["forecast_score"]
        optimizer_input["expected_return_is_calibrated"] = False
    optimizer_input["calibration_status"] = _calibration_status(calibration)
    optimizer_input["volatility"] = _volatility(optimizer_input)
    optimizer_input["eligible_for_optimization"] = (
        np.isfinite(optimizer_input["expected_return"].astype(float))
        & np.isfinite(optimizer_input["volatility"].astype(float))
        & (optimizer_input["volatility"].astype(float) > 0)
    )
    optimizer_input = _apply_benchmark_expected_return_gate(optimizer_input, config)
    optimizer_input["forecast_engine"] = "ml"
    optimizer_input["forecast_model_version"] = config.ml_model_version
    optimizer_input["as_of_date"] = latest_date.date().isoformat()

    columns = [
        "ticker",
        "security",
        "gics_sector",
        *[column for column in ["is_benchmark_candidate"] if column in optimizer_input.columns],
        "expected_return",
        "forecast_score",
        "calibrated_expected_return",
        "volatility",
        "benchmark_expected_return",
        "benchmark_expected_return_margin",
        "benchmark_return_gate_passed",
        "eligible_for_optimization",
        "forecast_engine",
        "forecast_model_version",
        "expected_return_is_calibrated",
        "calibration_status",
        "as_of_date",
        *feature_columns,
    ]
    for column in columns:
        if column not in optimizer_input.columns:
            optimizer_input[column] = pd.NA

    result = optimizer_input[columns].sort_values("ticker").reset_index(drop=True)
    covariance = _covariance_matrix(
        returns,
        result,
        config.covariance_lookback_days,
        scale_days=_horizon_days(config, candidate) if calibration_enabled else 252,
    )
    return MLOptimizerInputResult(
        optimizer_input=validate_columns(result, "optimizer_input"),
        covariance=covariance,
        calibration_predictions=calibration.predictions,
        calibration_diagnostics=calibration.diagnostics,
    )


def _apply_benchmark_expected_return_gate(
    optimizer_input: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    result = optimizer_input.copy()
    margin = float(config.ml_min_active_expected_return_vs_benchmark)
    result["benchmark_expected_return"] = np.nan
    result["benchmark_expected_return_margin"] = margin
    result["benchmark_return_gate_passed"] = True

    if margin <= 0 or "is_benchmark_candidate" not in result.columns:
        return result

    benchmark_mask = result["is_benchmark_candidate"].fillna(False).astype(bool)
    if not benchmark_mask.any():
        return result

    expected_return = pd.to_numeric(result["expected_return"], errors="coerce")
    if "expected_return_is_calibrated" in result.columns:
        calibrated = result["expected_return_is_calibrated"].fillna(False).astype(bool)
    else:
        calibrated = pd.Series(False, index=result.index)

    benchmark_values = expected_return.where(benchmark_mask & calibrated).dropna()
    if benchmark_values.empty:
        gate_passed = benchmark_mask
    else:
        benchmark_expected_return = float(benchmark_values.max())
        result["benchmark_expected_return"] = benchmark_expected_return
        gate_passed = benchmark_mask | (
            calibrated & expected_return.ge(benchmark_expected_return + margin)
        )

    result["benchmark_return_gate_passed"] = gate_passed.fillna(False).astype(bool)
    result["eligible_for_optimization"] = (
        result["eligible_for_optimization"].fillna(False).astype(bool)
        & result["benchmark_return_gate_passed"]
    )
    return result


def _model_factory(
    candidate: CandidateSpec | None,
    feature_columns: tuple[str, ...],
    config: ForecastConfig,
    target_column: str,
) -> ModelFactory:
    if candidate is not None:
        return cast(ModelFactory, build_model_factory(candidate, feature_columns))

    def factory(train_df: pd.DataFrame) -> BlendedForecastModel:
        return BlendedForecastModel(
            train_df,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=config.ml_random_seed,
            nested_cv=config.ml_lightgbm_nested_cv,
            inner_folds=config.ml_lightgbm_inner_folds,
        )

    return factory


def _calibration_result(
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    latest_features: pd.DataFrame,
    model_factory: ModelFactory,
    feature_columns: tuple[str, ...],
    target_column: str,
    horizon_days: int,
    config: ForecastConfig,
) -> ForecastCalibrationResult:
    if not config.ml_calibration_enabled:
        return ForecastCalibrationResult(
            calibrated_latest=pd.Series(
                np.nan,
                index=latest_features.index,
                name="calibrated_expected_return",
            ),
            predictions=empty_calibration_predictions(),
            diagnostics=disabled_calibration_diagnostics(
                method=config.ml_calibration_method,
                target_column=target_column,
                horizon_days=horizon_days,
            ),
        )
    return calibrate_forecast_scores(
        panel=panel,
        labels=labels,
        latest_features=latest_features,
        model_factory=model_factory,
        feature_columns=feature_columns,
        target_column=target_column,
        horizon_days=horizon_days,
        score_scale=config.ml_score_scale,
        method=config.ml_calibration_method,
        min_observations=config.ml_calibration_min_observations,
        min_validation_observations=config.ml_calibration_min_validation_observations,
        validation_fraction=config.ml_calibration_validation_fraction,
        min_rank_ic=config.ml_calibration_min_rank_ic,
        max_mae=config.ml_calibration_max_mae,
        max_rmse=config.ml_calibration_max_rmse,
        splits=config.ml_calibration_splits,
        embargo_days=max(config.ml_calibration_embargo_days, horizon_days),
        shrinkage=config.ml_calibration_shrinkage,
        lookback_days=config.ml_calibration_lookback_days,
        max_assets_per_date=config.ml_max_assets,
    )


def _calibration_status(calibration: ForecastCalibrationResult) -> str:
    if calibration.diagnostics.empty or "calibration_status" not in calibration.diagnostics:
        return "unknown"
    return str(calibration.diagnostics["calibration_status"].iat[0])


def _prepare_panel(feature_panel: pd.DataFrame) -> pd.DataFrame:
    panel = feature_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def _prepare_labels(labels_panel: pd.DataFrame) -> pd.DataFrame:
    labels = labels_panel.copy()
    labels["date"] = pd.to_datetime(labels["date"])
    labels["ticker"] = labels["ticker"].astype(str)
    return labels


def _top_liquidity_tickers(
    latest_features: pd.DataFrame, max_assets: int | None
) -> set[str] | None:
    if max_assets is None:
        return None
    if "dollar_volume_21d" not in latest_features.columns:
        msg = "ML max-assets filtering requires dollar_volume_21d in the feature panel"
        raise ValueError(msg)
    top = (
        latest_features.assign(
            _liquidity=pd.to_numeric(latest_features["dollar_volume_21d"], errors="coerce")
        )
        .sort_values("_liquidity", ascending=False)
        .head(max_assets)
    )
    return set(top["ticker"].astype(str))


def _feature_columns(panel: pd.DataFrame, config: ForecastConfig) -> tuple[str, ...]:
    configured = tuple(config.ml_feature_columns)
    candidates = configured or DEFAULT_FEATURE_CANDIDATES
    columns = tuple(column for column in candidates if column in panel.columns)
    if not columns:
        msg = "No ML feature columns are available in the feature panel"
        raise ValueError(msg)
    if configured and len(columns) != len(configured):
        missing = sorted(set(configured) - set(columns))
        msg = f"Configured ML feature columns are missing from the panel: {missing}"
        raise ValueError(msg)
    return columns


def _candidate_from_model_version(model_version: str) -> CandidateSpec | None:
    try:
        return get_candidate(model_version)
    except ValueError:
        return None


def _target_column(config: ForecastConfig, candidate: CandidateSpec | None) -> str:
    if candidate is not None:
        return candidate.training_target_column or f"fwd_return_{candidate.horizon_days}d"
    return f"fwd_return_{config.ml_horizon_days}d"


def _horizon_days(config: ForecastConfig, candidate: CandidateSpec | None) -> int:
    if candidate is not None:
        return candidate.horizon_days
    return config.ml_horizon_days


def _volatility(frame: pd.DataFrame) -> pd.Series:
    volatility_columns = [column for column in frame.columns if column.startswith("volatility_")]
    if not volatility_columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    preferred = (
        "volatility_63d" if "volatility_63d" in volatility_columns else volatility_columns[0]
    )
    return pd.to_numeric(frame[preferred], errors="coerce")


def _covariance_matrix(
    returns: pd.DataFrame,
    optimizer_input: pd.DataFrame,
    covariance_lookback_days: int,
    *,
    scale_days: int,
) -> pd.DataFrame:
    returns_matrix = returns.copy()
    returns_matrix["date"] = pd.to_datetime(returns_matrix["date"])
    returns_matrix["ticker"] = returns_matrix["ticker"].astype(str)
    pivot = (
        returns_matrix.pivot(index="date", columns="ticker", values="return_1d")
        .sort_index()
        .tail(covariance_lookback_days)
    )
    eligible = optimizer_input.loc[
        optimizer_input["eligible_for_optimization"].astype(bool),
        "ticker",
    ].tolist()
    covariance = pivot.reindex(columns=eligible).fillna(0).cov() * scale_days
    covariance = covariance.fillna(0)
    for ticker in covariance.columns:
        diagonal_value = float(covariance.loc[[ticker], [ticker]].to_numpy(dtype=float)[0, 0])
        if diagonal_value <= 0:
            covariance.loc[ticker, ticker] = 1e-8
    return (covariance + covariance.T) / 2 if not covariance.empty else covariance
