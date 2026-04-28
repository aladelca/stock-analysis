from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analysis.config import ForecastConfig
from stock_analysis.domain.schemas import validate_columns
from stock_analysis.ml.autoresearch_candidate import (
    CandidateSpec,
    build_model_factory,
    get_candidate,
    resolve_feature_columns,
)
from stock_analysis.ml.phase2 import DEFAULT_FEATURE_CANDIDATES, BlendedForecastModel


def build_ml_optimizer_inputs(
    feature_panel: pd.DataFrame,
    labels_panel: pd.DataFrame,
    returns: pd.DataFrame,
    config: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the selected Phase 2 blend and build optimizer inputs for the latest date."""

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

    model = (
        build_model_factory(candidate, feature_columns)(train)
        if candidate is not None
        else BlendedForecastModel(
            train,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=config.ml_random_seed,
            nested_cv=config.ml_lightgbm_nested_cv,
            inner_folds=config.ml_lightgbm_inner_folds,
        )
    )
    forecast_score = np.asarray(model.predict(latest_features), dtype=float) * float(
        config.ml_score_scale
    )

    optimizer_input = latest_features.copy()
    optimizer_input["forecast_score"] = forecast_score
    optimizer_input["expected_return"] = optimizer_input["forecast_score"]
    optimizer_input["volatility"] = _volatility(optimizer_input)
    optimizer_input["eligible_for_optimization"] = (
        np.isfinite(optimizer_input["expected_return"].astype(float))
        & np.isfinite(optimizer_input["volatility"].astype(float))
        & (optimizer_input["volatility"].astype(float) > 0)
    )
    optimizer_input["forecast_engine"] = "ml"
    optimizer_input["forecast_model_version"] = config.ml_model_version
    optimizer_input["expected_return_is_calibrated"] = False
    optimizer_input["as_of_date"] = latest_date.date().isoformat()

    columns = [
        "ticker",
        "security",
        "gics_sector",
        "expected_return",
        "forecast_score",
        "volatility",
        "eligible_for_optimization",
        "forecast_engine",
        "forecast_model_version",
        "expected_return_is_calibrated",
        "as_of_date",
        *feature_columns,
    ]
    for column in columns:
        if column not in optimizer_input.columns:
            optimizer_input[column] = pd.NA

    result = optimizer_input[columns].sort_values("ticker").reset_index(drop=True)
    covariance = _covariance_matrix(returns, result, config.covariance_lookback_days)
    return validate_columns(result, "optimizer_input"), covariance


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
    covariance = pivot.reindex(columns=eligible).fillna(0).cov() * 252
    covariance = covariance.fillna(0)
    for ticker in covariance.columns:
        diagonal_value = float(covariance.loc[[ticker], [ticker]].to_numpy(dtype=float)[0, 0])
        if diagonal_value <= 0:
            covariance.loc[ticker, ticker] = 1e-8
    return (covariance + covariance.T) / 2 if not covariance.empty else covariance
