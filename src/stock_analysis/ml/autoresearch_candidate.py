from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from stock_analysis.ml.phase2 import (
    CATBOOST_PARAM_GRID,
    DEFAULT_FEATURE_CANDIDATES,
    TORCH_MLP_PARAM_GRID,
    TORCH_SEQUENCE_PARAM_GRID,
    BlendedForecastModel,
    CatBoostForecastModel,
    LightGBMForecastModel,
    RidgeForecastModel,
    TorchMLPForecastModel,
    TorchSequenceForecastModel,
)


class CandidateModel(Protocol):
    def predict(self, features: pd.DataFrame) -> Sequence[float]:
        """Return forecast scores for a rebalance cross-section."""


ScoreTransform = Callable[[np.ndarray], np.ndarray]
ModelKind = Literal[
    "ridge",
    "ridge_rank",
    "lightgbm_regression",
    "lightgbm_rank",
    "catboost_regression",
    "catboost_classification",
    "torch_mlp_regression",
    "torch_lstm_regression",
    "torch_transformer_regression",
    "e8_blend",
    "weighted_e8_blend",
    "ridge_catboost_blend",
    "weighted_e8_catboost_blend",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    description: str
    model_kind: ModelKind
    horizon_days: int = 5
    feature_columns: tuple[str, ...] = ()
    training_target_column: str | None = None
    random_seed: int = 42
    lightgbm_nested_cv: bool = False
    lightgbm_inner_folds: int = 2
    ridge_weight: float = 1.0
    lightgbm_weight: float = 1.0
    catboost_weight: float = 1.0
    catboost_params: dict[str, Any] | None = None
    torch_params: dict[str, Any] | None = None
    score_transform: ScoreTransform = lambda values: values


class TransformedCandidateModel:
    def __init__(self, model: CandidateModel, transform: ScoreTransform) -> None:
        self.model = model
        self.transform = transform

    def predict(self, features: pd.DataFrame) -> list[float]:
        values = np.asarray(self.model.predict(features), dtype=float)
        transformed = self.transform(values)
        if len(transformed) != len(values):
            msg = "candidate score transform changed the number of predictions"
            raise ValueError(msg)
        return np.asarray(transformed, dtype=float).tolist()


class WeightedBlendedCandidateModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        return_target_column: str,
        random_seed: int,
        nested_cv: bool,
        inner_folds: int,
        ridge_weight: float,
        lightgbm_weight: float,
    ) -> None:
        self.ridge_model = RidgeForecastModel(
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
        self.ridge_weight = ridge_weight
        self.lightgbm_weight = lightgbm_weight

    def predict(self, features: pd.DataFrame) -> list[float]:
        ridge_scores = zscore_scores(np.asarray(self.ridge_model.predict(features), dtype=float))
        lightgbm_scores = zscore_scores(
            np.asarray(self.lightgbm_model.predict(features), dtype=float)
        )
        blended = self.ridge_weight * ridge_scores + self.lightgbm_weight * lightgbm_scores
        return blended.astype(float).tolist()


class WeightedRidgeCatBoostCandidateModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        return_target_column: str,
        random_seed: int,
        ridge_weight: float,
        catboost_weight: float,
        catboost_params: dict[str, Any] | None,
    ) -> None:
        self.ridge_model = RidgeForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=return_target_column,
        )
        self.catboost_model = CatBoostForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=return_target_column,
            task="regression",
            score_column=return_target_column,
            random_seed=random_seed,
            params=catboost_params,
        )
        self.ridge_weight = ridge_weight
        self.catboost_weight = catboost_weight

    def predict(self, features: pd.DataFrame) -> list[float]:
        ridge_scores = zscore_scores(np.asarray(self.ridge_model.predict(features), dtype=float))
        catboost_scores = zscore_scores(
            np.asarray(self.catboost_model.predict(features), dtype=float)
        )
        blended = self.ridge_weight * ridge_scores + self.catboost_weight * catboost_scores
        return blended.astype(float).tolist()


class WeightedRidgeLightGBMCatBoostCandidateModel:
    def __init__(
        self,
        train_df: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        return_target_column: str,
        random_seed: int,
        nested_cv: bool,
        inner_folds: int,
        ridge_weight: float,
        lightgbm_weight: float,
        catboost_weight: float,
        catboost_params: dict[str, Any] | None,
    ) -> None:
        self.ridge_model = RidgeForecastModel(
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
        self.catboost_model = CatBoostForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=return_target_column,
            task="regression",
            score_column=return_target_column,
            random_seed=random_seed,
            params=catboost_params,
        )
        self.ridge_weight = ridge_weight
        self.lightgbm_weight = lightgbm_weight
        self.catboost_weight = catboost_weight

    def predict(self, features: pd.DataFrame) -> list[float]:
        ridge_scores = zscore_scores(np.asarray(self.ridge_model.predict(features), dtype=float))
        lightgbm_scores = zscore_scores(
            np.asarray(self.lightgbm_model.predict(features), dtype=float)
        )
        catboost_scores = zscore_scores(
            np.asarray(self.catboost_model.predict(features), dtype=float)
        )
        blended = (
            self.ridge_weight * ridge_scores
            + self.lightgbm_weight * lightgbm_scores
            + self.catboost_weight * catboost_scores
        )
        return blended.astype(float).tolist()


def candidate_ids() -> tuple[str, ...]:
    return tuple(CANDIDATES)


def get_candidate(candidate_id: str) -> CandidateSpec:
    try:
        return CANDIDATES[candidate_id]
    except KeyError as exc:
        msg = f"unknown autoresearch candidate: {candidate_id}"
        raise ValueError(msg) from exc


def resolve_feature_columns(panel: pd.DataFrame, candidate: CandidateSpec) -> tuple[str, ...]:
    requested = candidate.feature_columns or DEFAULT_FEATURE_CANDIDATES
    columns = tuple(column for column in requested if column in panel.columns)
    if candidate.feature_columns and len(columns) != len(candidate.feature_columns):
        missing = sorted(set(candidate.feature_columns) - set(columns))
        msg = f"candidate feature columns are missing from the panel: {missing}"
        raise ValueError(msg)
    if not columns:
        msg = "no candidate feature columns are available in the panel"
        raise ValueError(msg)
    return columns


def build_model_factory(
    candidate: CandidateSpec,
    feature_columns: tuple[str, ...],
) -> Callable[[pd.DataFrame], CandidateModel]:
    target_column = candidate.training_target_column or f"fwd_return_{candidate.horizon_days}d"

    def factory(train_df: pd.DataFrame) -> CandidateModel:
        model = _build_model(candidate, train_df, feature_columns, target_column)
        return TransformedCandidateModel(model, candidate.score_transform)

    return factory


def _build_model(
    candidate: CandidateSpec,
    train_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> CandidateModel:
    if candidate.model_kind == "ridge":
        return RidgeForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
        )
    if candidate.model_kind == "ridge_rank":
        return RidgeForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            rank_normalize_features=True,
        )
    if candidate.model_kind == "lightgbm_regression":
        return LightGBMForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            task="regression",
            score_column=target_column,
            random_seed=candidate.random_seed,
            nested_cv=candidate.lightgbm_nested_cv,
            inner_folds=candidate.lightgbm_inner_folds,
        )
    if candidate.model_kind == "lightgbm_rank":
        return LightGBMForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            task="rank",
            score_column=f"fwd_return_{candidate.horizon_days}d",
            random_seed=candidate.random_seed,
            nested_cv=candidate.lightgbm_nested_cv,
            inner_folds=candidate.lightgbm_inner_folds,
        )
    if candidate.model_kind == "catboost_regression":
        return CatBoostForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            task="regression",
            score_column=target_column,
            random_seed=candidate.random_seed,
            params=candidate.catboost_params,
        )
    if candidate.model_kind == "catboost_classification":
        return CatBoostForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            task="classification",
            score_column=f"fwd_return_{candidate.horizon_days}d",
            random_seed=candidate.random_seed,
            params=candidate.catboost_params,
        )
    if candidate.model_kind == "torch_mlp_regression":
        return TorchMLPForecastModel(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            random_seed=candidate.random_seed,
            params=candidate.torch_params,
        )
    if candidate.model_kind == "torch_lstm_regression":
        return TorchSequenceForecastModel(
            train_df,
            architecture="lstm",
            feature_columns=feature_columns,
            target_column=target_column,
            random_seed=candidate.random_seed,
            params=candidate.torch_params,
        )
    if candidate.model_kind == "torch_transformer_regression":
        return TorchSequenceForecastModel(
            train_df,
            architecture="transformer",
            feature_columns=feature_columns,
            target_column=target_column,
            random_seed=candidate.random_seed,
            params=candidate.torch_params,
        )
    if candidate.model_kind == "e8_blend":
        return BlendedForecastModel(
            train_df,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=candidate.random_seed,
            nested_cv=candidate.lightgbm_nested_cv,
            inner_folds=candidate.lightgbm_inner_folds,
        )
    if candidate.model_kind == "weighted_e8_blend":
        return WeightedBlendedCandidateModel(
            train_df,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=candidate.random_seed,
            nested_cv=candidate.lightgbm_nested_cv,
            inner_folds=candidate.lightgbm_inner_folds,
            ridge_weight=candidate.ridge_weight,
            lightgbm_weight=candidate.lightgbm_weight,
        )
    if candidate.model_kind == "ridge_catboost_blend":
        return WeightedRidgeCatBoostCandidateModel(
            train_df,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=candidate.random_seed,
            ridge_weight=candidate.ridge_weight,
            catboost_weight=candidate.catboost_weight,
            catboost_params=candidate.catboost_params,
        )
    if candidate.model_kind == "weighted_e8_catboost_blend":
        return WeightedRidgeLightGBMCatBoostCandidateModel(
            train_df,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=candidate.random_seed,
            nested_cv=candidate.lightgbm_nested_cv,
            inner_folds=candidate.lightgbm_inner_folds,
            ridge_weight=candidate.ridge_weight,
            lightgbm_weight=candidate.lightgbm_weight,
            catboost_weight=candidate.catboost_weight,
            catboost_params=candidate.catboost_params,
        )
    msg = f"unsupported candidate model kind: {candidate.model_kind}"
    raise ValueError(msg)


def zscore_scores(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0 or not np.isfinite(std):
        return np.zeros(len(values), dtype=float)
    return (values - float(np.mean(values))) / std


def scale_scores(multiplier: float) -> ScoreTransform:
    def transform(values: np.ndarray) -> np.ndarray:
        return values * multiplier

    return transform


def zscore_scale_scores(multiplier: float) -> ScoreTransform:
    def transform(values: np.ndarray) -> np.ndarray:
        return zscore_scores(values) * multiplier

    return transform


def rank_pct_scores(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


MOMENTUM_RETURN_FEATURES = (
    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "momentum_252d",
    "momentum_21d_rank",
    "momentum_63d_rank",
    "momentum_126d_rank",
    "momentum_252d_rank",
    "return_5d",
    "return_21d",
    "return_21d_excess",
)

MOMENTUM_RISK_FEATURES = (
    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "momentum_252d",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
    "max_drawdown_63d",
    "max_drawdown_252d",
    "return_5d",
    "return_21d",
    "return_21d_excess",
)


CANDIDATES: dict[str, CandidateSpec] = {
    "e8_baseline": CandidateSpec(
        candidate_id="e8_baseline",
        description="Current Phase 2 E8 Ridge plus LightGBM regression blend.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
    ),
    "e8_scale_0p5": CandidateSpec(
        candidate_id="e8_scale_0p5",
        description="E8 blend with forecast scores scaled down by 0.5 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.5),
    ),
    "e8_scale_0p7": CandidateSpec(
        candidate_id="e8_scale_0p7",
        description="E8 blend with forecast scores scaled down by 0.7 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.7),
    ),
    "e8_scale_0p8": CandidateSpec(
        candidate_id="e8_scale_0p8",
        description="E8 blend with forecast scores scaled down by 0.8 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.8),
    ),
    "e8_scale_0p82": CandidateSpec(
        candidate_id="e8_scale_0p82",
        description="E8 blend with forecast scores scaled down by 0.82 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.82),
    ),
    "e8_scale_0p83": CandidateSpec(
        candidate_id="e8_scale_0p83",
        description="E8 blend with forecast scores scaled down by 0.83 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.83),
    ),
    "e8_scale_0p84": CandidateSpec(
        candidate_id="e8_scale_0p84",
        description="E8 blend with forecast scores scaled down by 0.84 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.84),
    ),
    "e8_scale_0p85": CandidateSpec(
        candidate_id="e8_scale_0p85",
        description="E8 blend with forecast scores scaled down by 0.85 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.85),
    ),
    "e8_scale_0p86": CandidateSpec(
        candidate_id="e8_scale_0p86",
        description="E8 blend with forecast scores scaled down by 0.86 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.86),
    ),
    "e8_scale_0p87": CandidateSpec(
        candidate_id="e8_scale_0p87",
        description="E8 blend with forecast scores scaled down by 0.87 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.87),
    ),
    "e8_scale_0p88": CandidateSpec(
        candidate_id="e8_scale_0p88",
        description="E8 blend with forecast scores scaled down by 0.88 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.88),
    ),
    "e8_scale_0p9": CandidateSpec(
        candidate_id="e8_scale_0p9",
        description="E8 blend with forecast scores scaled down by 0.9 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.9),
    ),
    "e8_scale_0p95": CandidateSpec(
        candidate_id="e8_scale_0p95",
        description="E8 blend with forecast scores scaled down by 0.95 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.95),
    ),
    "e8_scale_1p2": CandidateSpec(
        candidate_id="e8_scale_1p2",
        description="E8 blend with forecast scores scaled up by 1.2 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(1.2),
    ),
    "e8_scale_1p5": CandidateSpec(
        candidate_id="e8_scale_1p5",
        description="E8 blend with forecast scores scaled up by 1.5 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(1.5),
    ),
    "e8_scale_2p0": CandidateSpec(
        candidate_id="e8_scale_2p0",
        description="E8 blend with forecast scores scaled up by 2.0 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(2.0),
    ),
    "e8_scale_3p0": CandidateSpec(
        candidate_id="e8_scale_3p0",
        description="E8 blend with forecast scores scaled up by 3.0 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(3.0),
    ),
    "e8_scale_4p0": CandidateSpec(
        candidate_id="e8_scale_4p0",
        description="E8 blend with forecast scores scaled up by 4.0 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(4.0),
    ),
    "e8_rank_pct": CandidateSpec(
        candidate_id="e8_rank_pct",
        description="E8 blend converted to percentile ranks before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=rank_pct_scores,
    ),
    "e8_momentum_return": CandidateSpec(
        candidate_id="e8_momentum_return",
        description="E8 blend using momentum, rank, and return features without liquidity.",
        model_kind="e8_blend",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        lightgbm_nested_cv=False,
    ),
    "e8_momentum_risk": CandidateSpec(
        candidate_id="e8_momentum_risk",
        description="E8 blend using momentum, risk, drawdown, and short return features.",
        model_kind="e8_blend",
        horizon_days=5,
        feature_columns=MOMENTUM_RISK_FEATURES,
        lightgbm_nested_cv=False,
    ),
    "catboost_return_zscore": CandidateSpec(
        candidate_id="catboost_return_zscore",
        description="CatBoost return regression with z-scored forecast scores.",
        model_kind="catboost_regression",
        horizon_days=5,
        score_transform=zscore_scores,
    ),
    "catboost_return_zscore_deep": CandidateSpec(
        candidate_id="catboost_return_zscore_deep",
        description="CatBoost return regression with the deeper grid profile and z-scored scores.",
        model_kind="catboost_regression",
        horizon_days=5,
        catboost_params=CATBOOST_PARAM_GRID[1],
        score_transform=zscore_scores,
    ),
    "catboost_return_scale_0p85": CandidateSpec(
        candidate_id="catboost_return_scale_0p85",
        description="CatBoost return regression with z-scored scores scaled by 0.85.",
        model_kind="catboost_regression",
        horizon_days=5,
        score_transform=zscore_scale_scores(0.85),
    ),
    "catboost_return_scale_1p20": CandidateSpec(
        candidate_id="catboost_return_scale_1p20",
        description="CatBoost return regression with z-scored scores scaled by 1.20.",
        model_kind="catboost_regression",
        horizon_days=5,
        score_transform=zscore_scale_scores(1.20),
    ),
    "catboost_rank_target_zscore": CandidateSpec(
        candidate_id="catboost_rank_target_zscore",
        description="CatBoost regression trained on 5-day forward ranks with z-scored scores.",
        model_kind="catboost_regression",
        horizon_days=5,
        training_target_column="fwd_rank_5d",
        score_transform=zscore_scores,
    ),
    "catboost_top_tercile_zscore": CandidateSpec(
        candidate_id="catboost_top_tercile_zscore",
        description="CatBoost top-tercile classifier with z-scored probabilities.",
        model_kind="catboost_classification",
        horizon_days=5,
        training_target_column="fwd_is_top_tercile_5d",
        score_transform=zscore_scores,
    ),
    "catboost_momentum_return_zscore": CandidateSpec(
        candidate_id="catboost_momentum_return_zscore",
        description="CatBoost return regression using momentum, rank, and return features.",
        model_kind="catboost_regression",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        score_transform=zscore_scores,
    ),
    "catboost_momentum_risk_zscore": CandidateSpec(
        candidate_id="catboost_momentum_risk_zscore",
        description="CatBoost return regression using momentum, risk, drawdown, and short returns.",
        model_kind="catboost_regression",
        horizon_days=5,
        feature_columns=MOMENTUM_RISK_FEATURES,
        score_transform=zscore_scores,
    ),
    "torch_mlp_return_zscore": CandidateSpec(
        candidate_id="torch_mlp_return_zscore",
        description="PyTorch MLP return regression with z-scored forecast scores.",
        model_kind="torch_mlp_regression",
        horizon_days=5,
        score_transform=zscore_scores,
    ),
    "torch_mlp_return_deep_zscore": CandidateSpec(
        candidate_id="torch_mlp_return_deep_zscore",
        description="PyTorch MLP return regression with deeper params and z-scored scores.",
        model_kind="torch_mlp_regression",
        horizon_days=5,
        torch_params=TORCH_MLP_PARAM_GRID[1],
        score_transform=zscore_scores,
    ),
    "torch_mlp_momentum_return_zscore": CandidateSpec(
        candidate_id="torch_mlp_momentum_return_zscore",
        description="PyTorch MLP return regression using momentum, rank, and return features.",
        model_kind="torch_mlp_regression",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        score_transform=zscore_scores,
    ),
    "torch_mlp_momentum_return_deep_zscore": CandidateSpec(
        candidate_id="torch_mlp_momentum_return_deep_zscore",
        description=(
            "PyTorch MLP return regression using momentum/return features and deeper params."
        ),
        model_kind="torch_mlp_regression",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        torch_params=TORCH_MLP_PARAM_GRID[1],
        score_transform=zscore_scores,
    ),
    "torch_lstm_return_zscore": CandidateSpec(
        candidate_id="torch_lstm_return_zscore",
        description="PyTorch LSTM return regression over per-ticker feature windows.",
        model_kind="torch_lstm_regression",
        horizon_days=5,
        score_transform=zscore_scores,
    ),
    "torch_lstm_return_deep_zscore": CandidateSpec(
        candidate_id="torch_lstm_return_deep_zscore",
        description="PyTorch LSTM return regression with longer lookback and wider hidden state.",
        model_kind="torch_lstm_regression",
        horizon_days=5,
        torch_params=TORCH_SEQUENCE_PARAM_GRID[1],
        score_transform=zscore_scores,
    ),
    "torch_lstm_momentum_return_zscore": CandidateSpec(
        candidate_id="torch_lstm_momentum_return_zscore",
        description="PyTorch LSTM return regression using momentum, rank, and return features.",
        model_kind="torch_lstm_regression",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        score_transform=zscore_scores,
    ),
    "torch_transformer_return_zscore": CandidateSpec(
        candidate_id="torch_transformer_return_zscore",
        description="PyTorch Transformer encoder return regression over feature windows.",
        model_kind="torch_transformer_regression",
        horizon_days=5,
        score_transform=zscore_scores,
    ),
    "torch_transformer_return_deep_zscore": CandidateSpec(
        candidate_id="torch_transformer_return_deep_zscore",
        description=(
            "PyTorch Transformer encoder return regression with longer lookback and wider "
            "hidden state."
        ),
        model_kind="torch_transformer_regression",
        horizon_days=5,
        torch_params=TORCH_SEQUENCE_PARAM_GRID[1],
        score_transform=zscore_scores,
    ),
    "torch_transformer_momentum_return_zscore": CandidateSpec(
        candidate_id="torch_transformer_momentum_return_zscore",
        description=(
            "PyTorch Transformer encoder return regression using momentum, rank, and return "
            "features."
        ),
        model_kind="torch_transformer_regression",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        score_transform=zscore_scores,
    ),
    "ridge_catboost_1p0_1p0_scale_1p00": CandidateSpec(
        candidate_id="ridge_catboost_1p0_1p0_scale_1p00",
        description="Ridge plus CatBoost weighted blend with 1.0/1.0 weights and 1.00 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.0,
        catboost_weight=1.0,
        score_transform=scale_scores(1.00),
    ),
    "ridge_catboost_1p2_0p8_scale_1p00": CandidateSpec(
        candidate_id="ridge_catboost_1p2_0p8_scale_1p00",
        description="Ridge plus CatBoost weighted blend with 1.2/0.8 weights and 1.00 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.2,
        catboost_weight=0.8,
        score_transform=scale_scores(1.00),
    ),
    "ridge_catboost_0p8_1p2_scale_1p00": CandidateSpec(
        candidate_id="ridge_catboost_0p8_1p2_scale_1p00",
        description="Ridge plus CatBoost weighted blend with 0.8/1.2 weights and 1.00 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=0.8,
        catboost_weight=1.2,
        score_transform=scale_scores(1.00),
    ),
    "ridge_catboost_1p0_1p0_scale_1p20": CandidateSpec(
        candidate_id="ridge_catboost_1p0_1p0_scale_1p20",
        description="Ridge plus CatBoost weighted blend with 1.0/1.0 weights and 1.20 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.0,
        catboost_weight=1.0,
        score_transform=scale_scores(1.20),
    ),
    "ridge_catboost_1p2_0p8_scale_1p20": CandidateSpec(
        candidate_id="ridge_catboost_1p2_0p8_scale_1p20",
        description="Ridge plus CatBoost weighted blend with 1.2/0.8 weights and 1.20 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.2,
        catboost_weight=0.8,
        score_transform=scale_scores(1.20),
    ),
    "ridge_catboost_1p5_0p5_scale_1p00": CandidateSpec(
        candidate_id="ridge_catboost_1p5_0p5_scale_1p00",
        description="Ridge plus CatBoost weighted blend with 1.5/0.5 weights and 1.00 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.5,
        catboost_weight=0.5,
        score_transform=scale_scores(1.00),
    ),
    "ridge_catboost_1p5_0p5_scale_1p20": CandidateSpec(
        candidate_id="ridge_catboost_1p5_0p5_scale_1p20",
        description="Ridge plus CatBoost weighted blend with 1.5/0.5 weights and 1.20 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.5,
        catboost_weight=0.5,
        score_transform=scale_scores(1.20),
    ),
    "ridge_catboost_1p5_0p5_scale_1p30": CandidateSpec(
        candidate_id="ridge_catboost_1p5_0p5_scale_1p30",
        description="Ridge plus CatBoost weighted blend with 1.5/0.5 weights and 1.30 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.5,
        catboost_weight=0.5,
        score_transform=scale_scores(1.30),
    ),
    "ridge_catboost_1p5_0p5_scale_1p50": CandidateSpec(
        candidate_id="ridge_catboost_1p5_0p5_scale_1p50",
        description="Ridge plus CatBoost weighted blend with 1.5/0.5 weights and 1.50 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.5,
        catboost_weight=0.5,
        score_transform=scale_scores(1.50),
    ),
    "ridge_catboost_1p5_0p5_scale_1p80": CandidateSpec(
        candidate_id="ridge_catboost_1p5_0p5_scale_1p80",
        description="Ridge plus CatBoost weighted blend with 1.5/0.5 weights and 1.80 scale.",
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.5,
        catboost_weight=0.5,
        score_transform=scale_scores(1.80),
    ),
    "ridge_catboost_1p5_0p5_deep_scale_1p20": CandidateSpec(
        candidate_id="ridge_catboost_1p5_0p5_deep_scale_1p20",
        description=(
            "Ridge plus deeper CatBoost weighted blend with 1.5/0.5 weights and 1.20 scale."
        ),
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        ridge_weight=1.5,
        catboost_weight=0.5,
        catboost_params=CATBOOST_PARAM_GRID[1],
        score_transform=scale_scores(1.20),
    ),
    "ridge_catboost_momentum_return_1p5_0p5_scale_1p20": CandidateSpec(
        candidate_id="ridge_catboost_momentum_return_1p5_0p5_scale_1p20",
        description=(
            "Ridge plus CatBoost blend using momentum/return features, 1.5/0.5 weights, and "
            "1.20 scale."
        ),
        model_kind="ridge_catboost_blend",
        horizon_days=5,
        feature_columns=MOMENTUM_RETURN_FEATURES,
        ridge_weight=1.5,
        catboost_weight=0.5,
        score_transform=scale_scores(1.20),
    ),
    "e8_catboost_1p2_0p8_0p2_scale_1p20": CandidateSpec(
        candidate_id="e8_catboost_1p2_0p8_0p2_scale_1p20",
        description=(
            "Ridge, LightGBM, and CatBoost blend with 1.2/0.8/0.2 weights and 1.20 scale."
        ),
        model_kind="weighted_e8_catboost_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        catboost_weight=0.2,
        score_transform=scale_scores(1.20),
    ),
    "e8_catboost_1p2_0p8_0p5_scale_1p20": CandidateSpec(
        candidate_id="e8_catboost_1p2_0p8_0p5_scale_1p20",
        description=(
            "Ridge, LightGBM, and CatBoost blend with 1.2/0.8/0.5 weights and 1.20 scale."
        ),
        model_kind="weighted_e8_catboost_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        catboost_weight=0.5,
        score_transform=scale_scores(1.20),
    ),
    "e8_catboost_1p3_0p7_0p2_scale_1p20": CandidateSpec(
        candidate_id="e8_catboost_1p3_0p7_0p2_scale_1p20",
        description=(
            "Ridge, LightGBM, and CatBoost blend with 1.3/0.7/0.2 weights and 1.20 scale."
        ),
        model_kind="weighted_e8_catboost_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.3,
        lightgbm_weight=0.7,
        catboost_weight=0.2,
        score_transform=scale_scores(1.20),
    ),
    "e8_catboost_1p2_0p8_0p2_scale_1p30": CandidateSpec(
        candidate_id="e8_catboost_1p2_0p8_0p2_scale_1p30",
        description=(
            "Ridge, LightGBM, and CatBoost blend with 1.2/0.8/0.2 weights and 1.30 scale."
        ),
        model_kind="weighted_e8_catboost_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        catboost_weight=0.2,
        score_transform=scale_scores(1.30),
    ),
    "e8_weight_ridge_1p25_lgbm_0p75_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p25_lgbm_0p75_scale_0p85",
        description="E8 weighted blend with 1.25 Ridge, 0.75 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.25,
        lightgbm_weight=0.75,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p15_lgbm_0p85_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p15_lgbm_0p85_scale_0p85",
        description="E8 weighted blend with 1.15 Ridge, 0.85 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.15,
        lightgbm_weight=0.85,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p85",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p18_lgbm_0p82_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p18_lgbm_0p82_scale_0p85",
        description="E8 weighted blend with 1.18 Ridge, 0.82 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.18,
        lightgbm_weight=0.82,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p19_lgbm_0p81_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p19_lgbm_0p81_scale_0p85",
        description="E8 weighted blend with 1.19 Ridge, 0.81 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.19,
        lightgbm_weight=0.81,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p21_lgbm_0p79_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p21_lgbm_0p79_scale_0p85",
        description="E8 weighted blend with 1.21 Ridge, 0.79 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.21,
        lightgbm_weight=0.79,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p22_lgbm_0p78_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p22_lgbm_0p78_scale_0p85",
        description="E8 weighted blend with 1.22 Ridge, 0.78 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.22,
        lightgbm_weight=0.78,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p82": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p82",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.82 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.82),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p86": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p86",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.86 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.86),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p87": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p87",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.87 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.87),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p88": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p88",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.88 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.88),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p89": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p89",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.89 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.89),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p90": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p90",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.90 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.90),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p92": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p92",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.92 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.92),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p94": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p94",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.94 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.94),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p96": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p96",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.96 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.96),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_0p98": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_0p98",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 0.98 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(0.98),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p00": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p00",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.00 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.00),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p05": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p05",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.05 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.05),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p10": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p10",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.10 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.10),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p15": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p15",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.15 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.15),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p20": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p20",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.20 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.20),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p30": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p30",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.30 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.30),
    ),
    "e8_weight_ridge_1p2_lgbm_0p8_scale_1p50": CandidateSpec(
        candidate_id="e8_weight_ridge_1p2_lgbm_0p8_scale_1p50",
        description="E8 weighted blend with 1.2 Ridge, 0.8 LightGBM, and 1.50 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.2,
        lightgbm_weight=0.8,
        score_transform=scale_scores(1.50),
    ),
    "e8_weight_ridge_1p3_lgbm_0p7_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p3_lgbm_0p7_scale_0p85",
        description="E8 weighted blend with 1.3 Ridge, 0.7 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.3,
        lightgbm_weight=0.7,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p35_lgbm_0p65_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p35_lgbm_0p65_scale_0p85",
        description="E8 weighted blend with 1.35 Ridge, 0.65 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.35,
        lightgbm_weight=0.65,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_1p5_lgbm_0p5_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_1p5_lgbm_0p5_scale_0p85",
        description="E8 weighted blend with 1.5 Ridge, 0.5 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=1.5,
        lightgbm_weight=0.5,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_0p75_lgbm_1p25_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_0p75_lgbm_1p25_scale_0p85",
        description="E8 weighted blend with 0.75 Ridge, 1.25 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=0.75,
        lightgbm_weight=1.25,
        score_transform=scale_scores(0.85),
    ),
    "e8_weight_ridge_0p5_lgbm_1p5_scale_0p85": CandidateSpec(
        candidate_id="e8_weight_ridge_0p5_lgbm_1p5_scale_0p85",
        description="E8 weighted blend with 0.5 Ridge, 1.5 LightGBM, and 0.85 score scale.",
        model_kind="weighted_e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        ridge_weight=0.5,
        lightgbm_weight=1.5,
        score_transform=scale_scores(0.85),
    ),
    "ridge_return_zscore": CandidateSpec(
        candidate_id="ridge_return_zscore",
        description="Ridge return model with z-scored forecast scores.",
        model_kind="ridge",
        horizon_days=5,
        score_transform=zscore_scores,
    ),
    "lightgbm_return_zscore": CandidateSpec(
        candidate_id="lightgbm_return_zscore",
        description="LightGBM return model with z-scored forecast scores.",
        model_kind="lightgbm_regression",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=zscore_scores,
    ),
    "lightgbm_rank": CandidateSpec(
        candidate_id="lightgbm_rank",
        description="LightGBM LambdaRank model on 5-day forward ranks.",
        model_kind="lightgbm_rank",
        horizon_days=5,
        training_target_column="fwd_rank_5d",
        lightgbm_nested_cv=False,
        score_transform=zscore_scores,
    ),
    "ridge_rank": CandidateSpec(
        candidate_id="ridge_rank",
        description="Ridge model on rank-normalized features, matching the Phase 2 E4 family.",
        model_kind="ridge_rank",
        horizon_days=5,
        training_target_column="fwd_rank_5d",
    ),
}
