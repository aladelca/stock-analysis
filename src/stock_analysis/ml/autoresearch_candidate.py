from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from stock_analysis.ml.phase2 import (
    DEFAULT_FEATURE_CANDIDATES,
    BlendedForecastModel,
    LightGBMForecastModel,
    RidgeForecastModel,
)


class CandidateModel(Protocol):
    def predict(self, features: pd.DataFrame) -> Sequence[float]:
        """Return forecast scores for a rebalance cross-section."""


ScoreTransform = Callable[[np.ndarray], np.ndarray]
ModelKind = Literal["ridge", "ridge_rank", "lightgbm_regression", "lightgbm_rank", "e8_blend"]


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
    if candidate.model_kind == "e8_blend":
        return BlendedForecastModel(
            train_df,
            feature_columns=feature_columns,
            return_target_column=target_column,
            random_seed=candidate.random_seed,
            nested_cv=candidate.lightgbm_nested_cv,
            inner_folds=candidate.lightgbm_inner_folds,
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
    "e8_scale_0p85": CandidateSpec(
        candidate_id="e8_scale_0p85",
        description="E8 blend with forecast scores scaled down by 0.85 before optimization.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
        score_transform=scale_scores(0.85),
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
