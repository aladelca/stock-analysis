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


CANDIDATES: dict[str, CandidateSpec] = {
    "e8_baseline": CandidateSpec(
        candidate_id="e8_baseline",
        description="Current Phase 2 E8 Ridge plus LightGBM regression blend.",
        model_kind="e8_blend",
        horizon_days=5,
        lightgbm_nested_cv=False,
    ),
    "ridge_rank": CandidateSpec(
        candidate_id="ridge_rank",
        description="Ridge model on rank-normalized features, matching the Phase 2 E4 family.",
        model_kind="ridge_rank",
        horizon_days=5,
        training_target_column="fwd_rank_5d",
    ),
}
