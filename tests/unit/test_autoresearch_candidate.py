from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ml.autoresearch_candidate import (
    CandidateSpec,
    TransformedCandidateModel,
    build_model_factory,
    candidate_ids,
    get_candidate,
    resolve_feature_columns,
)


class _ConstantModel:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, features: pd.DataFrame) -> list[float]:
        del features
        return self.values


def test_candidate_registry_contains_seeded_baseline() -> None:
    assert "e8_baseline" in candidate_ids()
    candidate = get_candidate("e8_baseline")

    assert candidate.model_kind == "e8_blend"
    assert candidate.horizon_days == 5


def test_resolve_feature_columns_uses_default_existing_columns() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": ["2026-01-01"],
            "momentum_21d": [0.1],
            "volatility_21d": [0.2],
        }
    )

    columns = resolve_feature_columns(panel, get_candidate("e8_baseline"))

    assert columns == ("momentum_21d", "volatility_21d")


def test_resolve_feature_columns_rejects_missing_explicit_columns() -> None:
    panel = pd.DataFrame({"ticker": ["AAA"], "date": ["2026-01-01"], "momentum_21d": [0.1]})
    candidate = CandidateSpec(
        candidate_id="missing_feature",
        description="Missing explicit feature",
        model_kind="ridge",
        feature_columns=("momentum_21d", "volatility_21d"),
    )

    with pytest.raises(ValueError, match="missing from the panel"):
        resolve_feature_columns(panel, candidate)


def test_transformed_candidate_model_preserves_prediction_count() -> None:
    model = TransformedCandidateModel(
        _ConstantModel([1.0, 2.0, 3.0]),
        lambda values: values - float(np.mean(values)),
    )

    result = model.predict(pd.DataFrame({"feature": [1, 2, 3]}))

    assert result == [-1.0, 0.0, 1.0]


def test_transformed_candidate_model_rejects_count_changes() -> None:
    model = TransformedCandidateModel(
        _ConstantModel([1.0, 2.0, 3.0]),
        lambda values: values[:2],
    )

    with pytest.raises(ValueError, match="changed the number of predictions"):
        model.predict(pd.DataFrame({"feature": [1, 2, 3]}))


def test_build_model_factory_returns_predictive_model() -> None:
    train = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "date": ["2026-01-01", "2026-01-01", "2026-01-08", "2026-01-08"],
            "momentum_21d": [0.1, -0.1, 0.2, -0.2],
            "volatility_21d": [0.2, 0.3, 0.2, 0.3],
            "fwd_return_5d": [0.03, -0.01, 0.04, -0.02],
        }
    )
    candidate = CandidateSpec(
        candidate_id="ridge_test",
        description="Ridge test",
        model_kind="ridge",
        feature_columns=("momentum_21d", "volatility_21d"),
    )

    factory = build_model_factory(candidate, candidate.feature_columns)
    model = factory(train)
    predictions = model.predict(train[["momentum_21d", "volatility_21d"]])

    assert len(predictions) == len(train)
    assert all(np.isfinite(predictions))


def test_build_model_factory_returns_catboost_candidate() -> None:
    train = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-08",
                "2026-01-08",
                "2026-01-08",
            ],
            "momentum_21d": [0.1, -0.1, 0.05, 0.2, -0.2, 0.08],
            "volatility_21d": [0.2, 0.3, 0.25, 0.2, 0.3, 0.25],
            "fwd_return_5d": [0.03, -0.01, 0.01, 0.04, -0.02, 0.02],
        }
    )
    candidate = CandidateSpec(
        candidate_id="catboost_test",
        description="CatBoost test",
        model_kind="catboost_regression",
        feature_columns=("momentum_21d", "volatility_21d"),
        catboost_params={
            "iterations": 5,
            "learning_rate": 0.1,
            "depth": 2,
            "l2_leaf_reg": 3.0,
        },
    )

    factory = build_model_factory(candidate, candidate.feature_columns)
    model = factory(train)
    predictions = model.predict(train[["momentum_21d", "volatility_21d"]])

    assert len(predictions) == len(train)
    assert all(np.isfinite(predictions))
