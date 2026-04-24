from __future__ import annotations

import pandas as pd

from stock_analysis.config import OptimizerConfig
from stock_analysis.optimization.recommendations import (
    build_recommendations,
    build_risk_metrics,
    build_sector_exposure,
)


def test_recommendation_outputs() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "security": ["AAA Corp", "BBB Inc"],
            "gics_sector": ["Technology", "Health Care"],
            "expected_return": [0.1, 0.2],
            "volatility": [0.2, 0.3],
            "eligible_for_optimization": [True, True],
        }
    )
    weights = pd.Series([0.4, 0.6], index=["AAA", "BBB"])
    covariance = pd.DataFrame(
        [[0.1, 0.0], [0.0, 0.2]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    recommendations = build_recommendations(
        optimizer_input,
        weights,
        OptimizerConfig(max_weight=0.6),
        "2026-04-24",
        "run-1",
    )
    risk = build_risk_metrics(optimizer_input, covariance, weights, "2026-04-24", "run-1")
    sectors = build_sector_exposure(optimizer_input, weights, "2026-04-24", "run-1")

    assert recommendations["target_weight"].sum() == 1.0
    assert set(recommendations["action"]) == {"BUY"}
    assert set(risk["metric"]).issuperset({"expected_return", "expected_volatility"})
    assert sectors["target_weight"].sum() == 1.0
