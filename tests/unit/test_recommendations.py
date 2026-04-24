from __future__ import annotations

import pandas as pd
import pytest

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
        OptimizerConfig(max_weight=0.6, commission_rate=0.02),
        "2026-04-24",
        "run-1",
        current_weights=pd.Series({"AAA": 0.401, "BBB": 0.55, "ZZZ": 0.05}),
    )
    risk = build_risk_metrics(optimizer_input, covariance, weights, "2026-04-24", "run-1")
    sectors = build_sector_exposure(optimizer_input, weights, "2026-04-24", "run-1")

    assert recommendations["target_weight"].sum() == 1.0
    assert {
        "current_weight",
        "trade_weight",
        "trade_abs_weight",
        "estimated_commission_weight",
        "rebalance_required",
    } <= set(recommendations.columns)
    assert set(recommendations["action"]) == {"BUY", "SELL", "HOLD"}
    zzz = recommendations.set_index("ticker").loc["ZZZ"]
    assert zzz["action"] == "SELL"
    assert zzz["trade_weight"] == pytest.approx(-0.05)
    assert zzz["estimated_commission_weight"] == pytest.approx(0.001)
    aaa = recommendations.set_index("ticker").loc["AAA"]
    assert aaa["action"] == "HOLD"
    bbb = recommendations.set_index("ticker").loc["BBB"]
    assert bbb["action"] == "BUY"
    assert set(risk["metric"]).issuperset({"expected_return", "expected_volatility"})
    assert sectors["target_weight"].sum() == 1.0


def test_recommendation_hold_and_exclude_actions() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "security": ["AAA Corp", "BBB Inc", "CCC LLC"],
            "gics_sector": ["Technology", "Health Care", "Utilities"],
            "expected_return": [0.1, 0.2, 0.0],
            "volatility": [0.2, 0.3, 0.4],
            "eligible_for_optimization": [True, True, False],
        }
    )
    weights = pd.Series([0.401, 0.599], index=["AAA", "BBB"])

    recommendations = build_recommendations(
        optimizer_input,
        weights,
        OptimizerConfig(max_weight=0.6, min_rebalance_trade_weight=0.005),
        "2026-04-24",
        "run-1",
        current_weights=pd.Series({"AAA": 0.4, "BBB": 0.6}),
    )

    by_ticker = recommendations.set_index("ticker")
    assert by_ticker.loc["AAA", "action"] == "HOLD"
    assert by_ticker.loc["BBB", "action"] == "HOLD"
    assert by_ticker.loc["CCC", "action"] == "EXCLUDE"
