from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.config import OptimizerConfig
from stock_analysis.optimization.recommendations import (
    build_recommendations,
    build_risk_metrics,
    build_sector_exposure,
)
from stock_analysis.portfolio.holdings import PortfolioState
from stock_analysis.portfolio.rebalance import build_rebalance_context


def test_recommendation_outputs() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "security": ["AAA Corp", "BBB Inc"],
            "gics_sector": ["Technology", "Health Care"],
            "forecast_score": [1.0, 2.0],
            "expected_return": [0.1, 0.2],
            "expected_return_is_calibrated": [False, False],
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
    assert recommendations.set_index("ticker").loc["AAA", "forecast_score"] == pytest.approx(1.0)
    assert pd.isna(recommendations.set_index("ticker").loc["AAA", "calibrated_expected_return"])
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


def test_recommendations_include_contribution_dollar_fields() -> None:
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
    weights = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
    context = build_rebalance_context(
        PortfolioState(
            weights=pd.Series({"AAA": 0.5, "BBB": 0.3}),
            market_values=pd.Series({"AAA": 500.0, "BBB": 300.0}),
            cash_balance=200.0,
            portfolio_value=1000.0,
        ),
        ["AAA", "BBB"],
        contribution_amount=100.0,
    )

    recommendations = build_recommendations(
        optimizer_input,
        weights,
        OptimizerConfig(max_weight=0.6, commission_rate=0.02),
        "2026-04-24",
        "run-1",
        rebalance_context=context,
    )

    by_ticker = recommendations.set_index("ticker")
    assert by_ticker.loc["BBB", "action"] == "BUY"
    assert by_ticker.loc["BBB", "trade_notional"] == pytest.approx(245.0980392157)
    assert by_ticker.loc["BBB", "commission_amount"] == pytest.approx(4.9019607843)
    assert by_ticker.loc["BBB", "portfolio_value_after_contribution"] == pytest.approx(1100)
    assert by_ticker.loc["BBB", "deposit_used_amount"] > 0
    assert by_ticker["cash_after_trade_amount"].iloc[0] >= -1e-9


def test_recommendations_use_outside_holding_sale_proceeds_for_buys() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security": ["AAA Corp"],
            "gics_sector": ["Technology"],
            "expected_return": [0.1],
            "volatility": [0.2],
            "eligible_for_optimization": [True],
        }
    )
    weights = pd.Series([1.0], index=["AAA"])
    context = build_rebalance_context(
        PortfolioState(
            weights=pd.Series({"SPY": 1.0}),
            market_values=pd.Series({"SPY": 300.0}),
            cash_balance=0.0,
            portfolio_value=300.0,
        ),
        ["AAA", "SPY"],
    )

    recommendations = build_recommendations(
        optimizer_input,
        weights,
        OptimizerConfig(max_weight=1.0, commission_rate=0.02),
        "2026-04-27",
        "run-1",
        rebalance_context=context,
    )

    by_ticker = recommendations.set_index("ticker")
    assert by_ticker.loc["SPY", "action"] == "SELL"
    assert by_ticker.loc["SPY", "trade_notional"] == pytest.approx(-300.0)
    assert by_ticker.loc["SPY", "reason_code"] == "current_holding_outside_optimizer_universe"
    assert by_ticker.loc["AAA", "action"] == "BUY"
    assert by_ticker.loc["AAA", "trade_notional"] == pytest.approx(288.2352941176)
    assert by_ticker.loc["AAA", "commission_amount"] == pytest.approx(5.7647058824)
    assert by_ticker.loc["AAA", "cash_after_trade_amount"] == pytest.approx(0.0)


def test_recommendations_can_preserve_outside_holding_and_use_deposit_for_buys() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security": ["AAA Corp"],
            "gics_sector": ["Technology"],
            "expected_return": [0.1],
            "volatility": [0.2],
            "eligible_for_optimization": [True],
        }
    )
    weights = pd.Series([0.5], index=["AAA"])
    context = build_rebalance_context(
        PortfolioState(
            weights=pd.Series({"SPY": 0.5}),
            market_values=pd.Series({"SPY": 300.0}),
            cash_balance=0.0,
            portfolio_value=300.0,
        ),
        ["AAA", "SPY"],
        contribution_amount=300.0,
    )

    recommendations = build_recommendations(
        optimizer_input,
        weights,
        OptimizerConfig(max_weight=1.0, commission_rate=0.02),
        "2026-04-28",
        "run-1",
        rebalance_context=context,
        preserve_outside_holdings=True,
    )

    by_ticker = recommendations.set_index("ticker")
    assert by_ticker.loc["SPY", "action"] == "HOLD"
    assert by_ticker.loc["SPY", "target_weight"] == pytest.approx(0.5)
    assert by_ticker.loc["SPY", "trade_notional"] == pytest.approx(0.0)
    assert by_ticker.loc["AAA", "action"] == "BUY"
    assert by_ticker.loc["AAA", "trade_notional"] == pytest.approx(294.1176470588)
    assert by_ticker.loc["AAA", "commission_amount"] == pytest.approx(5.8823529412)
    assert by_ticker.loc["AAA", "deposit_used_amount"] == pytest.approx(300.0)


def test_no_trade_band_converts_small_trades_to_hold() -> None:
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
    weights = pd.Series([0.51, 0.49], index=["AAA", "BBB"])

    recommendations = build_recommendations(
        optimizer_input,
        weights,
        OptimizerConfig(max_weight=0.6, min_rebalance_trade_weight=0.005),
        "2026-04-24",
        "run-1",
        current_weights=pd.Series({"AAA": 0.5, "BBB": 0.5}),
        no_trade_band=0.02,
    )

    assert set(recommendations["action"]) == {"HOLD"}
    assert recommendations["no_trade_band_applied"].all()
