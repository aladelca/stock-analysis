from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.config import OptimizerConfig
from stock_analysis.optimization.engine import OptimizationError, optimize_long_only


def test_optimize_long_only_respects_constraints() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "expected_return": [0.2, 0.1, 0.05],
            "eligible_for_optimization": [True, True, True],
        }
    )
    covariance = pd.DataFrame(
        [[0.1, 0.01, 0.0], [0.01, 0.2, 0.0], [0.0, 0.0, 0.15]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )

    weights = optimize_long_only(
        optimizer_input,
        covariance,
        OptimizerConfig(max_weight=0.6, risk_aversion=1.0),
    )

    assert weights.min() >= 0
    assert weights.max() <= 0.600001
    assert weights.sum() == pytest.approx(1.0)


def test_optimize_long_only_requires_enough_assets() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "expected_return": [0.2],
            "eligible_for_optimization": [True],
        }
    )
    covariance = pd.DataFrame([[0.1]], index=["AAA"], columns=["AAA"])

    with pytest.raises(OptimizationError):
        optimize_long_only(optimizer_input, covariance, OptimizerConfig(max_weight=0.5))


def test_turnover_penalty_keeps_weights_closer_to_previous_weights() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "expected_return": [0.3, 0.2, 0.1, 0.0],
            "eligible_for_optimization": [True, True, True, True],
        }
    )
    covariance = pd.DataFrame(
        [[0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.2]],
        index=["AAA", "BBB", "CCC", "DDD"],
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    previous = pd.Series({"AAA": 0.0, "BBB": 0.0, "CCC": 0.5, "DDD": 0.5})

    no_penalty = optimize_long_only(
        optimizer_input,
        covariance,
        OptimizerConfig(max_weight=0.6, risk_aversion=0.1, lambda_turnover=0, commission_rate=0),
        w_prev=previous,
    )
    high_penalty = optimize_long_only(
        optimizer_input,
        covariance,
        OptimizerConfig(max_weight=0.6, risk_aversion=0.1, lambda_turnover=10, commission_rate=0),
        w_prev=previous,
    )

    no_penalty_turnover = (
        0.5 * (no_penalty - previous.reindex(no_penalty.index).fillna(0)).abs().sum()
    )
    high_penalty_turnover = (
        0.5 * (high_penalty - previous.reindex(high_penalty.index).fillna(0)).abs().sum()
    )

    assert high_penalty_turnover < no_penalty_turnover


def test_commission_penalty_keeps_weights_closer_to_previous_weights() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "expected_return": [0.3, 0.2, 0.1, 0.0],
            "eligible_for_optimization": [True, True, True, True],
        }
    )
    covariance = pd.DataFrame(
        [
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ],
        index=["AAA", "BBB", "CCC", "DDD"],
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    previous = pd.Series({"AAA": 0.0, "BBB": 0.0, "CCC": 0.5, "DDD": 0.5})

    no_commission = optimize_long_only(
        optimizer_input,
        covariance,
        OptimizerConfig(max_weight=0.6, risk_aversion=0.1, lambda_turnover=0, commission_rate=0),
        w_prev=previous,
    )
    high_commission = optimize_long_only(
        optimizer_input,
        covariance,
        OptimizerConfig(max_weight=0.6, risk_aversion=0.1, lambda_turnover=0, commission_rate=1),
        w_prev=previous,
    )

    no_commission_trade = (no_commission - previous.reindex(no_commission.index).fillna(0)).abs()
    high_commission_trade = (
        high_commission - previous.reindex(high_commission.index).fillna(0)
    ).abs()

    assert high_commission_trade.sum() < no_commission_trade.sum()


def test_sector_cap_limits_aggregate_sector_weight() -> None:
    optimizer_input = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "expected_return": [0.4, 0.3, 0.1, 0.0],
            "eligible_for_optimization": [True, True, True, True],
            "gics_sector": ["Technology", "Technology", "Health Care", "Health Care"],
        }
    )
    covariance = pd.DataFrame(
        0.0,
        index=["AAA", "BBB", "CCC", "DDD"],
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    for ticker in covariance.index:
        covariance.loc[ticker, ticker] = 0.2

    weights = optimize_long_only(
        optimizer_input,
        covariance,
        OptimizerConfig(
            max_weight=0.8,
            risk_aversion=0.1,
            commission_rate=0,
            sector_max_weight=0.5,
        ),
    )
    tech_weight = weights[["AAA", "BBB"]].sum()

    assert tech_weight <= 0.500001
    assert weights.sum() == pytest.approx(1.0)
