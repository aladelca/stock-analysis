from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from stock_analysis.config import OptimizerConfig
from stock_analysis.domain.schemas import validate_columns


def build_recommendations(
    optimizer_input: pd.DataFrame,
    weights: pd.Series,
    config: OptimizerConfig,
    as_of_date: str,
    run_id: str,
) -> pd.DataFrame:
    result = optimizer_input.copy()
    result["target_weight"] = result["ticker"].map(weights).fillna(0.0).astype(float)
    result["action"] = np.where(
        result["target_weight"] >= config.min_trade_weight, "BUY", "EXCLUDE"
    )
    result["reason_code"] = np.where(
        result["eligible_for_optimization"].astype(bool),
        "selected_by_optimizer",
        "ineligible_data_quality",
    )
    result.loc[result["target_weight"] < config.min_trade_weight, "reason_code"] = (
        "not_selected_or_below_threshold"
    )
    result["as_of_date"] = as_of_date
    result["run_id"] = run_id
    columns = [
        "ticker",
        "security",
        "gics_sector",
        "expected_return",
        "volatility",
        "target_weight",
        "action",
        "reason_code",
        "as_of_date",
        "run_id",
    ]
    return validate_columns(
        result[columns].sort_values("target_weight", ascending=False), "portfolio_recommendations"
    )


def build_risk_metrics(
    optimizer_input: pd.DataFrame,
    covariance: pd.DataFrame,
    weights: pd.Series,
    as_of_date: str,
    run_id: str,
) -> pd.DataFrame:
    selected = optimizer_input.set_index("ticker").reindex(weights.index)
    mu = selected["expected_return"].astype(float).to_numpy()
    cov = (
        covariance.reindex(index=weights.index, columns=weights.index)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    w = weights.to_numpy(dtype=float)
    portfolio_return = float(w @ mu)
    portfolio_volatility = float(sqrt(max(w @ cov @ w, 0)))
    rows = [
        ("expected_return", portfolio_return),
        ("expected_volatility", portfolio_volatility),
        ("num_holdings", float((weights > 0).sum())),
        ("max_weight", float(weights.max())),
        ("concentration_hhi", float(np.square(w).sum())),
    ]
    result = pd.DataFrame(rows, columns=["metric", "value"])
    result["as_of_date"] = as_of_date
    result["run_id"] = run_id
    return validate_columns(result, "portfolio_risk_metrics")


def build_sector_exposure(
    optimizer_input: pd.DataFrame,
    weights: pd.Series,
    as_of_date: str,
    run_id: str,
) -> pd.DataFrame:
    joined = optimizer_input[["ticker", "gics_sector"]].copy()
    joined["target_weight"] = joined["ticker"].map(weights).fillna(0.0)
    result = joined.groupby("gics_sector", dropna=False)["target_weight"].sum().reset_index()
    result["as_of_date"] = as_of_date
    result["run_id"] = run_id
    return validate_columns(result.sort_values("target_weight", ascending=False), "sector_exposure")
